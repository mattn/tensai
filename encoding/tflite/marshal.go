// Package tflite marshals trained tensai Sequential models into the
// TensorFlow Lite FlatBuffers format (FP32, batch size 1), so they can run
// on the TFLite / LiteRT runtimes — including from Go via
// github.com/mattn/go-tflite (alias one of the packages when importing
// both, e.g. `tensaitflite "github.com/mattn/tensai/encoding/tflite"`).
//
// Layout note: TFLite convolutions are NHWC, so exported models take NHWC
// input (row index = (y*width + x)*channels + c) even though tensai's own
// Conv2D consumes channel-major rows. Conv and Dense weights are reordered
// during export; feed the exported model NHWC data.
//
// Supported layers: Dense, Conv2D (padding must amount to VALID or SAME),
// MaxPool2D, BatchNorm (folded to Mul+Add), Dropout (dropped), Softmax,
// and the ReLU / LeakyReLU / Sigmoid / Tanh activations.
package tflite

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	tensai "github.com/mattn/tensai"
)

// Constants below are taken from the TFLite schema (schema.fbs).
const (
	fileIdentifier = "TFL3"
	schemaVersion  = 3

	tensorFloat32 = 0
	tensorInt32   = 2

	paddingSame  = 0
	paddingValid = 1

	opAdd            = 0
	opConv2D         = 3
	opFullyConnected = 9
	opLogistic       = 14
	opMaxPool2D      = 17
	opMul            = 18
	opRelu           = 19
	opReshape        = 22
	opSoftmax        = 25
	opTanh           = 28
	opLeakyRelu      = 98

	optNone           = 0
	optConv2D         = 1
	optPool2D         = 5
	optFullyConnected = 8
	optSoftmax        = 9
	optAdd            = 11
	optReshape        = 17
	optMul            = 21
	optLeakyRelu      = 75
)

func fbits(v float32) uint32 { return math.Float32bits(v) }

// graph accumulates tensors, buffers, and operators while walking layers.
type graph struct {
	b       *fbBuilder
	tensors []int     // built Tensor tables
	shapes  [][]int32 // shape of each tensor, by index
	buffers []int     // built Buffer tables
	opCodes []int32   // builtin codes in registration order
	ops     []op
}

type op struct {
	code            int32
	inputs, outputs []int32
	optType         byte
	options         int // built options table (0 = none)
}

func newGraph() *graph {
	g := &graph{b: newFbBuilder()}
	// Buffer 0 is the conventional empty buffer for activation tensors.
	g.b.startObject(1)
	g.buffers = append(g.buffers, g.b.endObject())
	return g
}

func (g *graph) opCodeIndex(code int32) int32 {
	for i, c := range g.opCodes {
		if c == code {
			return int32(i)
		}
	}
	g.opCodes = append(g.opCodes, code)
	return int32(len(g.opCodes) - 1)
}

// tensor adds a Tensor table and returns its index. bufferIdx 0 means an
// activation tensor with no constant data.
func (g *graph) tensor(name string, shape []int32, tensorType byte, bufferIdx int) int32 {
	nameOff := g.b.createString(name)
	shapeOff := g.b.int32Vector(shape)
	g.b.startObject(4)
	g.b.uoffsetSlot(0, shapeOff)
	g.b.byteSlot(1, tensorType, tensorFloat32)
	g.b.uint32Slot(2, uint32(bufferIdx), 0)
	g.b.uoffsetSlot(3, nameOff)
	g.tensors = append(g.tensors, g.b.endObject())
	g.shapes = append(g.shapes, shape)
	return int32(len(g.tensors) - 1)
}

// constFloat adds a float32 constant tensor with its data buffer.
func (g *graph) constFloat(name string, shape []int32, data []float32) int32 {
	raw := make([]byte, 4*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[4*i:], fbits(v))
	}
	return g.constRaw(name, shape, tensorFloat32, raw)
}

// constInt32 adds an int32 constant tensor (e.g. reshape targets).
func (g *graph) constInt32(name string, shape []int32, data []int32) int32 {
	raw := make([]byte, 4*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[4*i:], uint32(v))
	}
	return g.constRaw(name, shape, tensorInt32, raw)
}

func (g *graph) constRaw(name string, shape []int32, tensorType byte, raw []byte) int32 {
	dataOff := g.b.byteVector(raw, 16) // schema: force_align 16
	g.b.startObject(1)
	g.b.uoffsetSlot(0, dataOff)
	g.buffers = append(g.buffers, g.b.endObject())
	return g.tensor(name, shape, tensorType, len(g.buffers)-1)
}

func (g *graph) addOp(o op) { g.ops = append(g.ops, o) }

// Marshal encodes a compiled Sequential model as a TFLite flatbuffer.
func Marshal(m *tensai.Sequential) ([]byte, error) {
	layers := m.Layers()
	if len(layers) == 0 {
		return nil, fmt.Errorf("tflite: model has no layers")
	}
	g := newGraph()

	// The input tensor's shape comes from the first parameterized layer.
	var cur int32
	var spatialH, spatialW, spatialC int
	spatial := false
	switch l := layers[0].(type) {
	case *tensai.Conv2D:
		h, w, c, _, _, _, _ := l.Shape()
		spatialH, spatialW, spatialC, spatial = h, w, c, true
		cur = g.tensor("input", []int32{1, int32(h), int32(w), int32(c)}, tensorFloat32, 0)
	case *tensai.Dense:
		weights, _ := l.Params()
		cur = g.tensor("input", []int32{1, int32(weights.Rows)}, tensorFloat32, 0)
	default:
		return nil, fmt.Errorf("tflite: first layer must be Dense or Conv2D, got %T", layers[0])
	}

	features := 0 // valid when !spatial
	if !spatial {
		w, _ := layers[0].(*tensai.Dense).Params()
		features = w.Rows
	}

	for li, layer := range layers {
		name := fmt.Sprintf("l%d", li)
		switch l := layer.(type) {
		case *tensai.Dense:
			weights, bias := l.Params()
			if spatial {
				if spatialH*spatialW*spatialC != weights.Rows {
					return nil, fmt.Errorf("tflite: %s: dense input %d != %dx%dx%d",
						name, weights.Rows, spatialH, spatialW, spatialC)
				}
				cur = g.reshapeTo(name, cur, weights.Rows)
			}
			cur = g.fullyConnected(name, cur, weights, bias, spatial, spatialH, spatialW, spatialC)
			spatial = false
			features = weights.Cols
		case *tensai.Conv2D:
			var err error
			cur, spatialH, spatialW, spatialC, err = g.conv2D(name, cur, l, spatial, spatialH, spatialW, spatialC)
			if err != nil {
				return nil, err
			}
			spatial = true
		case *tensai.MaxPool2D:
			h, w, c, size := l.Shape()
			if !spatial || h != spatialH || w != spatialW || c != spatialC {
				return nil, fmt.Errorf("tflite: %s: maxpool input mismatch", name)
			}
			outH, outW := h/size, w/size
			out := g.tensor(name+"_out", []int32{1, int32(outH), int32(outW), int32(c)}, tensorFloat32, 0)
			g.b.startObject(6)
			g.b.byteSlot(0, paddingValid, 0)
			g.b.int32Slot(1, int32(size), 0)
			g.b.int32Slot(2, int32(size), 0)
			g.b.int32Slot(3, int32(size), 0)
			g.b.int32Slot(4, int32(size), 0)
			opts := g.b.endObject()
			g.addOp(op{code: opMaxPool2D, inputs: []int32{cur}, outputs: []int32{out}, optType: optPool2D, options: opts})
			cur = out
			spatialH, spatialW = outH, outW
		case *tensai.BatchNorm:
			cur = g.batchNorm(name, cur, l)
		case *tensai.Dropout:
			// Identity at inference time.
		case *tensai.ReLU:
			cur = g.activation(name, cur, opRelu, optNone, 0, spatial, spatialH, spatialW, spatialC, features)
		case *tensai.Sigmoid:
			cur = g.activation(name, cur, opLogistic, optNone, 0, spatial, spatialH, spatialW, spatialC, features)
		case *tensai.Tanh:
			cur = g.activation(name, cur, opTanh, optNone, 0, spatial, spatialH, spatialW, spatialC, features)
		case *tensai.LeakyReLU:
			g.b.startObject(1)
			g.b.float32Slot(0, float32(l.Alpha), 0)
			opts := g.b.endObject()
			cur = g.activation(name, cur, opLeakyRelu, optLeakyRelu, opts, spatial, spatialH, spatialW, spatialC, features)
		case *tensai.Softmax:
			g.b.startObject(1)
			g.b.float32Slot(0, 1.0, 0)
			opts := g.b.endObject()
			cur = g.activation(name, cur, opSoftmax, optSoftmax, opts, spatial, spatialH, spatialW, spatialC, features)
		default:
			return nil, fmt.Errorf("tflite: unsupported layer %T", layer)
		}
	}

	return g.finish(cur)
}

// MarshalFile writes the marshaled model to a file.
func MarshalFile(path string, m *tensai.Sequential) error {
	data, err := Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (g *graph) shapeOf(spatial bool, h, w, c, features int) []int32 {
	if spatial {
		return []int32{1, int32(h), int32(w), int32(c)}
	}
	return []int32{1, int32(features)}
}

func (g *graph) activation(name string, in int32, code int32, optType byte, opts int,
	spatial bool, h, w, c, features int) int32 {
	out := g.tensor(name+"_out", g.shapeOf(spatial, h, w, c, features), tensorFloat32, 0)
	g.addOp(op{code: code, inputs: []int32{in}, outputs: []int32{out}, optType: optType, options: opts})
	return out
}

// reshapeTo flattens the current NHWC tensor to [1, n].
func (g *graph) reshapeTo(name string, in int32, n int) int32 {
	shape := g.constInt32(name+"_shape", []int32{2}, []int32{1, int32(n)})
	out := g.tensor(name+"_flat", []int32{1, int32(n)}, tensorFloat32, 0)
	newShape := g.b.int32Vector([]int32{1, int32(n)})
	g.b.startObject(1)
	g.b.uoffsetSlot(0, newShape)
	opts := g.b.endObject()
	g.addOp(op{code: opReshape, inputs: []int32{in, shape}, outputs: []int32{out}, optType: optReshape, options: opts})
	return out
}

// fullyConnected emits FULLY_CONNECTED with weights transposed to TFLite's
// [out, in] layout. When the input just left a spatial domain, the input
// index order changes from tensai's channel-major flatten to the NHWC
// flatten produced by reshapeTo, so weight rows are permuted to match.
func (g *graph) fullyConnected(name string, in int32, weights *tensai.Matrix, bias []float32,
	fromSpatial bool, h, w, c int) int32 {
	inDim, outDim := weights.Rows, weights.Cols
	wData := make([]float32, inDim*outDim)
	for o := 0; o < outDim; o++ {
		for i := 0; i < inDim; i++ {
			src := i
			if fromSpatial {
				// dst position i is the NHWC index; find the tensai row.
				y := i / (w * c)
				x := i % (w * c) / c
				ch := i % c
				src = (ch*h+y)*w + x
			}
			wData[o*inDim+i] = weights.At(src, o)
		}
	}
	wT := g.constFloat(name+"_w", []int32{int32(outDim), int32(inDim)}, wData)
	bT := g.constFloat(name+"_b", []int32{int32(outDim)}, bias)
	out := g.tensor(name+"_out", []int32{1, int32(outDim)}, tensorFloat32, 0)
	g.b.startObject(1)
	opts := g.b.endObject() // all defaults (activation NONE)
	g.addOp(op{code: opFullyConnected, inputs: []int32{in, wT, bT}, outputs: []int32{out}, optType: optFullyConnected, options: opts})
	return out
}

// conv2D emits CONV_2D with the filter reordered to [outC, kh, kw, inC].
func (g *graph) conv2D(name string, in int32, l *tensai.Conv2D,
	spatial bool, curH, curW, curC int) (int32, int, int, int, error) {
	h, w, c, outC, k, stride, pad := l.Shape()
	if !spatial || h != curH || w != curW || c != curC {
		if spatial {
			return 0, 0, 0, 0, fmt.Errorf("tflite: %s: conv input %dx%dx%d != current %dx%dx%d",
				name, h, w, c, curH, curW, curC)
		}
		return 0, 0, 0, 0, fmt.Errorf("tflite: %s: conv after dense is not supported", name)
	}
	outH := (h+2*pad-k)/stride + 1
	outW := (w+2*pad-k)/stride + 1

	var padding byte
	switch {
	case pad == 0:
		padding = paddingValid
	case outH == (h+stride-1)/stride && outW == (w+stride-1)/stride &&
		(outH-1)*stride+k-h == 2*pad && (outW-1)*stride+k-w == 2*pad:
		padding = paddingSame
	default:
		return 0, 0, 0, 0, fmt.Errorf("tflite: %s: padding %d is neither VALID nor SAME", name, pad)
	}

	weights, bias := l.Params() // (c*k*k) x outC, patch index (ch*k+ky)*k+kx
	fData := make([]float32, outC*k*k*c)
	for o := 0; o < outC; o++ {
		for ky := 0; ky < k; ky++ {
			for kx := 0; kx < k; kx++ {
				for ch := 0; ch < c; ch++ {
					src := (ch*k+ky)*k + kx
					dst := ((o*k+ky)*k+kx)*c + ch
					fData[dst] = weights.At(src, o)
				}
			}
		}
	}
	fT := g.constFloat(name+"_w", []int32{int32(outC), int32(k), int32(k), int32(c)}, fData)
	bT := g.constFloat(name+"_b", []int32{int32(outC)}, bias)
	out := g.tensor(name+"_out", []int32{1, int32(outH), int32(outW), int32(outC)}, tensorFloat32, 0)
	g.b.startObject(6)
	g.b.byteSlot(0, padding, 0)
	g.b.int32Slot(1, int32(stride), 0)
	g.b.int32Slot(2, int32(stride), 0)
	opts := g.b.endObject()
	g.addOp(op{code: opConv2D, inputs: []int32{in, fT, bT}, outputs: []int32{out}, optType: optConv2D, options: opts})
	return out, outH, outW, outC, nil
}

// batchNorm folds inference-time BatchNorm into Mul + Add constants.
func (g *graph) batchNorm(name string, in int32, l *tensai.BatchNorm) int32 {
	mean, variance := l.RunningStats()
	gamma, beta := l.Params()
	n := gamma.Cols
	scale := make([]float32, n)
	offset := make([]float32, n)
	for i := 0; i < n; i++ {
		s := gamma.Data[i] / float32(math.Sqrt(float64(variance[i]+l.Eps)))
		scale[i] = s
		offset[i] = beta[i] - mean[i]*s
	}
	// Shapes of the current tensor: reuse via ops' outputs is awkward, so
	// broadcast constants of shape [n] work for both [1,n] and [1,h,w,n].
	sT := g.constFloat(name+"_scale", []int32{int32(n)}, scale)
	oT := g.constFloat(name+"_offset", []int32{int32(n)}, offset)

	mulOut := g.cloneShape(name+"_mul", in)
	g.b.startObject(1)
	mulOpts := g.b.endObject()
	g.addOp(op{code: opMul, inputs: []int32{in, sT}, outputs: []int32{mulOut}, optType: optMul, options: mulOpts})

	addOut := g.cloneShape(name+"_out", mulOut)
	g.b.startObject(1)
	addOpts := g.b.endObject()
	g.addOp(op{code: opAdd, inputs: []int32{mulOut, oT}, outputs: []int32{addOut}, optType: optAdd, options: addOpts})
	return addOut
}

// cloneShape adds an activation tensor with the same shape as an existing
// one (BatchNorm's output shape equals its input shape).
func (g *graph) cloneShape(name string, of int32) int32 {
	return g.tensor(name, g.shapes[of], tensorFloat32, 0)
}

func (g *graph) finish(output int32) ([]byte, error) {
	b := g.b

	// Operators (reverse-friendly: all referenced tables already built).
	opOffs := make([]int, len(g.ops))
	for i, o := range g.ops {
		ins := b.int32Vector(o.inputs)
		outs := b.int32Vector(o.outputs)
		b.startObject(5)
		b.uint32Slot(0, uint32(g.opCodeIndex(o.code)), 0)
		b.uoffsetSlot(1, ins)
		b.uoffsetSlot(2, outs)
		b.byteSlot(3, o.optType, 0)
		b.uoffsetSlot(4, o.options)
		opOffs[i] = b.endObject()
	}
	opsVec := b.offsetVector(opOffs)
	tensorsVec := b.offsetVector(g.tensors)
	inputsVec := b.int32Vector([]int32{0})
	outputsVec := b.int32Vector([]int32{output})
	sgName := b.createString("main")
	b.startObject(5)
	b.uoffsetSlot(0, tensorsVec)
	b.uoffsetSlot(1, inputsVec)
	b.uoffsetSlot(2, outputsVec)
	b.uoffsetSlot(3, opsVec)
	b.uoffsetSlot(4, sgName)
	subgraph := b.endObject()
	subgraphsVec := b.offsetVector([]int{subgraph})

	codeOffs := make([]int, len(g.opCodes))
	for i, code := range g.opCodes {
		b.startObject(4)
		if code <= 127 {
			b.byteSlot(0, byte(code), 0)
		}
		b.int32Slot(3, code, 0)
		codeOffs[i] = b.endObject()
	}
	codesVec := b.offsetVector(codeOffs)
	buffersVec := b.offsetVector(g.buffers)
	desc := b.createString("tensai export")

	b.startObject(5)
	b.uint32Slot(0, schemaVersion, 0)
	b.uoffsetSlot(1, codesVec)
	b.uoffsetSlot(2, subgraphsVec)
	b.uoffsetSlot(3, desc)
	b.uoffsetSlot(4, buffersVec)
	model := b.endObject()

	return b.finish(model, fileIdentifier), nil
}

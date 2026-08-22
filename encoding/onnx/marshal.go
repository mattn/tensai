// Package onnx marshals trained tensai Sequential models into the ONNX
// format (opset 13, FP32, batch size 1), with the protobuf writer
// implemented in-tree — no dependencies. Exported models run on
// onnxruntime and anything else that speaks ONNX.
//
// Layout note: ONNX convolutions are NCHW, which is exactly tensai's own
// row layout (channel-major rows), so unlike the TFLite export no data or
// weight reordering is visible to the caller: feed the exported model the
// same flattened rows tensai's Conv2D consumes, as a [1, C, H, W] tensor.
//
// Supported layers: Dense (Gemm), Conv2D (Conv), MaxPool2D (MaxPool),
// BatchNorm (folded to Mul+Add), Dropout (dropped), Softmax (dense inputs
// only), and the ReLU / LeakyReLU / Sigmoid / Tanh activations.
package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	tensai "github.com/mattn/tensai"
)

const (
	irVersion    = 8
	opsetVersion = 13
	dtFloat      = 1 // TensorProto.DataType FLOAT

	// AttributeProto.AttributeType
	attrFloat = 1
	attrInt   = 2
	attrInts  = 7
)

type node struct {
	op     string
	inputs []string
	output string
	attr   func(*pbuf) // extra AttributeProto fields, or nil
}

type initializer struct {
	name string
	dims []int64
	data []float32
}

type graph struct {
	nodes []node
	inits []initializer
	n     int
}

func (g *graph) add(op string, inputs []string, attr func(*pbuf)) string {
	out := fmt.Sprintf("t%d", g.n)
	g.n++
	g.nodes = append(g.nodes, node{op: op, inputs: inputs, output: out, attr: attr})
	return out
}

func (g *graph) tensor(name string, dims []int64, data []float32) string {
	g.inits = append(g.inits, initializer{name: name, dims: dims, data: data})
	return name
}

func attrIntv(name string, v int64) func(*pbuf) {
	return func(p *pbuf) {
		p.Msg(5, func(a *pbuf) {
			a.Str(1, name)
			a.Int(3, v)
			a.Int(20, attrInt)
		})
	}
}

func attrIntsv(name string, vs []int64) func(*pbuf) {
	return func(p *pbuf) {
		p.Msg(5, func(a *pbuf) {
			a.Str(1, name)
			a.Ints(8, vs)
			a.Int(20, attrInts)
		})
	}
}

func attrFloatv(name string, v float32) func(*pbuf) {
	return func(p *pbuf) {
		p.Msg(5, func(a *pbuf) {
			a.Str(1, name)
			a.Float(2, v)
			a.Int(20, attrFloat)
		})
	}
}

func attrs(fns ...func(*pbuf)) func(*pbuf) {
	return func(p *pbuf) {
		for _, fn := range fns {
			fn(p)
		}
	}
}

// Marshal encodes a trained Sequential model as an ONNX ModelProto.
func Marshal(m *tensai.Sequential) ([]byte, error) {
	layers := m.Layers()
	if len(layers) == 0 {
		return nil, fmt.Errorf("onnx: model has no layers")
	}
	g := &graph{}

	// The graph input's shape comes from the first parameterized layer.
	var inputDims []int64
	var h, w, c, features int
	spatial := false
	switch l := layers[0].(type) {
	case *tensai.Conv2D:
		h, w, c, _, _, _, _ = l.Shape()
		spatial = true
		inputDims = []int64{1, int64(c), int64(h), int64(w)}
	case *tensai.Dense:
		weights, _ := l.Params()
		features = weights.Rows
		inputDims = []int64{1, int64(features)}
	default:
		return nil, fmt.Errorf("onnx: first layer must be Dense or Conv2D, got %T", layers[0])
	}
	cur := "input"

	for li, layer := range layers {
		name := fmt.Sprintf("l%d", li)
		switch l := layer.(type) {
		case *tensai.Dense:
			weights, bias := l.Params()
			if spatial {
				if h*w*c != weights.Rows {
					return nil, fmt.Errorf("onnx: %s: dense input %d != %dx%dx%d", name, weights.Rows, c, h, w)
				}
				cur = g.add("Flatten", []string{cur}, attrIntv("axis", 1))
				spatial = false
			}
			wName := g.tensor(name+"_w", []int64{int64(weights.Rows), int64(weights.Cols)}, weights.Data)
			bName := g.tensor(name+"_b", []int64{int64(len(bias))}, bias)
			cur = g.add("Gemm", []string{cur, wName, bName}, nil)
			features = weights.Cols
		case *tensai.Conv2D:
			inH, inW, inC, outC, k, stride, pad := l.Shape()
			if li != 0 && (!spatial || inH != h || inW != w || inC != c) {
				return nil, fmt.Errorf("onnx: %s: conv input mismatch", name)
			}
			weights, bias := l.Params()
			// tensai stores [(inC*k+ky)*k+kx][outC]; ONNX wants [outC][inC][ky][kx].
			wd := make([]float32, outC*inC*k*k)
			for row := 0; row < inC*k*k; row++ {
				for oc := 0; oc < outC; oc++ {
					wd[oc*inC*k*k+row] = weights.Data[row*outC+oc]
				}
			}
			wName := g.tensor(name+"_w", []int64{int64(outC), int64(inC), int64(k), int64(k)}, wd)
			bName := g.tensor(name+"_b", []int64{int64(outC)}, bias)
			cur = g.add("Conv", []string{cur, wName, bName}, attrs(
				attrIntsv("kernel_shape", []int64{int64(k), int64(k)}),
				attrIntsv("strides", []int64{int64(stride), int64(stride)}),
				attrIntsv("pads", []int64{int64(pad), int64(pad), int64(pad), int64(pad)}),
			))
			h = (inH+2*pad-k)/stride + 1
			w = (inW+2*pad-k)/stride + 1
			c = outC
			spatial = true
		case *tensai.MaxPool2D:
			ph, pw, pc, size := l.Shape()
			if !spatial || ph != h || pw != w || pc != c {
				return nil, fmt.Errorf("onnx: %s: maxpool input mismatch", name)
			}
			cur = g.add("MaxPool", []string{cur}, attrs(
				attrIntsv("kernel_shape", []int64{int64(size), int64(size)}),
				attrIntsv("strides", []int64{int64(size), int64(size)}),
			))
			h, w = h/size, w/size
		case *tensai.BatchNorm:
			// tensai's BatchNorm normalizes every flattened column, so it
			// folds into an element-wise scale and offset; in NCHW those
			// constants carry shape [C, H, W] and broadcast over the batch.
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
			dims := []int64{int64(n)}
			if spatial {
				if n != c*h*w {
					return nil, fmt.Errorf("onnx: %s: batchnorm width %d != %dx%dx%d", name, n, c, h, w)
				}
				dims = []int64{int64(c), int64(h), int64(w)}
			}
			sName := g.tensor(name+"_scale", dims, scale)
			oName := g.tensor(name+"_offset", dims, offset)
			cur = g.add("Mul", []string{cur, sName}, nil)
			cur = g.add("Add", []string{cur, oName}, nil)
		case *tensai.Dropout:
			// Identity at inference time.
		case *tensai.ReLU:
			cur = g.add("Relu", []string{cur}, nil)
		case *tensai.LeakyReLU:
			cur = g.add("LeakyRelu", []string{cur}, attrFloatv("alpha", float32(l.Alpha)))
		case *tensai.Sigmoid:
			cur = g.add("Sigmoid", []string{cur}, nil)
		case *tensai.Tanh:
			cur = g.add("Tanh", []string{cur}, nil)
		case *tensai.Softmax:
			if spatial {
				return nil, fmt.Errorf("onnx: %s: softmax needs flattened features", name)
			}
			cur = g.add("Softmax", []string{cur}, nil) // axis defaults to -1
		default:
			return nil, fmt.Errorf("onnx: unsupported layer %T", layer)
		}
	}

	outDims := []int64{1, int64(features)}
	if spatial {
		outDims = []int64{1, int64(c), int64(h), int64(w)}
	}
	return encode(g, inputDims, cur, outDims), nil
}

func valueInfo(p *pbuf, field int, name string, dims []int64) {
	p.Msg(field, func(v *pbuf) {
		v.Str(1, name)
		v.Msg(2, func(t *pbuf) {
			t.Msg(1, func(tt *pbuf) {
				tt.Int(1, dtFloat)
				tt.Msg(2, func(sh *pbuf) {
					for _, d := range dims {
						sh.Msg(1, func(dim *pbuf) { dim.Int(1, d) })
					}
				})
			})
		})
	})
}

func encode(g *graph, inputDims []int64, output string, outputDims []int64) []byte {
	var p pbuf
	p.Int(1, irVersion)
	p.Str(2, "tensai")
	p.Msg(8, func(op *pbuf) {
		op.Str(1, "")
		op.Int(2, opsetVersion)
	})
	p.Msg(7, func(gr *pbuf) {
		gr.Str(2, "tensai")
		for _, n := range g.nodes {
			gr.Msg(1, func(np *pbuf) {
				for _, in := range n.inputs {
					np.Str(1, in)
				}
				np.Str(2, n.output)
				np.Str(4, n.op)
				if n.attr != nil {
					n.attr(np)
				}
			})
		}
		for _, ini := range g.inits {
			gr.Msg(5, func(tp *pbuf) {
				for _, d := range ini.dims {
					tp.Int(1, d)
				}
				tp.Int(2, dtFloat)
				tp.Str(8, ini.name)
				raw := make([]byte, 0, len(ini.data)*4)
				for _, f := range ini.data {
					raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(f))
				}
				tp.Bytes(9, raw)
			})
		}
		valueInfo(gr, 11, "input", inputDims)
		valueInfo(gr, 12, output, outputDims)
	})
	return p.b
}

// MarshalFile writes the marshaled model to a file.
func MarshalFile(path string, m *tensai.Sequential) error {
	data, err := Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

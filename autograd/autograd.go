package autograd

import (
	"fmt"
	"strings"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/dims"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/optim"
)

// Node is a tensor-valued node in a dynamically built computation graph for
// reverse-mode automatic differentiation. Build the forward computation by
// chaining operations, then call Backward on the (scalar) result to fill
// Grad on every Param node that contributed to it.
//
// Values are n-dimensional: a Matrix is accepted anywhere a leaf is built
// and becomes a zero-copy 2-D tensor view, so the two-dimensional API from
// before keeps working, and the same ops run on (batch, seq, model) shaped
// activations. Element-wise ops broadcast NumPy-style and MatMul multiplies
// stacks of matrices, exactly as tensai.Tensor does.
//
// Unlike the Layer API, shape mismatches panic: graph construction errors
// are programming errors, and error returns would make chaining unusable.
//
//	w := autograd.Param(tensai.RandomMatrix(2, 8, rng))
//	b := autograd.Param(tensai.NewMatrix(1, 8))
//	loss := autograd.Input(x).MatMul(w).Add(b).ReLU().MSELoss(y.Tensor())
//	loss.Backward()
//	// w.grad and b.grad now hold dLoss/dw and dLoss/db.
type Node struct {
	value *tensai.Tensor
	grad  *tensai.Tensor

	// A node whose graph runs on a device holds its value and gradient
	// there instead, and the host copies above are filled in only when
	// something reads them. shapeOf is the shape either way.
	dev     *gpu.Tensor
	devGrad *gpu.Tensor
	shapeOf []int

	parents      []*Node
	backFn       func()
	tape         *Tape // buffer pool this graph draws from, if any
	requiresGrad bool
	op           string // operation that produced this node ("" for leaves)
	name         string // optional display name for ToDot
}

// Array is the constraint for the array types a graph leaf can wrap. A
// Matrix is taken as a zero-copy 2-D tensor view, so a parameter built from
// a Matrix keeps sharing that matrix's backing data.
type Array interface {
	*tensai.Matrix | *tensai.Tensor
}

// tensorOf views v as a tensor without copying.
func tensorOf[T Array](v T) *tensai.Tensor {
	switch x := any(v).(type) {
	case *tensai.Matrix:
		if x == nil {
			panic("tensai: nil matrix")
		}
		return x.Tensor()
	case *tensai.Tensor:
		if x == nil {
			panic("tensai: nil tensor")
		}
		return x
	}
	panic("tensai: unreachable")
}

// Value returns the node's value. Reading it through a method rather than
// a field is what lets a value live somewhere other than host memory: when
// the graph runs on a device, this is where it comes back.
func (n *Node) Value() *tensai.Tensor {
	n.syncValue()
	return n.value
}

// Grad returns the gradient accumulated into the node by Backward, or nil
// if none has been.
func (n *Node) Grad() *tensai.Tensor {
	n.syncGrad()
	return n.grad
}

// Named sets a display name shown by ToDot and returns the node.
func (n *Node) Named(name string) *Node {
	n.name = name
	return n
}

// Input wraps a matrix or tensor as a constant graph leaf. No gradient is
// computed for it.
func Input[T Array](v T) *Node {
	t := tensorOf(v)
	return &Node{value: t, shapeOf: t.Shape}
}

// Param wraps a matrix or tensor as a trainable graph leaf. Backward
// accumulates its gradient into Grad.
func Param[T Array](v T) *Node {
	t := tensorOf(v)
	return &Node{value: t, shapeOf: t.Shape, requiresGrad: true}
}

// ZeroGrads clears the gradients of the given nodes. Call it between
// training steps: Backward accumulates.
func ZeroGrads(nodes ...*Node) {
	for _, n := range nodes {
		n.grad = nil
		n.devGrad = nil
	}
}

// hasGrad reports whether a gradient has reached the node, on either side.
func (n *Node) hasGrad() bool { return n.grad != nil || n.devGrad != nil }

// Shape returns the shape of the node's value, without bringing a
// device-resident value home to ask: every node records it when it is
// built, whichever side holds the data.
func (n *Node) Shape() []int { return n.shapeOf }

// Matrix returns a matrix view of a 2-D node value, sharing its backing
// data. It panics for any other rank.
func (n *Node) Matrix() *tensai.Matrix {
	m, err := n.value.Matrix()
	if err != nil {
		panic(err.Error())
	}
	return m
}

// ensureGrad lazily allocates a zero gradient buffer, from the tape when
// the graph has one.
func (n *Node) ensureGrad() *tensai.Tensor {
	n.syncGrad()
	n.devGrad = nil
	if n.grad == nil {
		if n.tape != nil {
			n.grad = n.tape.zeros(n.Shape())
		} else {
			n.grad = tensai.NewTensor(n.Shape()...)
		}
	}
	return n.grad
}

// accum adds g into n's gradient, summing over every axis along which n was
// broadcast to produce g.
func (n *Node) accum(g *tensai.Tensor) {
	addScaled(n.ensureGrad(), g, 1)
}

// accumScaled adds s*g into n's gradient, with the same reduction as accum.
func (n *Node) accumScaled(g *tensai.Tensor, s tensai.Float) {
	addScaled(n.ensureGrad(), g, s)
}

// addScaled adds s*src into dst, summing src over the axes along which dst
// was broadcast to reach src's shape. dst's shape must broadcast to src's.
func addScaled(dst, src *tensai.Tensor, s tensai.Float) {
	if dims.Same(dst.Shape, src.Shape) {
		if s == 1 {
			kernels.AddSlice(dst.Data, src.Data)
			return
		}
		for i, v := range src.Data {
			dst.Data[i] += s * v
		}
		return
	}
	if len(src.Shape) == 0 {
		dst.Data[0] += s * src.Data[0]
		return
	}
	if len(dst.Shape) > len(src.Shape) {
		panic(fmt.Sprintf("tensai: cannot reduce gradient %v into %v", src.Shape, dst.Shape))
	}
	// Walk src row-major; the stride table carries 0 on every axis dst is
	// broadcast along, so those elements all land on the same dst cell.
	strides := dims.BroadcastStrides(dst.Shape, src.Shape)
	last := len(src.Shape) - 1
	n := src.Shape[last]
	idx := make([]int, len(src.Shape))
	off := 0
	for pos := 0; pos < len(src.Data); pos += n {
		row := src.Data[pos : pos+n]
		if strides[last] == 1 {
			d := dst.Data[off : off+n]
			for i, v := range row {
				d[i] += s * v
			}
		} else { // broadcast along the last axis: the row collapses to a cell
			var sum tensai.Float
			for _, v := range row {
				sum += v
			}
			dst.Data[off] += s * sum
		}
		for d := last - 1; d >= 0; d-- {
			idx[d]++
			off += strides[d]
			if idx[d] < src.Shape[d] {
				break
			}
			idx[d] = 0
			off -= strides[d] * src.Shape[d]
		}
	}
}

// newNode wires up a result node, recording the operation that produced it
// (used by ToDot) and inheriting the parents' tape.
func newNode(op string, value *tensai.Tensor, parents ...*Node) *Node {
	out := &Node{value: value, shapeOf: value.Shape, op: op, parents: parents, tape: tapeOf(parents...)}
	for _, p := range parents {
		if p.requiresGrad {
			out.requiresGrad = true
		}
	}
	return out
}

// withBack replaces the node's backward closure with one that receives the
// node itself, keeping op definitions compact.
func (n *Node) withBack(fn func(out *Node)) *Node {
	if n.requiresGrad {
		n.backFn = func() { fn(n) }
	}
	return n
}

// Backward runs reverse-mode differentiation from n, which should be a
// scalar (single-element) loss. Gradients accumulate into the Grad field of
// every contributing Param node.
func (n *Node) Backward() {
	if dims.Prod(n.Shape()) != 1 {
		panic(fmt.Sprintf("tensai: backward root must be scalar, got %s", shapeString(n.Shape())))
	}
	var topo []*Node
	visited := map[*Node]bool{}
	var visit func(*Node)
	visit = func(x *Node) {
		if visited[x] {
			return
		}
		visited[x] = true
		for _, p := range x.parents {
			visit(p)
		}
		topo = append(topo, x)
	}
	visit(n)

	g := n.ensureGrad()
	g.Data[0] = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backFn != nil && topo[i].hasGrad() {
			topo[i].backFn()
		}
	}
}

// Scalar returns the value of a single-element node.
func (n *Node) Scalar() tensai.Float {
	if dims.Prod(n.Shape()) != 1 {
		panic(fmt.Sprintf("tensai: scalar called on %s node", shapeString(n.Shape())))
	}
	return n.Value().Data[0]
}

// shapeString renders a shape the way the docs and graphs do: 5x3, 2x4x8.
func shapeString(shape []int) string {
	if len(shape) == 0 {
		return "scalar"
	}
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprint(d)
	}
	return strings.Join(parts, "x")
}

// Trainer owns the optimizer bookkeeping for a set of autograd parameters,
// so a training step is just building the loss graph and calling Step.
//
//	trainer := autograd.NewTrainer(optim.NewAdam(0.05), w1, b1, w2, b2)
//	for step := 0; step < 2000; step++ {
//		loss := forward(x).MSELoss(y)
//		trainer.Step(loss)
//	}
type Trainer struct {
	ups    []optim.Updater
	params []*Node

	// The device path, used for a parameter whose gradient stayed on the
	// GPU: the update runs there too, so the weights never come home.
	adam     *optim.Adam
	devState []adamState
	steps    int
}

// adamState is one parameter's Adam moments, resident on the device.
type adamState struct {
	m, v *gpu.Tensor
}

// NewTrainer gives each parameter its own optimizer state and returns a
// Trainer that updates them.
func NewTrainer(opt optim.Optimizer, params ...*Node) *Trainer {
	ups := make([]optim.Updater, len(params))
	for i := range ups {
		ups[i] = opt.New()
	}
	t := &Trainer{ups: ups, params: params}
	if adam, ok := opt.(*optim.Adam); ok {
		// Only Adam has a device kernel; with any other optimizer a
		// device-resident gradient comes home for the update.
		t.adam = adam
		t.devState = make([]adamState, len(params))
	}
	return t
}

// Step runs backward from the scalar loss, applies one optimizer update to
// every parameter, clears the gradients, and returns the loss value.
func (t *Trainer) Step(loss *Node) tensai.Float {
	loss.Backward()
	t.steps++
	for i, p := range t.params {
		if !p.hasGrad() {
			// Parameter unused in this graph; its state simply sits out.
			continue
		}
		if t.deviceStep(i, p) {
			continue
		}
		// Optimizers update flat parameter buffers, so any rank works as
		// long as value and gradient agree.
		t.ups[i].Step(flatMatrix(p.Value()), flatMatrix(p.Grad()), nil, nil)
	}
	// The loss is read before the gradients are dropped, since reading it
	// may be what brings the last device value home.
	v := loss.Scalar()
	ZeroGrads(t.params...)
	return v
}

// deviceStep applies Adam to a parameter whose value and gradient are both
// on the device, and reports whether it did. The moments live beside the
// weights and outlive the tape's Reset, as the weights do.
func (t *Trainer) deviceStep(i int, p *Node) bool {
	if t.adam == nil || p.dev == nil || p.devGrad == nil {
		return false
	}
	d := p.device()
	if d == nil {
		return false
	}
	st := &t.devState[i]
	if st.m == nil {
		zero := tensai.NewTensor(p.Shape()...)
		m, err := d.Upload(zero)
		if err != nil {
			return false
		}
		v, err := d.Upload(zero)
		if err != nil {
			m.Free()
			return false
		}
		st.m, st.v = m, v
	}
	a := t.adam
	rc1 := 1 / (1 - kernels.PowF(a.Beta1, tensai.Float(t.steps)))
	rc2 := 1 / (1 - kernels.PowF(a.Beta2, tensai.Float(t.steps)))
	if err := p.dev.AdamStep(p.devGrad, st.m, st.v, a.LR, a.Beta1, a.Beta2,
		rc1, rc2, a.Eps, a.WeightDecay); err != nil {
		return false
	}
	// The weights just changed on the device; any host copy is stale.
	p.value = nil
	return true
}

// flatMatrix views a tensor's data as a 1 x N matrix for the optimizers,
// which work element-wise and do not care about the shape.
func flatMatrix(t *tensai.Tensor) *tensai.Matrix {
	m, err := t.Reshape(1, len(t.Data))
	if err != nil {
		panic(err.Error())
	}
	mm, err := m.Matrix()
	if err != nil {
		panic(err.Error())
	}
	return mm
}

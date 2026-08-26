package tensai

import (
	"fmt"
)

// Node is a matrix-valued node in a dynamically built computation graph for
// reverse-mode automatic differentiation. Build the forward computation by
// chaining operations, then call Backward on the (scalar) result to fill
// Grad on every Param node that contributed to it.
//
// Unlike the Layer API, shape mismatches panic: graph construction errors
// are programming errors, and error returns would make chaining unusable.
//
//	w := tensai.Param(tensai.RandomMatrix(2, 8, rng))
//	b := tensai.Param(tensai.NewMatrix(1, 8))
//	loss := tensai.Input(x).MatMul(w).AddRow(b).ReLU().MSELoss(y)
//	loss.Backward()
//	// w.Grad and b.Grad now hold dLoss/dw and dLoss/db.
type Node struct {
	Value *Matrix
	Grad  *Matrix

	parents      []*Node
	backFn       func()
	requiresGrad bool
	op           string // operation that produced this node ("" for leaves)
	name         string // optional display name for ToDot
}

// Named sets a display name shown by ToDot and returns the node.
func (n *Node) Named(name string) *Node {
	n.name = name
	return n
}

// Input wraps a matrix as a constant graph leaf. No gradient is computed
// for it.
func Input(m *Matrix) *Node {
	return &Node{Value: m}
}

// Param wraps a matrix as a trainable graph leaf. Backward accumulates its
// gradient into Grad.
func Param(m *Matrix) *Node {
	return &Node{Value: m, requiresGrad: true}
}

// ZeroGrads clears the gradients of the given nodes. Call it between
// training steps: Backward accumulates.
func ZeroGrads(nodes ...*Node) {
	for _, n := range nodes {
		n.Grad = nil
	}
}

// ensureGrad lazily allocates a zero gradient buffer.
func (n *Node) ensureGrad() *Matrix {
	if n.Grad == nil {
		n.Grad = NewMatrix(n.Value.Rows, n.Value.Cols)
	}
	return n.Grad
}

// newNode wires up a result node, recording the operation that produced it
// (used by ToDot).
func newNode(op string, value *Matrix, parents ...*Node) *Node {
	out := &Node{Value: value, op: op, parents: parents}
	for _, p := range parents {
		if p.requiresGrad {
			out.requiresGrad = true
		}
	}
	return out
}

// Backward runs reverse-mode differentiation from n, which should be a
// scalar (1x1) loss. Gradients accumulate into the Grad field of every
// contributing Param node.
func (n *Node) Backward() {
	if len(n.Value.Data) != 1 {
		panic(fmt.Sprintf("tensai: backward root must be scalar, got %dx%d", n.Value.Rows, n.Value.Cols))
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
		if topo[i].backFn != nil && topo[i].Grad != nil {
			topo[i].backFn()
		}
	}
}

// MatMul returns the matrix product n * o.
func (n *Node) MatMul(o *Node) *Node {
	v, err := Dot(n.Value, o.Value)
	if err != nil {
		panic(err.Error())
	}
	out := newNode("matmul", v, n, o)
	out.backFn = func() {
		if n.requiresGrad {
			d, _ := Dot(out.Grad, o.Value.T())
			addInto(n.ensureGrad(), d)
		}
		if o.requiresGrad {
			d, _ := Dot(n.Value.T(), out.Grad)
			addInto(o.ensureGrad(), d)
		}
	}
	return out
}

// Add returns the element-wise sum n + o. Shapes must match.
func (n *Node) Add(o *Node) *Node {
	v, err := Add(n.Value, o.Value)
	if err != nil {
		panic(err.Error())
	}
	return newNode("add", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			addInto(n.ensureGrad(), out.Grad)
		}
		if o.requiresGrad {
			addInto(o.ensureGrad(), out.Grad)
		}
	})
}

// Sub returns the element-wise difference n - o. Shapes must match.
func (n *Node) Sub(o *Node) *Node {
	if n.Value.Rows != o.Value.Rows || n.Value.Cols != o.Value.Cols {
		panic(fmt.Sprintf("tensai: sub shape mismatch: %dx%d vs %dx%d",
			n.Value.Rows, n.Value.Cols, o.Value.Rows, o.Value.Cols))
	}
	v := NewMatrix(n.Value.Rows, n.Value.Cols)
	for i := range v.Data {
		v.Data[i] = n.Value.Data[i] - o.Value.Data[i]
	}
	return newNode("sub", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			addInto(n.ensureGrad(), out.Grad)
		}
		if o.requiresGrad {
			g := o.ensureGrad()
			for i, gv := range out.Grad.Data {
				g.Data[i] -= gv
			}
		}
	})
}

// MulElem returns the element-wise (Hadamard) product. Shapes must match.
func (n *Node) MulElem(o *Node) *Node {
	if n.Value.Rows != o.Value.Rows || n.Value.Cols != o.Value.Cols {
		panic(fmt.Sprintf("tensai: mulelem shape mismatch: %dx%d vs %dx%d",
			n.Value.Rows, n.Value.Cols, o.Value.Rows, o.Value.Cols))
	}
	v := NewMatrix(n.Value.Rows, n.Value.Cols)
	for i := range v.Data {
		v.Data[i] = n.Value.Data[i] * o.Value.Data[i]
	}
	return newNode("mulelem", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			g := n.ensureGrad()
			for i, gv := range out.Grad.Data {
				g.Data[i] += gv * o.Value.Data[i]
			}
		}
		if o.requiresGrad {
			g := o.ensureGrad()
			for i, gv := range out.Grad.Data {
				g.Data[i] += gv * n.Value.Data[i]
			}
		}
	})
}

// Scale returns n multiplied by the scalar s.
func (n *Node) Scale(s Float) *Node {
	v := NewMatrix(n.Value.Rows, n.Value.Cols)
	for i, x := range n.Value.Data {
		v.Data[i] = s * x
	}
	return newNode("scale", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i, gv := range out.Grad.Data {
			g.Data[i] += s * gv
		}
	})
}

// AddRow adds a 1xN row node to every row of n (bias broadcast).
func (n *Node) AddRow(row *Node) *Node {
	v, err := AddBias(n.Value, row.Value.Data)
	if err != nil || row.Value.Rows != 1 {
		panic(fmt.Sprintf("tensai: addrow expects 1x%d row, got %dx%d",
			n.Value.Cols, row.Value.Rows, row.Value.Cols))
	}
	return newNode("addrow", v, n, row).withBack(func(out *Node) {
		if n.requiresGrad {
			addInto(n.ensureGrad(), out.Grad)
		}
		if row.requiresGrad {
			g := row.ensureGrad()
			for r := 0; r < out.Grad.Rows; r++ {
				for c := 0; c < out.Grad.Cols; c++ {
					g.Data[c] += out.Grad.Data[r*out.Grad.Cols+c]
				}
			}
		}
	})
}

// ReLU applies max(0, x) element-wise.
func (n *Node) ReLU() *Node {
	v := NewMatrix(n.Value.Rows, n.Value.Cols)
	reluFwd(v.Data, n.Value.Data)
	return newNode("relu", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i, gv := range out.Grad.Data {
			if n.Value.Data[i] > 0 {
				g.Data[i] += gv
			}
		}
	})
}

// Sigmoid applies 1/(1+e^-x) element-wise.
func (n *Node) Sigmoid() *Node {
	v := NewMatrix(n.Value.Rows, n.Value.Cols)
	sigmoidFwd(v.Data, n.Value.Data)
	return newNode("sigmoid", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i, gv := range out.Grad.Data {
			y := v.Data[i]
			g.Data[i] += gv * y * (1 - y)
		}
	})
}

// Tanh applies tanh(x) element-wise.
func (n *Node) Tanh() *Node {
	v := NewMatrix(n.Value.Rows, n.Value.Cols)
	tanhFwd(v.Data, n.Value.Data)
	return newNode("tanh", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i, gv := range out.Grad.Data {
			y := v.Data[i]
			g.Data[i] += gv * (1 - y*y)
		}
	})
}

// T returns the transpose of n.
func (n *Node) T() *Node {
	return newNode("transpose", n.Value.T(), n).withBack(func(out *Node) {
		addInto(n.ensureGrad(), out.Grad.T())
	})
}

// Softmax normalizes each row of n into a probability distribution.
func (n *Node) Softmax() *Node {
	cols := n.Value.Cols
	v := NewMatrix(n.Value.Rows, cols)
	for r := 0; r < n.Value.Rows; r++ {
		row := n.Value.Data[r*cols : (r+1)*cols]
		outRow := v.Data[r*cols : (r+1)*cols]
		maxVal := row[0]
		for _, x := range row {
			if x > maxVal {
				maxVal = x
			}
		}
		expShift(outRow, row, maxVal)
		var denom Float
		for _, e := range outRow {
			denom += e
		}
		scaleSlice(outRow, 1/denom)
	}
	return newNode("softmax", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for r := 0; r < v.Rows; r++ {
			y := v.Data[r*cols : (r+1)*cols]
			gv := out.Grad.Data[r*cols : (r+1)*cols]
			softmaxBwdAdd(g.Data[r*cols:(r+1)*cols], gv, y)
		}
	})
}

// Sum reduces n to a 1x1 scalar by summing all elements.
func (n *Node) Sum() *Node {
	v := NewMatrix(1, 1)
	for _, x := range n.Value.Data {
		v.Data[0] += x
	}
	return newNode("sum", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i := range g.Data {
			g.Data[i] += out.Grad.Data[0]
		}
	})
}

// Mean reduces n to a 1x1 scalar averaging all elements.
func (n *Node) Mean() *Node {
	inv := 1 / Float(len(n.Value.Data))
	return n.Sum().Scale(inv)
}

// MSELoss returns the scalar mean squared error against a constant target.
func (n *Node) MSELoss(target *Matrix) *Node {
	diff := n.Sub(Input(target))
	return diff.MulElem(diff).Mean()
}

// SoftmaxCELoss returns the scalar softmax cross-entropy against integer
// class labels (an Mx1 matrix of class indices), matching the
// SoftmaxCrossEntropy loss used by Sequential models.
func (n *Node) SoftmaxCELoss(target *Matrix) *Node {
	lossVal, grad, err := SoftmaxCrossEntropy{}.Loss(n.Value, target)
	if err != nil {
		panic(err.Error())
	}
	v := NewMatrix(1, 1)
	v.Data[0] = lossVal
	return newNode("softmax_ce", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i, gv := range grad.Data {
			g.Data[i] += out.Grad.Data[0] * gv
		}
	})
}

// Scalar returns the value of a 1x1 node.
func (n *Node) Scalar() Float {
	if len(n.Value.Data) != 1 {
		panic(fmt.Sprintf("tensai: scalar called on %dx%d node", n.Value.Rows, n.Value.Cols))
	}
	return n.Value.Data[0]
}

// Trainer owns the optimizer bookkeeping for a set of autograd parameters,
// so a training step is just building the loss graph and calling Step.
//
//	trainer := tensai.NewTrainer(tensai.NewAdam(0.05), w1, b1, w2, b2)
//	for step := 0; step < 2000; step++ {
//		loss := forward(x).MSELoss(y)
//		trainer.Step(loss)
//	}
type Trainer struct {
	opt    Optimizer
	params []*Node
}

// NewTrainer registers the parameters with the optimizer and returns a
// Trainer that updates them.
func NewTrainer(opt Optimizer, params ...*Node) *Trainer {
	for range params {
		opt.NewLayer()
	}
	return &Trainer{opt: opt, params: params}
}

// Step runs backward from the scalar loss, applies one optimizer update to
// every parameter, clears the gradients, and returns the loss value.
func (t *Trainer) Step(loss *Node) Float {
	loss.Backward()
	for i, p := range t.params {
		if p.Grad == nil {
			// Parameter unused in this graph; keep its optimizer slot aligned.
			continue
		}
		t.opt.Step(i, p.Value, p.Grad, nil, nil)
	}
	ZeroGrads(t.params...)
	return loss.Scalar()
}

// withBack replaces the node's backward closure with one that receives the
// node itself, keeping op definitions compact.
func (n *Node) withBack(fn func(out *Node)) *Node {
	if n.requiresGrad {
		n.backFn = func() { fn(n) }
	}
	return n
}

// addInto adds src into dst element-wise. Shapes must already match.
func addInto(dst, src *Matrix) {
	for i, v := range src.Data {
		dst.Data[i] += v
	}
}

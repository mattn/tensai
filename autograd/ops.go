package autograd

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/dims"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/loss"
)

// The operations below all follow the same shape: run the forward pass on
// the tensor kernels, then register a closure that turns the output
// gradient into the operands'. Broadcasting is handled once, in accum,
// which sums a gradient back over whatever axes an operand was stretched
// along -- so every element-wise op below works for any pair of
// broadcast-compatible shapes without knowing which one was broadcast.

// MatMul multiplies two stacks of matrices: the last two axes are the
// matrix dimensions and the leading axes broadcast, so (batch, m, k) times
// (k, n) yields (batch, m, n).
func (n *Node) MatMul(o *Node) *Node {
	if out, ok := devMatMul(n, o, gemmNN); ok {
		return out
	}
	tp := tapeOf(n, o)
	v := tp.tensor(matmulShape(n.Shape(), o.Shape(), gemmNN))
	if err := tensai.MatMulInto(v, n.Value(), o.Value()); err != nil {
		panic(err.Error())
	}
	return newNode("matmul", v, n, o).withBack(func(out *Node) {
		// The two gradients are grad * b^T and a^T * grad. Both run on the
		// transposed GEMM modes, so neither operand is copied into a
		// transposed buffer first.
		if n.requiresGrad {
			d := tp.tensor(matmulShape(out.Grad().Shape, o.Shape(), gemmNT))
			if err := tensai.MatMulNTInto(d, out.Grad(), o.Value()); err != nil {
				panic(err.Error())
			}
			n.accum(d)
		}
		if o.requiresGrad {
			d := tp.tensor(matmulShape(n.Shape(), out.Grad().Shape, gemmTN))
			if err := tensai.MatMulTNInto(d, n.Value(), out.Grad()); err != nil {
				panic(err.Error())
			}
			o.accum(d)
		}
	})
}

// Add returns the element-wise sum n + o, broadcasting the operands.
func (n *Node) Add(o *Node) *Node {
	if out, ok := devBinary(gpu.OpAdd, n, o); ok {
		return out
	}
	tp := tapeOf(n, o)
	v := tp.tensor(binShape(n.Shape(), o.Shape()))
	if err := tensai.AddInto(v, n.Value(), o.Value()); err != nil {
		panic(err.Error())
	}
	return newNode("add", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			n.accum(out.Grad())
		}
		if o.requiresGrad {
			o.accum(out.Grad())
		}
	})
}

// Sub returns the element-wise difference n - o, broadcasting the operands.
func (n *Node) Sub(o *Node) *Node {
	if out, ok := devBinary(gpu.OpSub, n, o); ok {
		return out
	}
	tp := tapeOf(n, o)
	v := tp.tensor(binShape(n.Shape(), o.Shape()))
	if err := tensai.SubInto(v, n.Value(), o.Value()); err != nil {
		panic(err.Error())
	}
	return newNode("sub", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			n.accum(out.Grad())
		}
		if o.requiresGrad {
			o.accumScaled(out.Grad(), -1)
		}
	})
}

// Mul returns the element-wise (Hadamard) product, broadcasting the
// operands.
func (n *Node) Mul(o *Node) *Node {
	if out, ok := devBinary(gpu.OpMul, n, o); ok {
		return out
	}
	tp := tapeOf(n, o)
	v := tp.tensor(binShape(n.Shape(), o.Shape()))
	if err := tensai.MulInto(v, n.Value(), o.Value()); err != nil {
		panic(err.Error())
	}
	return newNode("mul", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			d := tp.tensor(binShape(out.Grad().Shape, o.Shape()))
			if err := tensai.MulInto(d, out.Grad(), o.Value()); err != nil {
				panic(err.Error())
			}
			n.accum(d)
		}
		if o.requiresGrad {
			d := tp.tensor(binShape(out.Grad().Shape, n.Shape()))
			if err := tensai.MulInto(d, out.Grad(), n.Value()); err != nil {
				panic(err.Error())
			}
			o.accum(d)
		}
	})
}

// MulElem is the former name of Mul, kept for the matrix-shaped API.
func (n *Node) MulElem(o *Node) *Node { return n.Mul(o) }

// Div returns the element-wise quotient n / o, broadcasting the operands.
func (n *Node) Div(o *Node) *Node {
	tp := tapeOf(n, o)
	v := tp.tensor(binShape(n.Shape(), o.Shape()))
	if err := tensai.DivInto(v, n.Value(), o.Value()); err != nil {
		panic(err.Error())
	}
	return newNode("div", v, n, o).withBack(func(out *Node) {
		if n.requiresGrad {
			d := tp.tensor(binShape(out.Grad().Shape, o.Shape()))
			if err := tensai.DivInto(d, out.Grad(), o.Value()); err != nil {
				panic(err.Error())
			}
			n.accum(d)
		}
		if o.requiresGrad {
			// d/db (a/b) = -a/b^2, and a/b^2 is the output over b again.
			q := tp.tensor(binShape(out.Shape(), o.Shape()))
			if err := tensai.DivInto(q, out.Value(), o.Value()); err != nil {
				panic(err.Error())
			}
			d := tp.tensor(binShape(out.Grad().Shape, q.Shape))
			if err := tensai.MulInto(d, out.Grad(), q); err != nil {
				panic(err.Error())
			}
			o.accumScaled(d, -1)
		}
	})
}

// Scale returns n multiplied by the scalar s.
func (n *Node) Scale(s tensai.Float) *Node {
	if out, ok := devScale(n, s); ok {
		return out
	}
	v := n.tape.tensor(n.Shape()) // overwritten by the copy below
	copy(v.Data, n.Value().Data)
	v.Scale(s)
	return newNode("scale", v, n).withBack(func(out *Node) {
		n.accumScaled(out.Grad(), s)
	})
}

// Neg returns -n.
func (n *Node) Neg() *Node { return n.Scale(-1) }

// AddRow adds a 1xN row node to every row of n. Add broadcasts on its own,
// so this only pins down the intent (and the shape check) for the
// two-dimensional API.
func (n *Node) AddRow(row *Node) *Node {
	if len(row.Shape()) != 2 || row.Shape()[0] != 1 {
		panic(fmt.Sprintf("tensai: addrow expects a 1xN row, got %s", shapeString(row.Shape())))
	}
	return n.Add(row)
}

// unary builds an element-wise op: fwd fills the output from the input, and
// bwd accumulates the input gradient given (grad, input, output).
func (n *Node) unary(op string, fwd func(dst, src []tensai.Float), bwd func(dst, grad, src, out []tensai.Float)) *Node {
	// fwd writes every element, so the buffer needs no clearing.
	v := n.tape.tensor(n.Shape())
	fwd(v.Data, n.Value().Data)
	return newNode(op, v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		bwd(g.Data, out.Grad().Data, n.Value().Data, v.Data)
	})
}

// ReLU applies max(0, x) element-wise.
func (n *Node) ReLU() *Node {
	if out, ok := devActivate(gpu.ActReLU, "relu", n); ok {
		return out
	}
	return n.unary("relu", kernels.ReluFwd, func(dst, grad, src, _ []tensai.Float) {
		for i, gv := range grad {
			if src[i] > 0 {
				dst[i] += gv
			}
		}
	})
}

// LeakyReLU applies max(alpha*x, x) element-wise.
func (n *Node) LeakyReLU(alpha tensai.Float) *Node {
	return n.unary("leakyrelu", func(dst, src []tensai.Float) {
		kernels.LeakyFwd(dst, src, alpha)
	}, func(dst, grad, src, _ []tensai.Float) {
		for i, gv := range grad {
			if src[i] > 0 {
				dst[i] += gv
			} else {
				dst[i] += alpha * gv
			}
		}
	})
}

// Sigmoid applies 1/(1+e^-x) element-wise.
func (n *Node) Sigmoid() *Node {
	if out, ok := devActivate(gpu.ActSigmoid, "sigmoid", n); ok {
		return out
	}
	return n.unary("sigmoid", kernels.SigmoidFwd, func(dst, grad, _, out []tensai.Float) {
		for i, gv := range grad {
			y := out[i]
			dst[i] += gv * y * (1 - y)
		}
	})
}

// Tanh applies tanh(x) element-wise.
func (n *Node) Tanh() *Node {
	if out, ok := devActivate(gpu.ActTanh, "tanh", n); ok {
		return out
	}
	return n.unary("tanh", kernels.TanhFwd, func(dst, grad, _, out []tensai.Float) {
		for i, gv := range grad {
			y := out[i]
			dst[i] += gv * (1 - y*y)
		}
	})
}

// GELU applies the Gaussian error linear unit element-wise.
func (n *Node) GELU() *Node {
	if out, ok := devActivate(gpu.ActGELU, "gelu", n); ok {
		return out
	}
	return n.unary("gelu", kernels.GeluFwd, func(dst, grad, src, _ []tensai.Float) {
		// GeluBwd writes rather than accumulates, so it needs a scratch row.
		tmp := n.tape.buffer(len(grad))
		kernels.GeluBwd(tmp, grad, src)
		kernels.AddSlice(dst, tmp)
	})
}

// Exp applies e^x element-wise.
func (n *Node) Exp() *Node {
	return n.unary("exp", func(dst, src []tensai.Float) {
		for i, x := range src {
			dst[i] = kernels.ExpF(x)
		}
	}, func(dst, grad, _, out []tensai.Float) {
		for i, gv := range grad {
			dst[i] += gv * out[i]
		}
	})
}

// Log applies the natural logarithm element-wise.
func (n *Node) Log() *Node {
	return n.unary("log", func(dst, src []tensai.Float) {
		for i, x := range src {
			dst[i] = kernels.LogF(x)
		}
	}, func(dst, grad, src, _ []tensai.Float) {
		for i, gv := range grad {
			dst[i] += gv / src[i]
		}
	})
}

// T returns n with its last two axes swapped: the matrix transpose of every
// matrix in the stack.
func (n *Node) T() *Node { return n.Transpose() }

// Transpose permutes the axes of n; perm must list every axis exactly once.
// With no arguments it swaps the last two axes.
func (n *Node) Transpose(perm ...int) *Node {
	full := perm
	if len(full) == 0 {
		full = swapLastTwo(len(n.Shape()))
	}
	if out, ok := devTranspose(n, full); ok {
		return out
	}
	v, err := n.Value().Transpose(perm...)
	if err != nil {
		panic(err.Error())
	}
	if len(perm) == 0 {
		perm = swapLastTwo(len(n.Shape()))
	}
	inv := make([]int, len(perm))
	for i, p := range perm {
		inv[p] = i
	}
	return newNode("transpose", v, n).withBack(func(out *Node) {
		d, err := out.Grad().Transpose(inv...)
		if err != nil {
			panic(err.Error())
		}
		n.accum(d)
	})
}

// swapLastTwo returns the permutation that swaps the last two of n axes.
func swapLastTwo(n int) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	perm[n-2], perm[n-1] = perm[n-1], perm[n-2]
	return perm
}

// Reshape returns n with a new shape; one dimension may be -1 and is
// inferred. The value is a view sharing the input's buffer -- as elsewhere
// in the engine, node values are read-only once built -- so this is the
// cheap way in and out of the per-head layout attention wants.
func (n *Node) Reshape(shape ...int) *Node {
	if out, ok := devReshape(n, resolveShape(n.Shape(), shape)); ok {
		return out
	}
	v, err := n.Value().Reshape(shape...)
	if err != nil {
		panic(err.Error())
	}
	return newNode("reshape", v, n).withBack(func(o *Node) {
		d, err := o.Grad().Reshape(n.Shape()...)
		if err != nil {
			panic(err.Error())
		}
		n.accum(d)
	})
}

// Softmax normalizes the last axis of n into a probability distribution.
func (n *Node) Softmax() *Node {
	if out, ok := devSoftmax(n); ok {
		return out
	}
	shape := n.Shape()
	d := shape[len(shape)-1]
	v := n.tape.tensor(shape) // every element is written below
	for pos := 0; pos < len(v.Data); pos += d {
		row := n.Value().Data[pos : pos+d]
		outRow := v.Data[pos : pos+d]
		maxVal := row[0]
		for _, x := range row {
			if x > maxVal {
				maxVal = x
			}
		}
		kernels.ExpShift(outRow, row, maxVal)
		var denom tensai.Float
		for _, e := range outRow {
			denom += e
		}
		kernels.ScaleSlice(outRow, 1/denom)
	}
	return newNode("softmax", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for pos := 0; pos < len(v.Data); pos += d {
			kernels.SoftmaxBwdAdd(g.Data[pos:pos+d], out.Grad().Data[pos:pos+d], v.Data[pos:pos+d])
		}
	})
}

// Sum reduces n to a single-element node by summing every element.
func (n *Node) Sum() *Node {
	if out, ok := devSum(n); ok {
		return out
	}
	var total tensai.Float
	for _, x := range n.Value().Data {
		total += x
	}
	v := n.tape.tensor(scalarShape)
	v.Data[0] = total
	return newNode("sum", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		gv := out.Grad().Data[0]
		for i := range g.Data {
			g.Data[i] += gv
		}
	})
}

// Mean reduces n to a single-element node averaging every element.
func (n *Node) Mean() *Node {
	return n.Sum().Scale(1 / tensai.Float(len(n.Value().Data)))
}

// axisSpan splits a shape around one axis into (outer, dim, inner) element
// counts. A negative axis counts from the end, so -1 is the last axis.
func axisSpan(shape []int, axis int) (int, int, int, int) {
	a := axis
	if a < 0 {
		a += len(shape)
	}
	if a < 0 || a >= len(shape) {
		panic(fmt.Sprintf("tensai: axis %d out of range for shape %s", axis, shapeString(shape)))
	}
	return dims.Prod(shape[:a]), shape[a], dims.Prod(shape[a+1:]), a
}

// SumAxis sums n along one axis. With keepDims the axis stays as a
// length-1 dimension; otherwise it is dropped.
func (n *Node) SumAxis(axis int, keepDims bool) *Node {
	outer, dim, inner, a := axisSpan(n.Shape(), axis)
	v := n.tape.zeros(reducedShape(n.Shape(), a, keepDims)) // accumulated into
	for o := 0; o < outer; o++ {
		dst := v.Data[o*inner : (o+1)*inner]
		for d := 0; d < dim; d++ {
			src := n.Value().Data[(o*dim+d)*inner : (o*dim+d+1)*inner]
			kernels.AddSlice(dst, src)
		}
	}
	return newNode("sumaxis", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for o := 0; o < outer; o++ {
			src := out.Grad().Data[o*inner : (o+1)*inner]
			for d := 0; d < dim; d++ {
				kernels.AddSlice(g.Data[(o*dim+d)*inner:(o*dim+d+1)*inner], src)
			}
		}
	})
}

// MeanAxis averages n along one axis, following SumAxis for keepDims.
func (n *Node) MeanAxis(axis int, keepDims bool) *Node {
	_, dim, _, _ := axisSpan(n.Shape(), axis)
	return n.SumAxis(axis, keepDims).Scale(1 / tensai.Float(dim))
}

// reducedShape returns shape with axis a either dropped or kept as 1.
func reducedShape(shape []int, a int, keepDims bool) []int {
	out := make([]int, 0, len(shape))
	for i, d := range shape {
		switch {
		case i != a:
			out = append(out, d)
		case keepDims:
			out = append(out, 1)
		}
	}
	if len(out) == 0 { // reducing a 1-D tensor without keepDims
		out = append(out, 1)
	}
	return out
}

// LayerNorm normalizes the last axis of n to zero mean and unit variance,
// then scales by gain and shifts by bias, either of which may be nil. Both
// must hold one element per feature -- shape (d) or (1, d) -- and both may
// be parameters, so a transformer block's normalization trains along with
// the rest.
func (n *Node) LayerNorm(gain, bias *Node, eps tensai.Float) *Node {
	if out, ok := devLayerNorm(n, gain, bias, eps); ok {
		return out
	}
	shape := n.Shape()
	d := shape[len(shape)-1]
	gamma := affineData(gain, d, 1, "gain")
	beta := affineData(bias, d, 0, "bias")

	rows := len(n.Value().Data) / d
	tp := tapeOf(n, gain, bias)
	v := tp.tensor(shape) // LnFwdRow writes every element
	xhat := tp.buffer(len(n.Value().Data))
	invStd := tp.buffer(rows)
	for r := 0; r < rows; r++ {
		lo, hi := r*d, (r+1)*d
		invStd[r] = kernels.LnFwdRow(v.Data[lo:hi], xhat[lo:hi], n.Value().Data[lo:hi], gamma, beta, eps)
	}

	parents := []*Node{n}
	for _, p := range []*Node{gain, bias} {
		if p != nil {
			parents = append(parents, p)
		}
	}
	return newNode("layernorm", v, parents...).withBack(func(out *Node) {
		dx := tp.buffer(d)
		gradGamma, gradBeta := tp.zeros(featureShape(d)).Data, tp.zeros(featureShape(d)).Data
		if gain != nil && gain.requiresGrad {
			gradGamma = gain.ensureGrad().Data
		}
		if bias != nil && bias.requiresGrad {
			gradBeta = bias.ensureGrad().Data
		}
		// The gain and bias gradients are wanted even when the normalized
		// input is a constant, so dx is computed either way and simply
		// dropped when nothing upstream needs it.
		var g *tensai.Tensor
		if n.requiresGrad {
			g = n.ensureGrad()
		}
		for r := 0; r < rows; r++ {
			lo, hi := r*d, (r+1)*d
			// LnBwdRow writes dx and accumulates the parameter gradients.
			kernels.LnBwdRow(dx, out.Grad().Data[lo:hi], xhat[lo:hi], gamma, gradGamma, gradBeta, invStd[r])
			if g != nil {
				kernels.AddSlice(g.Data[lo:hi], dx)
			}
		}
	})
}

// affineData returns the per-feature data of a LayerNorm gain or bias node,
// or a constant fill when the node is nil.
func affineData(p *Node, d int, fill tensai.Float, what string) []tensai.Float {
	if p == nil {
		out := make([]tensai.Float, d)
		for i := range out {
			out[i] = fill
		}
		return out
	}
	if len(p.Value().Data) != d {
		panic(fmt.Sprintf("tensai: layernorm %s has %d elements, want %d", what, len(p.Value().Data), d))
	}
	return p.Value().Data
}

// Embed looks ids up in n, a (vocab, dim) embedding table, and returns the
// rows stacked in the given index shape: with shape (batch, seq) the result
// is (batch, seq, dim). With no shape the result is (len(ids), dim).
// Gradients scatter back into the table, so repeated ids accumulate.
func (n *Node) Embed(ids []int, shape ...int) *Node {
	if len(n.Shape()) != 2 {
		panic(fmt.Sprintf("tensai: embed needs a (vocab, dim) table, got %s", shapeString(n.Shape())))
	}
	vocab, d := n.Shape()[0], n.Shape()[1]
	if len(shape) == 0 {
		shape = []int{len(ids)}
	}
	if dims.Prod(shape) != len(ids) {
		panic(fmt.Sprintf("tensai: embed has %d ids for index shape %s", len(ids), shapeString(shape)))
	}
	for _, id := range ids {
		if id < 0 || id >= vocab {
			panic(fmt.Sprintf("tensai: embed id %d out of range for vocab %d", id, vocab))
		}
	}
	outShape := append(append(make([]int, 0, len(shape)+1), shape...), d)
	if out, ok := devEmbed(n, ids, outShape); ok {
		return out
	}
	v := n.tape.tensor(outShape)
	for i, id := range ids {
		copy(v.Data[i*d:(i+1)*d], n.Value().Data[id*d:(id+1)*d])
	}
	return newNode("embed", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for i, id := range ids {
			kernels.AddSlice(g.Data[id*d:(id+1)*d], out.Grad().Data[i*d:(i+1)*d])
		}
	})
}

// MSELoss returns the scalar mean squared error against a constant target
// of the same shape.
func (n *Node) MSELoss(target *tensai.Tensor) *Node {
	if !dims.Same(n.Shape(), target.Shape) {
		panic(fmt.Sprintf("tensai: mse target is %s, want %s",
			shapeString(target.Shape), shapeString(n.Shape())))
	}
	diff := n.Sub(Input(target))
	return diff.Mul(diff).Mean()
}

// SoftmaxCELoss returns the scalar softmax cross-entropy of n's last axis
// against integer class labels, matching the loss.SoftmaxCrossEntropy used
// by Sequential models. n holds logits shaped (..., classes) and target
// holds one class index per row of that stack, in any shape.
func (n *Node) SoftmaxCELoss(target *tensai.Tensor) *Node {
	shape := n.Shape()
	classes := shape[len(shape)-1]
	rows := len(n.Value().Data) / classes
	if len(target.Data) != rows {
		panic(fmt.Sprintf("tensai: softmax-ce target has %d labels, want %d", len(target.Data), rows))
	}
	logits := flatRows(n.Value(), rows, classes)
	labels := flatRows(target, rows, 1)
	lossVal, grad, err := loss.SoftmaxCrossEntropy{}.Loss(logits, labels)
	if err != nil {
		panic(err.Error())
	}
	v := n.tape.tensor(scalarShape)
	v.Data[0] = lossVal
	return newNode("softmax_ce", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		scale := out.Grad().Data[0]
		for i, gv := range grad.Data {
			g.Data[i] += scale * gv
		}
	})
}

// CrossEntropy is SoftmaxCELoss taking the labels as plain ints, which is
// how token targets usually arrive.
func (n *Node) CrossEntropy(labels []int) *Node {
	m, err := tensai.NewMatrixFromInts(len(labels), 1, labels)
	if err != nil {
		panic(err.Error())
	}
	return n.SoftmaxCELoss(m.Tensor())
}

// flatRows views a tensor's data as a rows x cols matrix without copying.
func flatRows(t *tensai.Tensor, rows, cols int) *tensai.Matrix {
	r, err := t.Reshape(rows, cols)
	if err != nil {
		panic(err.Error())
	}
	m, err := r.Matrix()
	if err != nil {
		panic(err.Error())
	}
	return m
}

// scalarShape is the shape every single-element node has: reductions and
// losses share it rather than each building its own.
var scalarShape = []int{1, 1}

// featureShape returns the one-axis shape of LayerNorm's per-feature
// buffers.
func featureShape(d int) []int { return []int{d} }

// binShape returns the shape an element-wise op produces. Equal operands --
// the common case -- share a shape header rather than building a new one.
func binShape(a, b []int) []int {
	if dims.Same(a, b) {
		return a
	}
	shape, err := dims.Broadcast(a, b)
	if err != nil {
		panic(err.Error())
	}
	return shape
}

// resolveShape fills in the one dimension a Reshape may leave as -1, so the
// device path knows the shape before it relabels a buffer.
func resolveShape(from, shape []int) []int {
	out := append([]int(nil), shape...)
	known, infer := 1, -1
	for i, d := range out {
		if d == -1 {
			infer = i
			continue
		}
		known *= d
	}
	if infer >= 0 && known > 0 {
		out[infer] = dims.Prod(from) / known
	}
	return out
}

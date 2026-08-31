package tensai

import (
	"errors"
	"fmt"
	"sync"

	"github.com/mattn/tensai/internal/dims"
	"github.com/mattn/tensai/internal/kernels"
)

// Tensor is an n-dimensional, contiguous, row-major array of Float — the
// generalization of Matrix beyond two dimensions. Element-wise arithmetic
// broadcasts NumPy-style: shapes are aligned at their trailing dimensions
// and a dimension of 1 stretches to match the other operand. MatMul
// multiplies stacks of matrices in one call, broadcasting the leading batch
// dimensions the same way.
type Tensor struct {
	Shape []int
	Data  []Float
}

// NewTensor creates a tensor of the given shape filled with zeros.
func NewTensor(shape ...int) *Tensor {
	return &Tensor{
		Shape: append([]int(nil), shape...),
		Data:  make([]Float, dims.Prod(shape)),
	}
}

// NewTensorFromSlice creates a tensor of the given shape from row-major data.
func NewTensorFromSlice(data []Float, shape ...int) (*Tensor, error) {
	if len(data) != dims.Prod(shape) {
		return nil, fmt.Errorf("tensai: data length %d != shape %v", len(data), shape)
	}
	t := NewTensor(shape...)
	copy(t.Data, data)
	return t, nil
}

// newTensorShape wraps a shape slice the caller has just built and will not
// touch again, skipping the copy NewTensor makes. Every tensor produced by
// an operation goes through here: at these sizes a second small allocation
// per op is a measurable share of the garbage a training step makes.
func newTensorShape(shape []int) *Tensor {
	return &Tensor{Shape: shape, Data: make([]Float, dims.Prod(shape))}
}

// Clone returns a copy of t that shares nothing with it: the way to keep a
// value that would otherwise be recycled or overwritten.
func (t *Tensor) Clone() *Tensor {
	out := NewTensor(t.Shape...)
	copy(out.Data, t.Data)
	return out
}

// ZerosLike returns a zero tensor with the same shape as t. The two share
// the shape header, which every operation treats as read-only, so the
// result costs one allocation instead of two.
func (t *Tensor) ZerosLike() *Tensor {
	return &Tensor{Shape: t.Shape, Data: make([]Float, len(t.Data))}
}

// contiguousStrides returns the row-major element strides of a shape.
func contiguousStrides(shape []int) []int {
	s := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = stride
		stride *= shape[i]
	}
	return s
}

// Size returns the total number of elements.
func (t *Tensor) Size() int { return dims.Prod(t.Shape) }

func (t *Tensor) offset(idx []int) int {
	if len(idx) != len(t.Shape) {
		panic(fmt.Sprintf("tensai: tensor index %v does not match shape %v", idx, t.Shape))
	}
	off := 0
	for d, i := range idx {
		off = off*t.Shape[d] + i
	}
	return off
}

// At returns the element at the given multi-index.
func (t *Tensor) At(idx ...int) Float { return t.Data[t.offset(idx)] }

// Set sets the element at the given multi-index.
func (t *Tensor) Set(v Float, idx ...int) { t.Data[t.offset(idx)] = v }

// Scale multiplies every element by s, in place.
func (t *Tensor) Scale(s Float) { kernels.ScaleSlice(t.Data, s) }

// Reshape returns a tensor with a new shape sharing the same backing data.
// One dimension may be -1 and is inferred from the element count.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	shape = append([]int(nil), shape...)
	infer, known := -1, 1
	for d, s := range shape {
		switch {
		case s == -1:
			if infer >= 0 {
				return nil, fmt.Errorf("tensai: reshape %v has more than one -1", shape)
			}
			infer = d
		case s <= 0:
			return nil, fmt.Errorf("tensai: reshape invalid dimension %d in %v", s, shape)
		default:
			known *= s
		}
	}
	if infer >= 0 {
		if len(t.Data)%known != 0 {
			return nil, fmt.Errorf("tensai: cannot reshape %d elements to %v", len(t.Data), shape)
		}
		shape[infer] = len(t.Data) / known
		known *= shape[infer]
	}
	if known != len(t.Data) {
		return nil, fmt.Errorf("tensai: cannot reshape %d elements to %v", len(t.Data), shape)
	}
	return &Tensor{Shape: shape, Data: t.Data}, nil
}

// Tensor returns a 2-D tensor view of the matrix sharing the same backing
// data.
func (m *Matrix) Tensor() *Tensor {
	return &Tensor{Shape: []int{m.Rows, m.Cols}, Data: m.Data}
}

// Matrix returns a matrix view of a 2-D tensor sharing the same backing
// data.
func (t *Tensor) Matrix() (*Matrix, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("tensai: cannot view shape %v as a matrix", t.Shape)
	}
	return &Matrix{Rows: t.Shape[0], Cols: t.Shape[1], Data: t.Data}, nil
}

// Validate returns an error if the tensor shape or data length is
// inconsistent.
func (t *Tensor) Validate() error {
	if t == nil {
		return errors.New("tensai: nil tensor")
	}
	for _, d := range t.Shape {
		if d <= 0 {
			return fmt.Errorf("tensai: tensor has invalid shape %v", t.Shape)
		}
	}
	if len(t.Data) != dims.Prod(t.Shape) {
		return fmt.Errorf("tensai: tensor %v has %d elements", t.Shape, len(t.Data))
	}
	return nil
}

// binKind names the element-wise binary ops tensors broadcast with. The
// ops are dispatched per row rather than per element: a row is handed to a
// vector kernel whole, so the arithmetic runs 8 lanes wide instead of
// paying an indirect call per element.
type binKind int

const (
	binAdd binKind = iota
	binSub
	binMul
	binDiv
)

// rows computes dst = x op y over a whole row.
func (k binKind) rows(dst, x, y []Float) {
	switch k {
	case binAdd:
		kernels.AddSlices(dst, x, y)
	case binSub:
		kernels.SubSlices(dst, x, y)
	case binMul:
		kernels.MulSlices(dst, x, y)
	default:
		kernels.DivSlices(dst, x, y)
	}
}

// rowScalarY computes dst = x op y with y broadcast along the row.
func (k binKind) rowScalarY(dst, x []Float, y Float) {
	switch k {
	case binAdd:
		for i, v := range x {
			dst[i] = v + y
		}
	case binSub:
		for i, v := range x {
			dst[i] = v - y
		}
	case binMul:
		for i, v := range x {
			dst[i] = v * y
		}
	default:
		for i, v := range x {
			dst[i] = v / y
		}
	}
}

// rowScalarX computes dst = x op y with x broadcast along the row.
func (k binKind) rowScalarX(dst []Float, x Float, y []Float) {
	switch k {
	case binAdd:
		for i, v := range y {
			dst[i] = x + v
		}
	case binSub:
		for i, v := range y {
			dst[i] = x - v
		}
	case binMul:
		for i, v := range y {
			dst[i] = x * v
		}
	default:
		for i, v := range y {
			dst[i] = x / v
		}
	}
}

// scalar computes one element.
func (k binKind) scalar(x, y Float) Float {
	switch k {
	case binAdd:
		return x + y
	case binSub:
		return x - y
	case binMul:
		return x * y
	default:
		return x / y
	}
}

// tensorBinOp applies op element-wise over the broadcast of a and b.
func tensorBinOp(a, b *Tensor, kind binKind) (*Tensor, error) {
	if dims.Same(a.Shape, b.Shape) {
		// Equal shapes are the common case and need no broadcast plan at
		// all: one kernel call over the whole buffer.
		out := a.ZerosLike()
		kind.rows(out.Data, a.Data, b.Data)
		return out, nil
	}
	shape, err := dims.Broadcast(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	out := newTensorShape(shape)
	binBroadcast(out, a, b, shape, kind)
	return out, nil
}

// AddInto, SubInto, MulInto and DivInto write a op b into an existing
// tensor instead of allocating one, the way DotInto does for products. out
// must already have the broadcast shape of the two operands; a training
// loop that hands the same buffers back every step never allocates here.
func AddInto(out, a, b *Tensor) error { return tensorBinOpInto(out, a, b, binAdd) }
func SubInto(out, a, b *Tensor) error { return tensorBinOpInto(out, a, b, binSub) }
func MulInto(out, a, b *Tensor) error { return tensorBinOpInto(out, a, b, binMul) }
func DivInto(out, a, b *Tensor) error { return tensorBinOpInto(out, a, b, binDiv) }

func tensorBinOpInto(out, a, b *Tensor, kind binKind) error {
	if dims.Same(a.Shape, b.Shape) {
		if !dims.Same(out.Shape, a.Shape) {
			return fmt.Errorf("tensai: binop output shape %v, want %v", out.Shape, a.Shape)
		}
		kind.rows(out.Data, a.Data, b.Data)
		return nil
	}
	shape, err := dims.Broadcast(a.Shape, b.Shape)
	if err != nil {
		return err
	}
	if !dims.Same(out.Shape, shape) {
		return fmt.Errorf("tensai: binop output shape %v, want %v", out.Shape, shape)
	}
	binBroadcast(out, a, b, shape, kind)
	return nil
}

// binBroadcast walks the broadcast of a and b, writing each output row with
// the row kernel of the op.
func binBroadcast(out, a, b *Tensor, shape []int, kind binKind) {
	as := dims.BroadcastStrides(a.Shape, shape)
	bs := dims.BroadcastStrides(b.Shape, shape)
	// Walk the output row-major: the innermost axis runs as a tight loop
	// (each operand's innermost stride is 1, or 0 when broadcast) and an
	// odometer over the outer axes carries the operand offsets.
	last := len(shape) - 1
	n := shape[last]
	asl, bsl := as[last], bs[last]
	idx := make([]int, len(shape))
	offA, offB := 0, 0
	for pos := 0; pos < len(out.Data); pos += n {
		dst := out.Data[pos : pos+n]
		switch {
		case asl == 1 && bsl == 1:
			kind.rows(dst, a.Data[offA:offA+n], b.Data[offB:offB+n])
		case asl == 1:
			kind.rowScalarY(dst, a.Data[offA:offA+n], b.Data[offB])
		case bsl == 1:
			kind.rowScalarX(dst, a.Data[offA], b.Data[offB:offB+n])
		default: // both broadcast, so n == 1
			dst[0] = kind.scalar(a.Data[offA], b.Data[offB])
		}
		for d := last - 1; d >= 0; d-- {
			idx[d]++
			offA += as[d]
			offB += bs[d]
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
			offA -= as[d] * shape[d]
			offB -= bs[d] * shape[d]
		}
	}
}

// Add returns t + o element-wise with broadcasting.
func (t *Tensor) Add(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, binAdd)
}

// Sub returns t - o element-wise with broadcasting.
func (t *Tensor) Sub(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, binSub)
}

// Mul returns t * o element-wise with broadcasting.
func (t *Tensor) Mul(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, binMul)
}

// Div returns t / o element-wise with broadcasting, with IEEE semantics for
// division by zero.
func (t *Tensor) Div(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, binDiv)
}

// Transpose returns a copy of the tensor with its axes permuted; perm must
// list every axis exactly once. With no arguments it swaps the last two
// axes — the matrix transpose of every matrix in the stack — matching
// Matrix.T for 2-D tensors.
func (t *Tensor) Transpose(perm ...int) (*Tensor, error) {
	n := len(t.Shape)
	if len(perm) == 0 {
		if n < 2 {
			return nil, fmt.Errorf("tensai: transpose needs at least 2 axes, shape %v", t.Shape)
		}
		return t.swapLastTwo(), nil
	}
	if len(perm) != n {
		return nil, fmt.Errorf("tensai: transpose permutation %v does not match shape %v", perm, t.Shape)
	}
	seen := make([]bool, n)
	for _, p := range perm {
		if p < 0 || p >= n || seen[p] {
			return nil, fmt.Errorf("tensai: invalid transpose permutation %v", perm)
		}
		seen[p] = true
	}
	if isLastTwoSwap(perm) {
		return t.swapLastTwo(), nil
	}
	outShape := make([]int, n)
	strides := contiguousStrides(t.Shape)
	src := make([]int, n) // stride in t along each output axis
	for i, p := range perm {
		outShape[i] = t.Shape[p]
		src[i] = strides[p]
	}
	out := newTensorShape(outShape)
	last := n - 1
	nLast, sLast := outShape[last], src[last]
	idx := make([]int, n)
	off := 0
	for pos := 0; pos < len(out.Data); pos += nLast {
		dst := out.Data[pos : pos+nLast]
		if sLast == 1 {
			copy(dst, t.Data[off:off+nLast])
		} else {
			so := off
			for i := range dst {
				dst[i] = t.Data[so]
				so += sLast
			}
		}
		for d := last - 1; d >= 0; d-- {
			idx[d]++
			off += src[d]
			if idx[d] < outShape[d] {
				break
			}
			idx[d] = 0
			off -= src[d] * outShape[d]
		}
	}
	return out, nil
}

// swapLastTwo transposes every matrix in the stack -- the last two axes --
// with the cache-tiled kernel Matrix.T uses, rather than the strided walk
// a general permutation needs.
func (t *Tensor) swapLastTwo() *Tensor {
	n := len(t.Shape)
	rows, cols := t.Shape[n-2], t.Shape[n-1]
	outShape := append(append(make([]int, 0, n), t.Shape[:n-2]...), cols, rows)
	out := newTensorShape(outShape)
	size := rows * cols
	for off := 0; off < len(t.Data); off += size {
		transposeData(out.Data[off:off+size], t.Data[off:off+size], rows, cols)
	}
	return out
}

// isLastTwoSwap reports whether perm leaves every axis in place but the
// last two, which it swaps.
func isLastTwoSwap(perm []int) bool {
	n := len(perm)
	if n < 2 {
		return false
	}
	for i := 0; i < n-2; i++ {
		if perm[i] != i {
			return false
		}
	}
	return perm[n-2] == n-1 && perm[n-1] == n-2
}

// gemmMode selects which operand of a stacked product is transposed.
// Backward passes need all three: the forward product, the input gradient
// (a * b^T) and the weight gradient (a^T * b). Naming them here means none
// of them has to materialize a transposed copy first.
type gemmMode int

const (
	gemmNN gemmMode = iota // a * b
	gemmTN                 // a^T * b
	gemmNT                 // a * b^T
)

// gemmInto runs one matrix product of the given mode, splitting it across
// CPUs when it is large enough.
func gemmInto(out, a, b *Matrix, mode gemmMode) error {
	switch mode {
	case gemmTN:
		return DotTAInto(out, a, b)
	case gemmNT:
		return DotTBInto(out, a, b)
	default:
		return DotInto(out, a, b)
	}
}

// gemmRows runs one matrix product of the given mode on the calling
// goroutine; the batched path parallelizes across matrices instead.
func gemmRows(out, a, b *Matrix, mode gemmMode) {
	switch mode {
	case gemmTN:
		clear(out.Data) // dotTARows accumulates into out
		dotTARows(out, a, b, 0, a.Cols)
	case gemmNT:
		dotTBRows(out, a, b, 0, a.Rows)
	default:
		dotRows(out, a, b, 0, a.Rows)
	}
}

// MatMul multiplies two stacks of matrices: the last two axes of each
// operand are the matrix dimensions and the leading axes broadcast like the
// element-wise ops, so a (batch..., m, k) tensor times a (batch..., k, n)
// tensor yields (batch..., m, n). Both operands need at least 2 axes. The
// per-matrix products run on the same kernel as Dot, parallelized across
// the batch.
func MatMul(a, b *Tensor) (*Tensor, error) { return matmulStack(a, b, gemmNN) }

// MatMulTN multiplies the transpose of every matrix in a by the matching
// matrix in b: a is (batch..., k, m), b is (batch..., k, n), and the result
// is (batch..., m, n). This is the weight gradient of a matmul, computed
// without transposing a first.
func MatMulTN(a, b *Tensor) (*Tensor, error) { return matmulStack(a, b, gemmTN) }

// MatMulNT multiplies every matrix in a by the transpose of the matching
// matrix in b: a is (batch..., m, k), b is (batch..., n, k), and the result
// is (batch..., m, n). This is the input gradient of a matmul, and also the
// q * k^T of attention, computed without transposing b first.
func MatMulNT(a, b *Tensor) (*Tensor, error) { return matmulStack(a, b, gemmNT) }

// MatMulInto, MatMulTNInto and MatMulNTInto write the product into an
// existing tensor rather than allocating one, like DotInto one rank down.
// out must already have the product's shape.
func MatMulInto(out, a, b *Tensor) error   { return matmulStackInto(out, a, b, gemmNN) }
func MatMulTNInto(out, a, b *Tensor) error { return matmulStackInto(out, a, b, gemmTN) }
func MatMulNTInto(out, a, b *Tensor) error { return matmulStackInto(out, a, b, gemmNT) }

// gemmDims holds the per-matrix geometry of a stacked product: the logical
// product is (m, k) * (k, n), while aRows/aCols and bRows/bCols are how the
// operands are actually stored (swapped for a transposed mode).
type gemmDims struct {
	m, k, n                    int
	aRows, aCols, bRows, bCols int
}

// gemmResolve works out that geometry and checks the inner dimensions.
func gemmResolve(a, b *Tensor, mode gemmMode) (gemmDims, error) {
	na, nb := len(a.Shape), len(b.Shape)
	if na < 2 || nb < 2 {
		return gemmDims{}, fmt.Errorf("tensai: matmul needs at least 2 axes: %v * %v", a.Shape, b.Shape)
	}
	d := gemmDims{
		aRows: a.Shape[na-2], aCols: a.Shape[na-1],
		bRows: b.Shape[nb-2], bCols: b.Shape[nb-1],
	}
	switch mode {
	case gemmTN:
		d.k, d.m, d.n = d.aRows, d.aCols, d.bCols
		if d.bRows != d.k {
			return gemmDims{}, fmt.Errorf("tensai: matmul shape mismatch: (%v)^T * %v", a.Shape, b.Shape)
		}
	case gemmNT:
		d.m, d.k, d.n = d.aRows, d.aCols, d.bRows
		if d.bCols != d.k {
			return gemmDims{}, fmt.Errorf("tensai: matmul shape mismatch: %v * (%v)^T", a.Shape, b.Shape)
		}
	default:
		d.m, d.k, d.n = d.aRows, d.aCols, d.bCols
		if d.bRows != d.k {
			return gemmDims{}, fmt.Errorf("tensai: matmul shape mismatch: %v * %v", a.Shape, b.Shape)
		}
	}
	return d, nil
}

// matmulStack is the shared body of the three allocating products above.
func matmulStack(a, b *Tensor, mode gemmMode) (*Tensor, error) {
	d, err := gemmResolve(a, b, mode)
	if err != nil {
		return nil, err
	}
	na, nb := len(a.Shape), len(b.Shape)
	if na == 2 && nb == 2 {
		out := newTensorShape([]int{d.m, d.n})
		return out, gemmExec(out, a, b, d, mode, nil)
	}
	batch, err := dims.Broadcast(a.Shape[:na-2], b.Shape[:nb-2])
	if err != nil {
		return nil, err
	}
	out := newTensorShape(append(append(make([]int, 0, len(batch)+2), batch...), d.m, d.n))
	return out, gemmExec(out, a, b, d, mode, batch)
}

// matmulStackInto is the same product written into a caller-owned tensor.
func matmulStackInto(out, a, b *Tensor, mode gemmMode) error {
	d, err := gemmResolve(a, b, mode)
	if err != nil {
		return err
	}
	na, nb := len(a.Shape), len(b.Shape)
	no := len(out.Shape)
	if no < 2 || out.Shape[no-2] != d.m || out.Shape[no-1] != d.n {
		return fmt.Errorf("tensai: matmul output shape %v, want (..., %d, %d)", out.Shape, d.m, d.n)
	}
	var batch []int
	if na > 2 || nb > 2 {
		if batch, err = dims.Broadcast(a.Shape[:na-2], b.Shape[:nb-2]); err != nil {
			return err
		}
		if !dims.Same(out.Shape[:no-2], batch) {
			return fmt.Errorf("tensai: matmul output shape %v, want batch %v", out.Shape, batch)
		}
	} else if no != 2 {
		return fmt.Errorf("tensai: matmul output shape %v, want (%d, %d)", out.Shape, d.m, d.n)
	}
	return gemmExec(out, a, b, d, mode, batch)
}

// gemmExec runs the product. batch is nil when both operands are plain
// matrices; otherwise it is the broadcast shape of the leading axes.
func gemmExec(out, a, b *Tensor, d gemmDims, mode gemmMode, batch []int) error {
	view := func(rows, cols int, data []Float) *Matrix {
		return &Matrix{Rows: rows, Cols: cols, Data: data}
	}
	batches := dims.Prod(batch)
	if batches == 1 {
		// One product: let the kernel split its rows across CPUs.
		return gemmInto(view(d.m, d.n, out.Data),
			view(d.aRows, d.aCols, a.Data), view(d.bRows, d.bCols, b.Data), mode)
	}
	// Strides are in whole matrices; broadcast batch axes advance by 0.
	na, nb := len(a.Shape), len(b.Shape)
	as := dims.BroadcastStrides(a.Shape[:na-2], batch)
	bs := dims.BroadcastStrides(b.Shape[:nb-2], batch)
	sizeA, sizeB, sizeO := d.aRows*d.aCols, d.bRows*d.bCols, d.m*d.n
	run := func(lo, hi int) {
		for bi := lo; bi < hi; bi++ {
			offA, offB := 0, 0
			for ax, rem := len(batch)-1, bi; ax >= 0; ax-- {
				i := rem % batch[ax]
				rem /= batch[ax]
				offA += i * as[ax]
				offB += i * bs[ax]
			}
			gemmRows(view(d.m, d.n, out.Data[bi*sizeO:(bi+1)*sizeO]),
				view(d.aRows, d.aCols, a.Data[offA*sizeA:(offA+1)*sizeA]),
				view(d.bRows, d.bCols, b.Data[offB*sizeB:(offB+1)*sizeB]), mode)
		}
	}
	workers := dotWorkerCount(batches, sizeA, d.n)
	if workers == 1 {
		run(0, batches)
		return nil
	}
	chunk := (batches + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, (w+1)*chunk
		if hi > batches {
			hi = batches
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			run(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

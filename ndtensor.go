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

// tensorBinOp applies fn element-wise over the broadcast of a and b.
func tensorBinOp(a, b *Tensor, fn func(x, y Float) Float) (*Tensor, error) {
	shape, err := dims.Broadcast(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	out := NewTensor(shape...)
	if dims.Same(a.Shape, shape) && dims.Same(b.Shape, shape) {
		for i := range out.Data {
			out.Data[i] = fn(a.Data[i], b.Data[i])
		}
		return out, nil
	}
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
			av, bv := a.Data[offA:offA+n], b.Data[offB:offB+n]
			for i := range dst {
				dst[i] = fn(av[i], bv[i])
			}
		case asl == 1:
			av, bv := a.Data[offA:offA+n], b.Data[offB]
			for i := range dst {
				dst[i] = fn(av[i], bv)
			}
		case bsl == 1:
			av, bv := a.Data[offA], b.Data[offB:offB+n]
			for i := range dst {
				dst[i] = fn(av, bv[i])
			}
		default: // both broadcast, so n == 1
			dst[0] = fn(a.Data[offA], b.Data[offB])
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
	return out, nil
}

// Add returns t + o element-wise with broadcasting.
func (t *Tensor) Add(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, func(x, y Float) Float { return x + y })
}

// Sub returns t - o element-wise with broadcasting.
func (t *Tensor) Sub(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, func(x, y Float) Float { return x - y })
}

// Mul returns t * o element-wise with broadcasting.
func (t *Tensor) Mul(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, func(x, y Float) Float { return x * y })
}

// Div returns t / o element-wise with broadcasting, with IEEE semantics for
// division by zero.
func (t *Tensor) Div(o *Tensor) (*Tensor, error) {
	return tensorBinOp(t, o, func(x, y Float) Float { return x / y })
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
		perm = make([]int, n)
		for i := range perm {
			perm[i] = i
		}
		perm[n-2], perm[n-1] = perm[n-1], perm[n-2]
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
	outShape := make([]int, n)
	strides := contiguousStrides(t.Shape)
	src := make([]int, n) // stride in t along each output axis
	for i, p := range perm {
		outShape[i] = t.Shape[p]
		src[i] = strides[p]
	}
	out := NewTensor(outShape...)
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

// MatMul multiplies two stacks of matrices: the last two axes of each
// operand are the matrix dimensions and the leading axes broadcast like the
// element-wise ops, so a (batch..., m, k) tensor times a (batch..., k, n)
// tensor yields (batch..., m, n). Both operands need at least 2 axes. The
// per-matrix products run on the same kernel as Dot, parallelized across
// the batch.
func MatMul(a, b *Tensor) (*Tensor, error) {
	na, nb := len(a.Shape), len(b.Shape)
	if na < 2 || nb < 2 {
		return nil, fmt.Errorf("tensai: matmul needs at least 2 axes: %v * %v", a.Shape, b.Shape)
	}
	m, k := a.Shape[na-2], a.Shape[na-1]
	if b.Shape[nb-2] != k {
		return nil, fmt.Errorf("tensai: matmul shape mismatch: %v * %v", a.Shape, b.Shape)
	}
	n := b.Shape[nb-1]
	batch, err := dims.Broadcast(a.Shape[:na-2], b.Shape[:nb-2])
	if err != nil {
		return nil, err
	}
	out := NewTensor(append(append([]int(nil), batch...), m, n)...)
	batches := dims.Prod(batch)
	if batches == 1 {
		// A single product: let DotInto split its rows across CPUs.
		am := &Matrix{Rows: m, Cols: k, Data: a.Data}
		bm := &Matrix{Rows: k, Cols: n, Data: b.Data}
		om := &Matrix{Rows: m, Cols: n, Data: out.Data}
		if err := DotInto(om, am, bm); err != nil {
			return nil, err
		}
		return out, nil
	}
	// Strides are in whole matrices; broadcast batch axes advance by 0.
	as := dims.BroadcastStrides(a.Shape[:na-2], batch)
	bs := dims.BroadcastStrides(b.Shape[:nb-2], batch)
	sizeA, sizeB, sizeO := m*k, k*n, m*n
	run := func(lo, hi int) {
		for bi := lo; bi < hi; bi++ {
			offA, offB := 0, 0
			for d, rem := len(batch)-1, bi; d >= 0; d-- {
				i := rem % batch[d]
				rem /= batch[d]
				offA += i * as[d]
				offB += i * bs[d]
			}
			am := &Matrix{Rows: m, Cols: k, Data: a.Data[offA*sizeA : (offA+1)*sizeA]}
			bm := &Matrix{Rows: k, Cols: n, Data: b.Data[offB*sizeB : (offB+1)*sizeB]}
			om := &Matrix{Rows: m, Cols: n, Data: out.Data[bi*sizeO : (bi+1)*sizeO]}
			dotRows(om, am, bm, 0, m)
		}
	}
	workers := dotWorkerCount(batches, sizeA, n)
	if workers == 1 {
		run(0, batches)
		return out, nil
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
	return out, nil
}

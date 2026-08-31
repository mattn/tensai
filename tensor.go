package tensai

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"

	"github.com/mattn/tensai/internal/kernels"
)

// Float is the element type of every tensor. float32 halves memory traffic
// versus float64 and enables the 8-lane AVX2 kernel (see dot_simd.go); its
// ~7 decimal digits are plenty for neural-network training.
type Float = float32

// Matrix is a row-major 2D tensor of Float.
type Matrix struct {
	Rows int
	Cols int
	Data []Float
}

// NewMatrix creates a matrix filled with zeros.
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{Rows: rows, Cols: cols, Data: make([]Float, rows*cols)}
}

// NewMatrixFromSlice creates a rows x cols matrix from row-major data.
func NewMatrixFromSlice(rows, cols int, data []Float) (*Matrix, error) {
	if len(data) != rows*cols {
		return nil, fmt.Errorf("tensai: data length %d != %dx%d", len(data), rows, cols)
	}
	m := NewMatrix(rows, cols)
	copy(m.Data, data)
	return m, nil
}

// NewMatrixFromInts builds a matrix from integer values, verifying each one
// survives the float32 conversion exactly. This is the safe way to build
// token-id inputs for an Embedding layer, whose ids travel as Float.
func NewMatrixFromInts(rows, cols int, data []int) (*Matrix, error) {
	if len(data) != rows*cols {
		return nil, fmt.Errorf("tensai: data length %d != %dx%d", len(data), rows, cols)
	}
	m := NewMatrix(rows, cols)
	for i, v := range data {
		f := Float(v)
		if int(f) != v {
			return nil, fmt.Errorf("tensai: value %d at index %d does not fit float32 exactly", v, i)
		}
		m.Data[i] = f
	}
	return m, nil
}

// At returns the element at (r, c).
func (m *Matrix) At(r, c int) Float {
	return m.Data[r*m.Cols+c]
}

// Set sets the element at (r, c).
func (m *Matrix) Set(r, c int, v Float) {
	m.Data[r*m.Cols+c] = v
}

// SetRow copies vals into row r.
func (m *Matrix) SetRow(r int, vals []Float) error {
	if len(vals) != m.Cols {
		return fmt.Errorf("tensai: setrow mismatch: cols=%d vals=%d", m.Cols, len(vals))
	}
	copy(m.Data[r*m.Cols:(r+1)*m.Cols], vals)
	return nil
}

// Row returns a copy of row r as a slice.
func (m *Matrix) Row(r int) []Float {
	out := make([]Float, m.Cols)
	copy(out, m.Data[r*m.Cols:(r+1)*m.Cols])
	return out
}

// ArgmaxRow returns the column index of the largest value in row r; ties
// go to the lowest index. Classification models emit one column per class,
// so this maps a row of scores to its predicted class.
func (m *Matrix) ArgmaxRow(r int) int {
	best := 0
	for c := 1; c < m.Cols; c++ {
		if m.At(r, c) > m.At(r, best) {
			best = c
		}
	}
	return best
}

// T returns the transpose of the matrix.
func (m *Matrix) T() *Matrix {
	out := NewMatrix(m.Cols, m.Rows)
	transposeData(out.Data, m.Data, m.Rows, m.Cols)
	return out
}

// EnsureMatrix returns m when it already has the wanted shape, otherwise a
// freshly allocated matrix. The contents are unspecified; callers must
// overwrite (or clear) every element. Layers use this to reuse forward and
// backward scratch buffers between training steps.
func EnsureMatrix(m *Matrix, rows, cols int) *Matrix {
	if m != nil && m.Rows == rows && m.Cols == cols {
		return m
	}
	return NewMatrix(rows, cols)
}

// Dot computes the matrix product a * b.
func Dot(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, fmt.Errorf("tensai: dot shape mismatch: %dx%d * %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	out := NewMatrix(a.Rows, b.Cols)
	if err := DotInto(out, a, b); err != nil {
		return nil, err
	}
	return out, nil
}

// DotInto computes out = a * b into an existing matrix, overwriting it.
func DotInto(out, a, b *Matrix) error {
	if a.Cols != b.Rows {
		return fmt.Errorf("tensai: dot shape mismatch: %dx%d * %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	if out.Rows != a.Rows || out.Cols != b.Cols {
		return fmt.Errorf("tensai: dot output shape mismatch: got %dx%d, want %dx%d",
			out.Rows, out.Cols, a.Rows, b.Cols)
	}
	// Rows are independent, so large products are split across CPUs. Small
	// ones stay single-threaded to avoid goroutine overhead.
	workers := dotWorkerCount(a.Rows, a.Cols, b.Cols)
	if workers == 1 {
		dotRows(out, a, b, 0, a.Rows)
		return nil
	}
	chunk := (a.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, (w+1)*chunk
		if hi > a.Rows {
			hi = a.Rows
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			dotRows(out, a, b, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// DotTAInto computes out = a^T * b into an existing matrix, overwriting it,
// without materializing the transpose: a is read row by row and scattered
// into out with the same vector kernel Dot uses. Shapes: a is RxI, b is
// RxJ, out is IxJ.
func DotTAInto(out, a, b *Matrix) error {
	if a.Rows != b.Rows {
		return fmt.Errorf("tensai: dotta shape mismatch: %dx%d^T * %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	if out.Rows != a.Cols || out.Cols != b.Cols {
		return fmt.Errorf("tensai: dotta output shape mismatch: got %dx%d, want %dx%d",
			out.Rows, out.Cols, a.Cols, b.Cols)
	}
	clear(out.Data) // dotTARows accumulates
	workers := dotWorkerCount(a.Cols, a.Rows, b.Cols)
	if workers == 1 {
		// Small products stay on this goroutine: waking workers for them
		// costs more than the arithmetic, and a training step runs
		// thousands of small products.
		dotTARows(out, a, b, 0, a.Cols)
		return nil
	}
	chunk := (a.Cols + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, (w+1)*chunk
		if hi > a.Cols {
			hi = a.Cols
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			dotTARows(out, a, b, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// dotTARowsGeneric computes out rows lo..hi of out = a^T * b in pure Go.
// dotTARows (see dot_simd.go and dot_generic.go) dispatches to a SIMD
// kernel when one is available.
func dotTARowsGeneric(out, a, b *Matrix, lo, hi int) {
	for r := 0; r < a.Rows; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		bRow := b.Data[r*b.Cols : (r+1)*b.Cols]
		for i := lo; i < hi; i++ {
			av := aRow[i]
			if av == 0 {
				continue
			}
			outRow := out.Data[i*b.Cols : (i+1)*b.Cols]
			for c, bv := range bRow {
				outRow[c] += av * bv
			}
		}
	}
}

// DotTBInto computes out = a * b^T, the product every backward pass needs
// for the left operand of a matmul. a is (m, k) and b is (n, k): both
// operands are read row-wise, so the whole product runs on the vectorized
// row-dot kernel and no transpose is materialized.
func DotTBInto(out, a, b *Matrix) error {
	if a.Cols != b.Cols {
		return fmt.Errorf("tensai: dottb shape mismatch: %dx%d * (%dx%d)^T", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	if out.Rows != a.Rows || out.Cols != b.Rows {
		return fmt.Errorf("tensai: dottb output shape mismatch: got %dx%d, want %dx%d",
			out.Rows, out.Cols, a.Rows, b.Rows)
	}
	workers := dotWorkerCount(a.Rows, a.Cols, b.Rows)
	if workers == 1 {
		dotTBRows(out, a, b, 0, a.Rows)
		return nil
	}
	chunk := (a.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, (w+1)*chunk
		if hi > a.Rows {
			hi = a.Rows
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			dotTBRows(out, a, b, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// dotTBRows computes rows lo..hi of out = a * b^T. Each output row is one
// row of a dotted against every row of b, which is what DotVecs does.
func dotTBRows(out, a, b *Matrix, lo, hi int) {
	for r := lo; r < hi; r++ {
		kernels.DotVecs(b.Data, a.Data[r*a.Cols:(r+1)*a.Cols], out.Data[r*out.Cols:(r+1)*out.Cols])
	}
}

// TInto writes the transpose of src into dst.
func TInto(dst, src *Matrix) error {
	if dst.Rows != src.Cols || dst.Cols != src.Rows {
		return fmt.Errorf("tensai: transpose shape mismatch: dst %dx%d, src %dx%d",
			dst.Rows, dst.Cols, src.Rows, src.Cols)
	}
	transposeData(dst.Data, src.Data, src.Rows, src.Cols)
	return nil
}

// transposeData uses cache-sized tiles so both the source reads and the
// destination writes stay local. This also gives the SIMD compiler compact
// 8-element runs to optimize instead of one full-matrix strided stream.
func transposeData(dst, src []Float, rows, cols int) {
	const tile = 32
	for r0 := 0; r0 < rows; r0 += tile {
		r1 := min(r0+tile, rows)
		for c0 := 0; c0 < cols; c0 += tile {
			c1 := min(c0+tile, cols)
			for r := r0; r < r1; r++ {
				s := src[r*cols+c0 : r*cols+c1]
				for i, v := range s {
					dst[(c0+i)*rows+r] = v
				}
			}
		}
	}
}

// dotRowsGeneric computes rows lo..hi of out = a * b in pure Go. Loop order
// r,k,c keeps the inner loop sequential over b and out, which is far more
// cache-friendly than the naive r,c,k order. dotRows (see dot_simd.go and
// dot_generic.go) dispatches to a SIMD kernel when one is available.
func dotRowsGeneric(out, a, b *Matrix, lo, hi int) {
	for r := lo; r < hi; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		outRow := out.Data[r*b.Cols : (r+1)*b.Cols]
		initialized := false
		for k, av := range aRow {
			if av == 0 {
				continue
			}
			bRow := b.Data[k*b.Cols : (k+1)*b.Cols]
			if !initialized {
				for c, bv := range bRow {
					outRow[c] = av * bv
				}
				initialized = true
				continue
			}
			for c, bv := range bRow {
				outRow[c] += av * bv
			}
		}
		if !initialized {
			clear(outRow)
		}
	}
}

// Add returns a + b (element-wise). Shapes must match.
func Add(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("tensai: add shape mismatch: %dx%d vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	out := NewMatrix(a.Rows, a.Cols)
	kernels.AddSlices(out.Data, a.Data, b.Data)
	return out, nil
}

// AddBias adds a 1xCols bias vector to every row of a.
func AddBias(a *Matrix, bias []Float) (*Matrix, error) {
	if a.Cols != len(bias) {
		return nil, fmt.Errorf("tensai: addbias mismatch: cols=%d bias=%d", a.Cols, len(bias))
	}
	out := NewMatrix(a.Rows, a.Cols)
	for r := 0; r < a.Rows; r++ {
		lo, hi := r*a.Cols, (r+1)*a.Cols
		kernels.AddSlices(out.Data[lo:hi], a.Data[lo:hi], bias)
	}
	return out, nil
}

// Scale multiplies every element by s, in place.
func (m *Matrix) Scale(s Float) {
	for i := range m.Data {
		m.Data[i] *= s
	}
}

// RandomMatrix fills a matrix with samples from a normal distribution
// scaled by the Glorot/Bengio gain for the given fan-in / fan-out.
func RandomMatrix(rows, cols int, rng *rand.Rand) *Matrix {
	m := NewMatrix(rows, cols)
	// He / Glorot-style scaling keeps early training stable.
	scale := kernels.SqrtF(2.0 / Float(rows+cols))
	for i := range m.Data {
		m.Data[i] = Float(rng.NormFloat64()) * scale
	}
	return m
}

// Validate returns an error if the matrix data length is inconsistent.
func (m *Matrix) Validate() error {
	if m == nil {
		return errors.New("tensai: nil matrix")
	}
	if len(m.Data) != m.Rows*m.Cols {
		return fmt.Errorf("tensai: matrix %dx%d has %d elements", m.Rows, m.Cols, len(m.Data))
	}
	return nil
}

// DotVec returns the dot product of two equally long vectors, running on
// the AVX2 FMA kernel in SIMD builds — the score kernel of attention over
// a KV cache.
func DotVec(a, b []Float) Float {
	if len(a) != len(b) {
		panic("tensai: DotVec length mismatch")
	}
	return kernels.DotVec(a, b)
}

// Axpy computes y += a*x elementwise over equally long vectors — the
// weighted value accumulation of attention.
func Axpy(a Float, x, y []Float) {
	if len(x) != len(y) {
		panic("tensai: Axpy length mismatch")
	}
	kernels.Axpy(a, x, y)
}

// SiluMul computes gate[i] = silu(gate[i]) * up[i] in place — the SwiGLU
// activation between a transformer block's fused gate/up projection and
// its down projection. The AVX2 build evaluates the sigmoid with the
// same polynomial exp the training kernels use, so results can differ
// from the portable build by a few float32 ulps.
func SiluMul(gate, up []Float) {
	if len(gate) != len(up) {
		panic("tensai: SiluMul length mismatch")
	}
	kernels.SiluMul(gate, up)
}

// DotVecs is the grouped-query form of DotVec: out[i] gets the dot of k
// with the i-th of len(out) query vectors packed contiguously in qs, the
// shared k streamed once for up to four of them per pass — the score
// kernel of grouped-query attention, where several query heads share one
// cached key row. Every result is bit-identical to the matching DotVec.
func DotVecs(qs, k []Float, out []Float) {
	if len(qs) != len(out)*len(k) {
		panic("tensai: DotVecs length mismatch")
	}
	kernels.DotVecs(qs, k, out)
}

// Axpys is the grouped form of Axpy: the i-th of len(ws) rows packed
// contiguously in outs accumulates ws[i]*v, the shared v streamed once
// for up to four rows per pass — grouped-query attention's weighted
// value accumulation. Bit-identical to per-row Axpy.
func Axpys(ws []Float, v, outs []Float) {
	if len(outs) != len(ws)*len(v) {
		panic("tensai: Axpys length mismatch")
	}
	kernels.Axpys(ws, v, outs)
}

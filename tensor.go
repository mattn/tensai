package tensai

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// Float is the element type of every tensor. float32 halves memory traffic
// versus float64 and enables the 8-lane AVX2 kernel (see dot_simd.go); its
// ~7 decimal digits are plenty for neural-network training.
type Float = float32

// float32 wrappers for the float64-only math package.
func expF(x Float) Float    { return Float(math.Exp(float64(x))) }
func logF(x Float) Float    { return Float(math.Log(float64(x))) }
func tanhF(x Float) Float   { return Float(math.Tanh(float64(x))) }
func sqrtF(x Float) Float   { return Float(math.Sqrt(float64(x))) }
func powF(x, y Float) Float { return Float(math.Pow(float64(x), float64(y))) }

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

// T returns the transpose of the matrix.
func (m *Matrix) T() *Matrix {
	out := NewMatrix(m.Cols, m.Rows)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			out.Data[c*m.Rows+r] = m.Data[r*m.Cols+c]
		}
	}
	return out
}

// ensureMatrix returns m when it already has the wanted shape, otherwise a
// freshly allocated matrix. The contents are unspecified; callers must
// overwrite (or clear) every element. Layers use this to reuse forward and
// backward scratch buffers between training steps.
func ensureMatrix(m *Matrix, rows, cols int) *Matrix {
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
	clear(out.Data) // dotRows accumulates
	// Rows are independent, so large products are split across CPUs. Small
	// ones stay single-threaded to avoid goroutine overhead.
	workers := 1
	if a.Rows*a.Cols*b.Cols >= 1<<23 {
		workers = runtime.NumCPU()
		if workers > a.Rows {
			workers = a.Rows
		}
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

// TInto writes the transpose of src into dst.
func TInto(dst, src *Matrix) error {
	if dst.Rows != src.Cols || dst.Cols != src.Rows {
		return fmt.Errorf("tensai: transpose shape mismatch: dst %dx%d, src %dx%d",
			dst.Rows, dst.Cols, src.Rows, src.Cols)
	}
	for r := 0; r < src.Rows; r++ {
		for c := 0; c < src.Cols; c++ {
			dst.Data[c*src.Rows+r] = src.Data[r*src.Cols+c]
		}
	}
	return nil
}

// dotRowsGeneric computes rows lo..hi of out = a * b in pure Go. Loop order
// r,k,c keeps the inner loop sequential over b and out, which is far more
// cache-friendly than the naive r,c,k order. dotRows (see dot_simd.go and
// dot_generic.go) dispatches to a SIMD kernel when one is available.
func dotRowsGeneric(out, a, b *Matrix, lo, hi int) {
	for r := lo; r < hi; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		outRow := out.Data[r*b.Cols : (r+1)*b.Cols]
		for k, av := range aRow {
			if av == 0 {
				continue
			}
			bRow := b.Data[k*b.Cols : (k+1)*b.Cols]
			for c, bv := range bRow {
				outRow[c] += av * bv
			}
		}
	}
}

// Add returns a + b (element-wise). Shapes must match.
func Add(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("tensai: add shape mismatch: %dx%d vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	out := NewMatrix(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out, nil
}

// AddBias adds a 1xCols bias vector to every row of a.
func AddBias(a *Matrix, bias []Float) (*Matrix, error) {
	if a.Cols != len(bias) {
		return nil, fmt.Errorf("tensai: addbias mismatch: cols=%d bias=%d", a.Cols, len(bias))
	}
	out := NewMatrix(a.Rows, a.Cols)
	for r := 0; r < a.Rows; r++ {
		for c := 0; c < a.Cols; c++ {
			out.Data[r*a.Cols+c] = a.Data[r*a.Cols+c] + bias[c]
		}
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
	scale := sqrtF(2.0 / Float(rows+cols))
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

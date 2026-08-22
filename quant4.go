package tensai

import (
	"fmt"
	"sync"
)

// q4Group is the number of input rows sharing one scale. Smaller groups
// track the weight distribution closer but pay more per-group folding
// traffic in the kernel; 64 keeps Qwen-class models accurate while
// halving the fold cost of the conventional 32.
const q4Group = 64

// Q4Matrix is a weight matrix quantized to 4 bits with one scale per
// (row group, output column) — the group-wise weight-only scheme that
// keeps int4 usable for real models. Nibbles pack column-split: the byte
// at (row, j) holds column j in its low nibble and column j+Cols/2 in its
// high one, so a vector load unpacks into two runs of consecutive
// columns. Values are stored offset-binary (0..15 encodes -8..7).
type Q4Matrix struct {
	Rows, Cols int
	Q          []byte // Rows x Cols/2, padded so 16-byte loads stay in bounds
	Scale      []Float
}

// QuantizeMatrix4 quantizes group-wise, symmetric with round-to-nearest.
// The column count must be even.
func QuantizeMatrix4(m *Matrix) (*Q4Matrix, error) {
	if m.Cols%2 != 0 {
		return nil, fmt.Errorf("tensai: quantize4 needs an even column count, got %d", m.Cols)
	}
	half := m.Cols / 2
	groups := (m.Rows + q4Group - 1) / q4Group
	q := &Q4Matrix{
		Rows:  m.Rows,
		Cols:  m.Cols,
		Q:     make([]byte, m.Rows*half+16),
		Scale: make([]Float, groups*m.Cols),
	}
	nib := make([]uint8, m.Rows)
	for j := 0; j < m.Cols; j++ {
		for g := 0; g < groups; g++ {
			rlo := g * q4Group
			rhi := min(rlo+q4Group, m.Rows)
			var maxAbs Float
			for i := rlo; i < rhi; i++ {
				v := m.Data[i*m.Cols+j]
				if v < 0 {
					v = -v
				}
				if v > maxAbs {
					maxAbs = v
				}
			}
			s := maxAbs / 7
			q.Scale[g*m.Cols+j] = s
			if s == 0 {
				for i := rlo; i < rhi; i++ {
					nib[i] = 8
				}
				continue
			}
			inv := 1 / s
			for i := rlo; i < rhi; i++ {
				v := m.Data[i*m.Cols+j] * inv
				if v >= 0 {
					v += 0.5
				} else {
					v -= 0.5
				}
				n := int(v)
				if n < -8 {
					n = -8
				} else if n > 7 {
					n = 7
				}
				nib[i] = uint8(n + 8)
			}
		}
		for i := 0; i < m.Rows; i++ {
			if j < half {
				q.Q[i*half+j] |= nib[i]
			} else {
				q.Q[i*half+j-half] |= nib[i] << 4
			}
		}
	}
	return q, nil
}

// MatVec computes out = x @ Q for a single activation row: len(x) must be
// Rows and len(out) Cols. Column pairs are split across CPUs.
func (q *Q4Matrix) MatVec(x, out []Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: q4matvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	half := q.Cols / 2
	workers := dotWorkerCount(half, q.Rows, 1)
	run := func(lo, hi int) {
		q4matvecCols(out, x, q.Q, q.Scale, q.Cols, lo, hi, make([]Float, q.Cols))
	}
	if workers == 1 {
		run(0, half)
		return nil
	}
	chunk := ((half+workers-1)/workers + 7) &^ 7
	var wg sync.WaitGroup
	for lo := 0; lo < half; lo += chunk {
		hi := min(lo+chunk, half)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			run(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// q4matvecColsGeneric accumulates the column pairs [lo,hi) and
// [half+lo, half+hi) of out = x @ Q in pure Go: per 32-row group the
// unscaled products gather in tmp, then fold into out under that group's
// scales. q4matvecCols (see quant4_simd.go and quant4_generic.go)
// dispatches to the AVX2 kernel when available.
func q4matvecColsGeneric(out, x []Float, qw []byte, scale []Float, cols, lo, hi int, tmp []Float) {
	half := cols / 2
	clear(out[lo:hi])
	clear(out[half+lo : half+hi])
	groups := (len(x) + q4Group - 1) / q4Group
	for g := 0; g < groups; g++ {
		rlo := g * q4Group
		rhi := min(rlo+q4Group, len(x))
		clear(tmp[lo:hi])
		clear(tmp[half+lo : half+hi])
		for i := rlo; i < rhi; i++ {
			xi := x[i]
			if xi == 0 {
				continue
			}
			row := qw[i*half:]
			for j := lo; j < hi; j++ {
				b := row[j]
				tmp[j] += xi * Float(int(b&0x0F)-8)
				tmp[half+j] += xi * Float(int(b>>4)-8)
			}
		}
		srow := scale[g*cols:]
		for j := lo; j < hi; j++ {
			out[j] += tmp[j] * srow[j]
			out[half+j] += tmp[half+j] * srow[half+j]
		}
	}
}

package tensai

import (
	"fmt"
	"sync"
)

// q4Group is the number of input rows sharing one scale. 64 keeps
// Qwen-class models accurate while halving the per-group folding work of
// the conventional 32.
const q4Group = 64

// Q4Matrix is a weight matrix quantized to 4 bits with one scale per
// (64-row group, output column). The byte at (i/2)*Cols + j holds column j
// of an interleaved row pair — row i (even) in the low nibble, row i+1 in
// the high one, offset-binary (0..15 encodes -8..7), with a zero row
// appended when Rows is odd. That pairing matches the u8 x s8 pairwise
// multiply-add the AVX2 kernel is built on, the same W-A7 scheme as
// QMatrix: activations quantize per call to 7 bits, and since nibbles are
// at most 15 the i16 pair sums sit far inside saturation.
type Q4Matrix struct {
	Rows, Cols int
	Q          []uint8 // row pairs x Cols, padded for 16-byte loads
	Scale      []Float
}

// QuantizeMatrix4 quantizes group-wise, symmetric with round-to-nearest.
// Columns split across CPUs for large matrices, like QuantizeMatrix.
func QuantizeMatrix4(m *Matrix) (*Q4Matrix, error) {
	pairs := (m.Rows + 1) / 2
	groups := (m.Rows + q4Group - 1) / q4Group
	q := &Q4Matrix{
		Rows:  m.Rows,
		Cols:  m.Cols,
		Q:     make([]uint8, pairs*m.Cols+16),
		Scale: make([]Float, groups*m.Cols),
	}
	parallelCols(m.Rows, m.Cols, func(lo, hi int) {
		quantize4Columns(m, q, groups, lo, hi)
	})
	return q, nil
}

func quantize4Columns(m *Matrix, q *Q4Matrix, groups, colLo, colHi int) {
	for j := colLo; j < colHi; j++ {
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
			inv := Float(0)
			if s != 0 {
				inv = 1 / s
			}
			for i := rlo; i < rhi; i++ {
				n := 0
				if inv != 0 {
					v := m.Data[i*m.Cols+j] * inv
					if v >= 0 {
						v += 0.5
					} else {
						v -= 0.5
					}
					n = int(v)
					if n < -8 {
						n = -8
					} else if n > 7 {
						n = 7
					}
				}
				q.Q[(i/2)*m.Cols+j] |= uint8(n+8) << (4 * (i % 2))
			}
		}
		if m.Rows%2 == 1 {
			q.Q[(m.Rows/2)*m.Cols+j] |= 8 << 4 // zero weight for the pad row
		}
	}
}

// MatVec computes out = x @ Q for a single activation row: len(x) must be
// Rows and len(out) Cols. The activation row quantizes once per call, with
// its per-group sums carrying the nibble offset correction.
func (q *Q4Matrix) MatVec(x, out []Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: q4matvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	xu, sx := quantizeActs(x)
	gsum := make([]int32, (q.Rows+q4Group-1)/q4Group)
	for i, u := range xu {
		gsum[min(i/q4Group, len(gsum)-1)] += int32(u) - 64
	}
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		q4matvecCols(out, xu, sx, gsum, q.Q, q.Scale, q.Cols, 0, q.Cols)
		return nil
	}
	chunk := ((q.Cols+workers-1)/workers + 7) &^ 7
	var wg sync.WaitGroup
	for lo := 0; lo < q.Cols; lo += chunk {
		hi := min(lo+chunk, q.Cols)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			q4matvecCols(out, xu, sx, gsum, q.Q, q.Scale, q.Cols, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// MatMul computes out = x @ Q for a batch of activation rows — the
// prompt-prefill shape, mirroring QMatrix.MatMul: rows quantize to the
// same 7-bit form MatVec uses and the kernel processes them in blocks of
// four against one streaming pass over the nibbles.
func (q *Q4Matrix) MatMul(x, out *Matrix) error {
	if x.Cols != q.Rows || out.Rows != x.Rows || out.Cols != q.Cols {
		return fmt.Errorf("tensai: q4matmul shape mismatch: x %dx%d out %dx%d, want %dx%d",
			x.Rows, x.Cols, out.Rows, out.Cols, q.Rows, q.Cols)
	}
	rows := x.Rows
	groups := (q.Rows + q4Group - 1) / q4Group
	xus := make([][]uint8, rows)
	sxs := make([]Float, rows)
	gsums := make([][]int32, rows)
	for r := 0; r < rows; r++ {
		xus[r], sxs[r] = quantizeActs(x.Data[r*x.Cols : (r+1)*x.Cols])
		gsums[r] = make([]int32, groups)
		for i, u := range xus[r] {
			gsums[r][min(i/q4Group, groups-1)] += int32(u) - 64
		}
	}
	run := func(lo, hi int) {
		var r int
		for ; r+4 <= rows; r += 4 {
			q4matmulCols4(out, xus[r:r+4], sxs[r:r+4], gsums[r:r+4], r, q.Q, q.Scale, q.Cols, lo, hi)
		}
		for ; r < rows; r++ {
			q4matvecCols(out.Data[r*q.Cols:(r+1)*q.Cols], xus[r], sxs[r], gsums[r], q.Q, q.Scale, q.Cols, lo, hi)
		}
	}
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		run(0, q.Cols)
		return nil
	}
	chunk := ((q.Cols+workers-1)/workers + 7) &^ 7
	var wg sync.WaitGroup
	for lo := 0; lo < q.Cols; lo += chunk {
		hi := min(lo+chunk, q.Cols)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			run(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// q4matmulCols4Generic is the four-row batched body: each nibble byte read
// once feeds four output rows.
func q4matmulCols4Generic(out *Matrix, xus [][]uint8, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, cols, lo, hi int) {
	for r := 0; r < 4; r++ {
		clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+hi])
	}
	pairs := len(xus[0]) / 2
	var acc [4][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	for g := 0; g < len(gsums[0]); g++ {
		ib := g * q4Group / 2
		ie := min(ib+q4Group/2, pairs)
		for r := range acc {
			clear(acc[r])
		}
		for i2 := ib; i2 < ie; i2++ {
			row := qw[i2*cols:]
			for r := 0; r < 4; r++ {
				x0 := int32(xus[r][2*i2]) - 64
				x1 := int32(xus[r][2*i2+1]) - 64
				a := acc[r]
				for j := lo; j < hi; j++ {
					b := row[j]
					a[j-lo] += int32(b&0x0F)*x0 + int32(b>>4)*x1
				}
			}
		}
		srow := scale[g*cols:]
		for r := 0; r < 4; r++ {
			o := out.Data[(r0+r)*cols:]
			corr := 8 * gsums[r][g]
			for j := lo; j < hi; j++ {
				o[j] += Float(acc[r][j-lo]-corr) * srow[j]
			}
		}
	}
	for r := 0; r < 4; r++ {
		o := out.Data[(r0+r)*cols:]
		for j := lo; j < hi; j++ {
			o[j] *= sxs[r]
		}
	}
}

// q4matvecColsGeneric accumulates out[lo:hi] in pure Go over the same
// 7-bit activations as the AVX2 kernel, so both builds agree exactly.
// q4matvecCols (see quant4_simd.go and quant4_generic.go) dispatches to
// the AVX2 kernel when available.
func q4matvecColsGeneric(out []Float, xu []uint8, sx Float, gsum []int32, qw []uint8, scale []Float, cols, lo, hi int) {
	clear(out[lo:hi])
	pairs := len(xu) / 2
	acc := make([]int32, hi-lo)
	for g := 0; g < len(gsum); g++ {
		ib := g * q4Group / 2
		ie := min(ib+q4Group/2, pairs)
		clear(acc)
		for i2 := ib; i2 < ie; i2++ {
			x0 := int32(xu[2*i2]) - 64
			x1 := int32(xu[2*i2+1]) - 64
			row := qw[i2*cols:]
			for j := lo; j < hi; j++ {
				b := row[j]
				acc[j-lo] += int32(b&0x0F)*x0 + int32(b>>4)*x1
			}
		}
		srow := scale[g*cols:]
		corr := 8 * gsum[g]
		for j := lo; j < hi; j++ {
			out[j] += Float(acc[j-lo]-corr) * srow[j]
		}
	}
	for j := lo; j < hi; j++ {
		out[j] *= sx
	}
}

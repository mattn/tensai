package tensai

import (
	"fmt"
	"math"
	"sync"
)

// q4Group is the number of input rows sharing one scale. 64 keeps
// Qwen-class models accurate while halving the per-group folding work of
// the conventional 32.
const q4Group = 64

// Q4Matrix is a weight matrix quantized to 4 bits with one scale per
// (64-row group, output column). Rows are stored in interleaved quads of
// two bytes per column: byte (i/4)*2*Cols + 2*j + (i%4)/2 holds rows
// i..i+3 of column j as four nibbles (each byte low nibble first),
// offset-binary (0..15 encodes -8..7), zero rows padding the final quad.
// The 256-bit kernel's nibble unpack turns those two bytes into four
// consecutive u8 lanes — exactly QMatrix's quad layout — so the same
// two-instruction multiply-add chain takes a column four rows deep, with
// activations re-centered to signed bytes and the nibble offset folded
// out through per-group activation sums.
type Q4Matrix struct {
	Rows, Cols int
	Q          []uint8 // row quads x 2*Cols, padded for 32-byte loads
	Scale      []Float
	// ScaleMin, when non-nil, switches the per-(group, column)
	// dequantization from the symmetric offset-binary form
	// scale*(nibble-8) to the asymmetric scale*nibble - min — the form
	// GGUF's Q4_K sub-blocks carry — with Scale unused. Each entry packs
	// the pair as bfloat16 (PackScaleMin), halving what the kernels
	// stream per group next to two float32 tables. Nil keeps the
	// symmetric form.
	ScaleMin []uint32
	Group    int // input rows per scale; 0 means the default q4Group (64)
}

// PackScaleMin rounds a min-form group's scale and min to bfloat16 and
// packs them into one ScaleMin entry (scale low, min high).
func PackScaleMin(scale, min Float) uint32 {
	return uint32(bf16(scale)) | uint32(bf16(min))<<16
}

// UnpackScaleMin is the inverse of PackScaleMin.
func UnpackScaleMin(u uint32) (scale, min Float) {
	return math.Float32frombits(u << 16), math.Float32frombits(u & 0xffff0000)
}

// bf16 rounds a float32 to nearest-even bfloat16, returning the top bits.
func bf16(f Float) uint16 {
	b := math.Float32bits(f)
	return uint16((b + 0x7fff + (b >> 16 & 1)) >> 16)
}

// group returns the effective scale-group length.
func (q *Q4Matrix) group() int {
	if q.Group != 0 {
		return q.Group
	}
	return q4Group
}

// QuantizeMatrix4 quantizes group-wise, symmetric with round-to-nearest.
// Columns split across CPUs for large matrices, like QuantizeMatrix.
func QuantizeMatrix4(m *Matrix) (*Q4Matrix, error) {
	quads := (m.Rows + 3) / 4
	groups := (m.Rows + q4Group - 1) / q4Group
	q := &Q4Matrix{
		Rows:  m.Rows,
		Cols:  m.Cols,
		Q:     make([]uint8, quads*2*m.Cols+32),
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
				q.Q[(i/4)*2*m.Cols+2*j+(i%4)/2] |= uint8(n+8) << (4 * (i % 2))
			}
		}
		for i := m.Rows; i < 4*((m.Rows+3)/4); i++ {
			q.Q[(i/4)*2*m.Cols+2*j+(i%4)/2] |= 8 << (4 * (i % 2)) // zero pad rows
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
	xu, sx := quantizeActs(x) // quad-padded, matching the weight layout
	grp := q.group()
	gsum := make([]int32, (q.Rows+grp-1)/grp)
	for i, u := range xu {
		gsum[min(i/grp, len(gsum)-1)] += int32(u) - 64
	}
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		q4matvecCols(out, xu, sx, gsum, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, 0, q.Cols)
		return nil
	}
	chunk := ((q.Cols+workers-1)/workers + 7) &^ 7
	var wg sync.WaitGroup
	for lo := 0; lo < q.Cols; lo += chunk {
		hi := min(lo+chunk, q.Cols)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			q4matvecCols(out, xu, sx, gsum, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
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
	grp := q.group()
	groups := (q.Rows + grp - 1) / grp
	xus := make([][]uint8, rows)
	sxs := make([]Float, rows)
	gsums := make([][]int32, rows)
	for r := 0; r < rows; r++ {
		xus[r], sxs[r] = quantizeActs(x.Data[r*x.Cols : (r+1)*x.Cols])
		gsums[r] = make([]int32, groups)
		for i, u := range xus[r] {
			gsums[r][min(i/grp, groups-1)] += int32(u) - 64
		}
	}
	run := func(lo, hi int) {
		var r int
		for ; r+4 <= rows; r += 4 {
			q4matmulCols4(out, xus[r:r+4], sxs[r:r+4], gsums[r:r+4], r, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
		}
		for ; r < rows; r++ {
			q4matvecCols(out.Data[r*q.Cols:(r+1)*q.Cols], xus[r], sxs[r], gsums[r], q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
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
func q4matmulCols4Generic(out *Matrix, xus [][]uint8, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	for r := 0; r < 4; r++ {
		clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+hi])
	}
	quads := len(xus[0]) / 4
	var acc [4][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	for g := 0; g < len(gsums[0]); g++ {
		ib := g * group / 4
		ie := min(ib+group/4, quads)
		for r := range acc {
			clear(acc[r])
		}
		for i4 := ib; i4 < ie; i4++ {
			row := qw[i4*2*cols:]
			for r := 0; r < 4; r++ {
				x0 := int32(xus[r][4*i4]) - 64
				x1 := int32(xus[r][4*i4+1]) - 64
				x2 := int32(xus[r][4*i4+2]) - 64
				x3 := int32(xus[r][4*i4+3]) - 64
				a := acc[r]
				for j := lo; j < hi; j++ {
					b0, b1 := row[2*j], row[2*j+1]
					a[j-lo] += int32(b0&0x0F)*x0 + int32(b0>>4)*x1 +
						int32(b1&0x0F)*x2 + int32(b1>>4)*x3
				}
			}
		}
		if sm != nil {
			smrow := sm[g*cols:]
			for r := 0; r < 4; r++ {
				o := out.Data[(r0+r)*cols:]
				gs := Float(gsums[r][g])
				for j := lo; j < hi; j++ {
					s, m := UnpackScaleMin(smrow[j])
					o[j] += Float(acc[r][j-lo])*s - gs*m
				}
			}
		} else {
			srow := scale[g*cols:]
			for r := 0; r < 4; r++ {
				o := out.Data[(r0+r)*cols:]
				corr := 8 * gsums[r][g]
				for j := lo; j < hi; j++ {
					o[j] += Float(acc[r][j-lo]-corr) * srow[j]
				}
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
func q4matvecColsGeneric(out []Float, xu []uint8, sx Float, gsum []int32, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	clear(out[lo:hi])
	quads := len(xu) / 4
	acc := make([]int32, hi-lo)
	for g := 0; g < len(gsum); g++ {
		ib := g * group / 4
		ie := min(ib+group/4, quads)
		clear(acc)
		for i4 := ib; i4 < ie; i4++ {
			x0 := int32(xu[4*i4]) - 64
			x1 := int32(xu[4*i4+1]) - 64
			x2 := int32(xu[4*i4+2]) - 64
			x3 := int32(xu[4*i4+3]) - 64
			row := qw[i4*2*cols:]
			for j := lo; j < hi; j++ {
				b0, b1 := row[2*j], row[2*j+1]
				acc[j-lo] += int32(b0&0x0F)*x0 + int32(b0>>4)*x1 +
					int32(b1&0x0F)*x2 + int32(b1>>4)*x3
			}
		}
		if sm != nil {
			smrow := sm[g*cols:]
			gs := Float(gsum[g])
			for j := lo; j < hi; j++ {
				s, m := UnpackScaleMin(smrow[j])
				out[j] += Float(acc[j-lo])*s - gs*m
			}
		} else {
			srow := scale[g*cols:]
			corr := 8 * gsum[g]
			for j := lo; j < hi; j++ {
				out[j] += Float(acc[j-lo]-corr) * srow[j]
			}
		}
	}
	for j := lo; j < hi; j++ {
		out[j] *= sx
	}
}

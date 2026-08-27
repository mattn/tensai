package quant

import (
	"fmt"
	"math"

	"github.com/mattn/tensai"
)

// q4Group is the number of input rows sharing one scale. 64 keeps
// Qwen-class models accurate while halving the per-group folding work of
// the conventional 32.
const q4Group = 64

// Q4Matrix is a weight matrix quantized to 4 bits with one scale per
// (64-row group, output column). Rows are stored in interleaved quads of
// two bytes per column, tiled 32 columns at a time: tile j/32 packs its
// row quads back to back (64 bytes apiece, see Index), so a kernel
// worker sweeping a tile range streams strictly sequential memory
// instead of striding across the full row width. Each byte holds two
// rows' nibbles (low nibble first), offset-binary (0..15 encodes -8..7),
// zero rows padding the final quad. The 256-bit kernel's nibble unpack
// turns those two bytes into four consecutive u8 lanes — exactly
// QMatrix's quad layout — so the same two-instruction multiply-add chain
// takes a column four rows deep, with activations re-centered to signed
// bytes and the nibble offset folded out through per-group activation
// sums.
type Q4Matrix struct {
	Rows, Cols int
	Q          []uint8 // row quads x 2*Cols, padded for 32-byte loads
	Scale      []tensai.Float
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
func PackScaleMin(scale, min tensai.Float) uint32 {
	return uint32(bf16(scale)) | uint32(bf16(min))<<16
}

// UnpackScaleMin is the inverse of PackScaleMin.
func UnpackScaleMin(u uint32) (scale, min tensai.Float) {
	return math.Float32frombits(u << 16), math.Float32frombits(u & 0xffff0000)
}

// bf16 rounds a float32 to nearest-even bfloat16, returning the top bits.
func bf16(f tensai.Float) uint16 {
	b := math.Float32bits(f)
	return uint16((b + 0x7fff + (b >> 16 & 1)) >> 16)
}

// q4Tile is the column-tile width of the nibble layout.
const q4Tile = 32

// Index returns the position in Q of the byte carrying column j of rows
// i and i+1 (the low and the high nibble; shift by 4*(i%2)).
func (q *Q4Matrix) Index(i, j int) int {
	quads := (q.Rows + 3) / 4
	return (j/q4Tile)*quads*2*q4Tile + (i/4)*2*q4Tile + (j%q4Tile)*2 + (i%4)/2
}

// TableIndex returns the position in Scale or ScaleMin of group g,
// column j. Tables are tile-major like the nibbles — tile, then group,
// then the 32 columns — so a kernel worker's table walk is sequential.
func (q *Q4Matrix) TableIndex(g, j int) int {
	groups := (q.Rows + q.group() - 1) / q.group()
	return ((j/q4Tile)*groups+g)*q4Tile + j%q4Tile
}

// NewQ4Matrix allocates the layout for rows x cols with `group` input
// rows per scale (0 for the default 64); minForm picks the packed
// asymmetric scale/min table over the symmetric Scale. The caller fills
// Q via Index and the table it asked for.
func NewQ4Matrix(rows, cols, group int, minForm bool) *Q4Matrix {
	if group == 0 {
		group = q4Group
	}
	quads := (rows + 3) / 4
	groups := (rows + group - 1) / group
	tiles := (cols + q4Tile - 1) / q4Tile
	q := &Q4Matrix{
		Rows:  rows,
		Cols:  cols,
		Q:     make([]uint8, tiles*quads*2*q4Tile+32),
		Group: group,
	}
	if minForm {
		q.ScaleMin = make([]uint32, tiles*groups*q4Tile)
	} else {
		q.Scale = make([]tensai.Float, tiles*groups*q4Tile)
	}
	return q
}

// group returns the effective scale-group length.
func (q *Q4Matrix) group() int {
	if q.Group != 0 {
		return q.Group
	}
	return q4Group
}

// Quantize4 quantizes group-wise, symmetric with round-to-nearest.
// Columns split across CPUs for large matrices, like Quantize.
func Quantize4(m *tensai.Matrix) (*Q4Matrix, error) {
	groups := (m.Rows + q4Group - 1) / q4Group
	q := NewQ4Matrix(m.Rows, m.Cols, 0, false)
	parallelCols(m.Rows, m.Cols, func(lo, hi int) {
		quantize4Columns(m, q, groups, lo, hi)
	})
	return q, nil
}

func quantize4Columns(m *tensai.Matrix, q *Q4Matrix, groups, colLo, colHi int) {
	for j := colLo; j < colHi; j++ {
		for g := 0; g < groups; g++ {
			rlo := g * q4Group
			rhi := min(rlo+q4Group, m.Rows)
			var maxAbs tensai.Float
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
			q.Scale[q.TableIndex(g, j)] = s
			inv := tensai.Float(0)
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
				q.Q[q.Index(i, j)] |= uint8(n+8) << (4 * (i % 2))
			}
		}
		for i := m.Rows; i < 4*((m.Rows+3)/4); i++ {
			q.Q[q.Index(i, j)] |= 8 << (4 * (i % 2)) // zero pad rows
		}
	}
}

// packQuads re-centers the 7-bit activations to signed bytes and packs
// each quad into the u32 the kernels broadcast, so the scalar assembly
// runs once per call instead of once per column tile.
func packQuads(xu []uint8) []uint32 {
	xq := make([]uint32, len(xu)/4)
	for i := range xq {
		x0 := uint32(uint8(int8(int(xu[4*i]) - 64)))
		x1 := uint32(uint8(int8(int(xu[4*i+1]) - 64)))
		x2 := uint32(uint8(int8(int(xu[4*i+2]) - 64)))
		x3 := uint32(uint8(int8(int(xu[4*i+3]) - 64)))
		xq[i] = x0 | x1<<8 | x2<<16 | x3<<24
	}
	return xq
}

// MatVec computes out = x @ Q for a single activation row: len(x) must be
// Rows and len(out) Cols. The activation row quantizes once per call, with
// its per-group sums carrying the nibble offset correction.
func (q *Q4Matrix) MatVec(x, out []tensai.Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: q4matvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	xu, sx := quantizeActs(x) // quad-padded, matching the weight layout
	xq := packQuads(xu)
	grp := q.group()
	gsum := make([]int32, (q.Rows+grp-1)/grp)
	for i, u := range xu {
		gsum[min(i/grp, len(gsum)-1)] += int32(u) - 64
	}
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		q4matvecCols(out, xu, xq, sx, gsum, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, 0, q.Cols)
		return nil
	}
	// Tile-aligned chunks keep every worker's vector span inside whole
	// layout tiles, and its memory walk sequential.
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		q4matvecCols(out, xu, xq, sx, gsum, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
	})
	return nil
}

// MatMul computes out = x @ Q for a batch of activation rows — the
// prompt-prefill shape, mirroring QMatrix.MatMul: rows quantize to the
// same 7-bit form MatVec uses and the kernel processes them in blocks of
// four against one streaming pass over the nibbles.
func (q *Q4Matrix) MatMul(x, out *tensai.Matrix) error {
	if x.Cols != q.Rows || out.Rows != x.Rows || out.Cols != q.Cols {
		return fmt.Errorf("tensai: q4matmul shape mismatch: x %dx%d out %dx%d, want %dx%d",
			x.Rows, x.Cols, out.Rows, out.Cols, q.Rows, q.Cols)
	}
	rows := x.Rows
	grp := q.group()
	groups := (q.Rows + grp - 1) / grp
	xus := make([][]uint8, rows)
	xqs := make([][]uint32, rows)
	sxs := make([]tensai.Float, rows)
	gsums := make([][]int32, rows)
	for r := 0; r < rows; r++ {
		xus[r], sxs[r] = quantizeActs(x.Data[r*x.Cols : (r+1)*x.Cols])
		xqs[r] = packQuads(xus[r])
		gsums[r] = make([]int32, groups)
		for i, u := range xus[r] {
			gsums[r][min(i/grp, groups-1)] += int32(u) - 64
		}
	}
	// See padRows8: the row tail pads to a full block over a shared
	// scratch matrix so the weights stream once. A tail of exactly four
	// takes the four-row kernel instead of padding to eight — the madd
	// work scales with padded rows too, and speculative decoding's
	// four-row verification block must not pay double.
	var pxus [][]uint8
	var pxqs [][]uint32
	var psxs []tensai.Float
	var pgs [][]int32
	var scratch *tensai.Matrix
	rem := rows % 8
	if rem != 0 && rem != 4 {
		t := rows - rem
		pxus, psxs, scratch = padRows8(xus[t:], sxs[t:], len(xus[0]), q.Cols)
		pxqs = make([][]uint32, 8)
		for i := range pxqs {
			pxqs[i] = packQuads(pxus[i])
		}
		pgs = make([][]int32, 8)
		copy(pgs, gsums[t:])
		for i := rem; i < 8; i++ {
			pgs[i] = make([]int32, groups)
		}
	}
	run := func(lo, hi int) {
		var r int
		for ; r+8 <= rows; r += 8 {
			q4matmulCols8(out, xus[r:r+8], xqs[r:r+8], sxs[r:r+8], gsums[r:r+8], r, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
		}
		switch {
		case rem == 0:
		case rem == 4:
			q4matmulCols4(out, xus[r:r+4], xqs[r:r+4], sxs[r:r+4], gsums[r:r+4], r, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
		default:
			if rem < 4 {
				q4matmulCols4(scratch, pxus[:4], pxqs[:4], psxs[:4], pgs[:4], 0, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
			} else {
				q4matmulCols8(scratch, pxus, pxqs, psxs, pgs, 0, q.Q, q.Scale, q.ScaleMin, grp, q.Cols, lo, hi)
			}
			for i := 0; i < rem; i++ {
				copy(out.Data[(r+i)*q.Cols+lo:(r+i)*q.Cols+hi], scratch.Data[i*q.Cols+lo:i*q.Cols+hi])
			}
		}
	}
	workers := matvecWorkerCount(q.Cols, q.Rows*rows)
	if workers == 1 {
		run(0, q.Cols)
		return nil
	}
	// Tile-aligned, so the row-tail matvec's vector span starts on a
	// layout tile like the batch kernel's 8-column steps do.
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		run(lo, hi)
	})
	return nil
}

// q4matmulCols4Generic is the four-row batched body: each nibble byte read
// once feeds four output rows.
func q4matmulCols4Generic(out *tensai.Matrix, xus [][]uint8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	for r := 0; r < 4; r++ {
		clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+hi])
	}
	quads := len(xus[0]) / 4
	groups := len(gsums[0])
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
			row := qw[i4*2*q4Tile:]
			for r := 0; r < 4; r++ {
				x0 := int32(xus[r][4*i4]) - 64
				x1 := int32(xus[r][4*i4+1]) - 64
				x2 := int32(xus[r][4*i4+2]) - 64
				x3 := int32(xus[r][4*i4+3]) - 64
				a := acc[r]
				for j := lo; j < hi; j++ {
					o := (j/q4Tile)*quads*2*q4Tile + (j%q4Tile)*2
					b0, b1 := row[o], row[o+1]
					a[j-lo] += int32(b0&0x0F)*x0 + int32(b0>>4)*x1 +
						int32(b1&0x0F)*x2 + int32(b1>>4)*x3
				}
			}
		}
		if sm != nil {
			for r := 0; r < 4; r++ {
				o := out.Data[(r0+r)*cols:]
				gs := tensai.Float(gsums[r][g])
				for j := lo; j < hi; j++ {
					s, m := UnpackScaleMin(sm[((j/q4Tile)*groups+g)*q4Tile+j%q4Tile])
					o[j] += tensai.Float(acc[r][j-lo])*s - gs*m
				}
			}
		} else {
			for r := 0; r < 4; r++ {
				o := out.Data[(r0+r)*cols:]
				corr := 8 * gsums[r][g]
				for j := lo; j < hi; j++ {
					o[j] += tensai.Float(acc[r][j-lo]-corr) * scale[((j/q4Tile)*groups+g)*q4Tile+j%q4Tile]
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
func q4matvecColsGeneric(out []tensai.Float, xu []uint8, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	clear(out[lo:hi])
	quads := len(xu) / 4
	groups := len(gsum)
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
			for tile := lo / q4Tile; tile <= (hi-1)/q4Tile; tile++ {
				j0 := max(lo, tile*q4Tile)
				j1 := min(hi, j0+(q4Tile-j0%q4Tile))
				row := qw[tile*quads*2*q4Tile+i4*2*q4Tile+(j0%q4Tile)*2:]
				for j := j0; j < j1; j++ {
					b0, b1 := row[0], row[1]
					acc[j-lo] += int32(b0&0x0F)*x0 + int32(b0>>4)*x1 +
						int32(b1&0x0F)*x2 + int32(b1>>4)*x3
					row = row[2:]
				}
			}
		}
		if sm != nil {
			gs := tensai.Float(gsum[g])
			for j := lo; j < hi; j++ {
				s, m := UnpackScaleMin(sm[((j/q4Tile)*groups+g)*q4Tile+j%q4Tile])
				out[j] += tensai.Float(acc[j-lo])*s - gs*m
			}
		} else {
			corr := 8 * gsum[g]
			for j := lo; j < hi; j++ {
				out[j] += tensai.Float(acc[j-lo]-corr) * scale[((j/q4Tile)*groups+g)*q4Tile+j%q4Tile]
			}
		}
	}
	for j := lo; j < hi; j++ {
		out[j] *= sx
	}
}

package quant

import (
	"fmt"
	"math"

	"github.com/mattn/tensai"
)

// MXFP4Matrix is a weight matrix in microscaling FP4: 32-row groups per
// column share one power-of-two E8M0 factor, and each weight is a 4-bit
// FP4 code (E2M1: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 before scaling) —
// the format gpt-oss ships its expert weights in. Codes store in
// Q4Matrix's tiled quad-nibble layout; the kernels expand each code to
// twice its FP4 value on the integer grid {0, ±1..±4, ±6, ±8, ±12} with
// a 16-entry table lookup and run the grouped-int8 multiply-add chain,
// Scale carrying half the E8M0 factor so the products stay exact.
type MXFP4Matrix struct {
	Rows, Cols int
	Q          []uint8        // FP4 codes, tiled quad nibbles (Index)
	Scale      []tensai.Float // per (32-row group, column), tile-major (TableIndex)
	ColSum64   []int32        // 64 * sum of a group's expanded codes, tile-major
}

// mxfp4LUT maps an FP4 code to twice its value; the high code bit is
// the sign, and both zero encodings map to zero.
var mxfp4LUT = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// MXFP4Scale converts an E8M0 exponent byte to the matrix's Scale entry:
// half of 2^(e-127), matching the doubled integer grid.
func MXFP4Scale(e uint8) tensai.Float {
	return tensai.Float(math.Ldexp(1, int(e)-128))
}

// MXFP4Value returns an FP4 code's expanded integer value (twice the FP4
// value); callers building ColSum64 sum these.
func MXFP4Value(code uint8) int8 {
	return mxfp4LUT[code&0x0F]
}

// NewMXFP4Matrix allocates the layout for rows x cols; the caller fills
// Q via Index and the tile-major tables via TableIndex.
func NewMXFP4Matrix(rows, cols int) *MXFP4Matrix {
	quads := (rows + 3) / 4
	groups := (rows + 31) / 32
	tiles := (cols + q4Tile - 1) / q4Tile
	return &MXFP4Matrix{
		Rows:     rows,
		Cols:     cols,
		Q:        make([]uint8, tiles*quads*2*q4Tile+32),
		Scale:    make([]tensai.Float, tiles*groups*q4Tile),
		ColSum64: make([]int32, tiles*groups*q4Tile),
	}
}

// Index returns the position in Q of the byte carrying column j of rows
// i and i+1 (low and high nibble; shift by 4*(i%2)).
func (q *MXFP4Matrix) Index(i, j int) int {
	quads := (q.Rows + 3) / 4
	return (j/q4Tile)*quads*2*q4Tile + (i/4)*2*q4Tile + (j%q4Tile)*2 + (i%4)/2
}

// TableIndex returns the position in Scale and ColSum64 of group g,
// column j.
func (q *MXFP4Matrix) TableIndex(g, j int) int {
	groups := (q.Rows + 31) / 32
	return ((j/q4Tile)*groups+g)*q4Tile + j%q4Tile
}

// MatVec computes out = x @ Q for a single activation row: len(x) must
// be Rows and len(out) Cols.
func (q *MXFP4Matrix) MatVec(x, out []tensai.Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: mxfp4 matvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	xu, sx := quantizeActs(x)
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		mxfp4MatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, q.Cols, 0, q.Cols)
		return nil
	}
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		mxfp4MatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
	})
	return nil
}

// MatMul computes out = x @ Q for a batch of activation rows.
func (q *MXFP4Matrix) MatMul(x, out *tensai.Matrix) error {
	if x.Cols != q.Rows || out.Rows != x.Rows || out.Cols != q.Cols {
		return fmt.Errorf("tensai: mxfp4 matmul shape mismatch: x %dx%d out %dx%d, want %dx%d",
			x.Rows, x.Cols, out.Rows, out.Cols, q.Rows, q.Cols)
	}
	rows := x.Rows
	xus := make([][]uint8, rows)
	sxs := make([]tensai.Float, rows)
	for r := 0; r < rows; r++ {
		xus[r], sxs[r] = quantizeActs(x.Data[r*x.Cols : (r+1)*x.Cols])
	}
	// The row tail (rows%8 leftovers) pads to one full block: zero
	// rows are harmless in the integer kernels, the scratch matrix is
	// shared by every column chunk (they write disjoint ranges), and
	// only the real rows copy back — so the tail streams the weights
	// once instead of once per row.
	var pxus [][]uint8
	var psxs []tensai.Float
	var scratch *tensai.Matrix
	if rows%8 != 0 {
		pxus, psxs, scratch = padRows8(xus[rows-rows%8:], sxs[rows-rows%8:], len(xus[0]), q.Cols)
	}
	run := func(lo, hi int) {
		var r int
		for ; r+8 <= rows; r += 8 {
			mxfp4MatmulRows8(out, xus[r:r+8], sxs[r:r+8], r, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
		}
		if r < rows {
			mxfp4MatmulRows8(scratch, pxus, psxs, 0, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
			for i := 0; i < rows-r; i++ {
				copy(out.Data[(r+i)*q.Cols+lo:(r+i)*q.Cols+hi], scratch.Data[i*q.Cols+lo:i*q.Cols+hi])
			}
		}
	}
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		run(0, q.Cols)
		return nil
	}
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		run(lo, hi)
	})
	return nil
}

// mxfp4MatvecColsGeneric accumulates out[lo:hi] in pure Go over the same
// 7-bit activations as the AVX2 kernel, so both builds agree exactly.
func mxfp4MatvecColsGeneric(out []tensai.Float, xu []uint8, sx tensai.Float, qw []uint8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	clear(out[lo:hi])
	quads := len(xu) / 4
	groups := (len(xu) + 31) / 32
	acc := make([]int32, hi-lo)
	for g := 0; g < groups; g++ {
		ib := g * 8
		ie := min(ib+8, quads)
		clear(acc)
		for i4 := ib; i4 < ie; i4++ {
			x0, x1 := int32(xu[4*i4]), int32(xu[4*i4+1])
			x2, x3 := int32(xu[4*i4+2]), int32(xu[4*i4+3])
			row := qw[i4*2*q4Tile:]
			for j := lo; j < hi; j++ {
				o := (j/q4Tile)*quads*2*q4Tile + (j%q4Tile)*2
				b0, b1 := row[o], row[o+1]
				acc[j-lo] += x0*int32(mxfp4LUT[b0&0x0F]) + x1*int32(mxfp4LUT[b0>>4]) +
					x2*int32(mxfp4LUT[b1&0x0F]) + x3*int32(mxfp4LUT[b1>>4])
			}
		}
		for j := lo; j < hi; j++ {
			t := ((j/q4Tile)*groups + g) * q4Tile
			out[j] += tensai.Float(acc[j-lo]-colSum64[t+j%q4Tile]) * scale[t+j%q4Tile]
		}
	}
	for j := lo; j < hi; j++ {
		out[j] *= sx
	}
}

// mxfp4MatmulRows8Generic is the eight-row batched body.
func mxfp4MatmulRows8Generic(out *tensai.Matrix, xus [][]uint8, sxs []tensai.Float, r0 int, qw []uint8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	var acc [8][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + 31) / 32
	for r := 0; r < 8; r++ {
		clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+hi])
	}
	for g := 0; g < groups; g++ {
		ib := g * 8
		ie := min(ib+8, quads)
		for r := range acc {
			clear(acc[r])
		}
		for i4 := ib; i4 < ie; i4++ {
			row := qw[i4*2*q4Tile:]
			for r := 0; r < 8; r++ {
				x0, x1 := int32(xus[r][4*i4]), int32(xus[r][4*i4+1])
				x2, x3 := int32(xus[r][4*i4+2]), int32(xus[r][4*i4+3])
				a := acc[r]
				for j := lo; j < hi; j++ {
					o := (j/q4Tile)*quads*2*q4Tile + (j%q4Tile)*2
					b0, b1 := row[o], row[o+1]
					a[j-lo] += x0*int32(mxfp4LUT[b0&0x0F]) + x1*int32(mxfp4LUT[b0>>4]) +
						x2*int32(mxfp4LUT[b1&0x0F]) + x3*int32(mxfp4LUT[b1>>4])
				}
			}
		}
		for r := 0; r < 8; r++ {
			o := out.Data[(r0+r)*cols:]
			for j := lo; j < hi; j++ {
				t := ((j/q4Tile)*groups + g) * q4Tile
				o[j] += tensai.Float(acc[r][j-lo]-colSum64[t+j%q4Tile]) * scale[t+j%q4Tile]
			}
		}
	}
	for r := 0; r < 8; r++ {
		o := out.Data[(r0+r)*cols:]
		for j := lo; j < hi; j++ {
			o[j] *= sxs[r]
		}
	}
}

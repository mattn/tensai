package quant

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/workpool"
)

// TernaryMatrix is a weight matrix whose every weight is -1, 0 or +1,
// with one scale per (128-row group, output column): the form a ternary
// checkpoint such as PrismML's Bonsai ships, at two bits a weight here
// against the 1.75 of its dense packing, which no kernel can consume
// without unpacking.
//
// The codes are w+1, so 0, 1 and 2, four to a byte in bits 2s..2s+1 for
// slot s. A byte carries four rows of one column, but not consecutive
// rows: within a block of 16 rows and 8 columns (32 bytes), the byte at
// lane 4c+t carries rows 4s+t of column c in slot s. Shifting the whole
// block right by 2s and masking the low two bits therefore lands rows
// 4s..4s+3 of the 8 columns in the quad layout the grouped-int8 chain
// multiplies (four consecutive lanes, one column, four rows), which is
// what makes the unpack two instructions. Blocks tile 32 columns at a
// time, the four column-octets of a 16-row block back to back (128
// bytes), so a kernel worker sweeping a tile streams sequential memory.
//
// The weights are the unsigned operand of the multiply-add and the
// activations the signed one, the opposite of the other layouts: a code
// times an activation sums to the dot product plus the activations'
// sum, which every column shares, so there is no per-column correction
// table to stream. A pad row's code is 1, weight zero.
type TernaryMatrix struct {
	Rows, Cols int
	Q          []uint8        // codes, in the block layout above
	Scale      []tensai.Float // per (128-row group, column), tile-major
}

// tGroup is the number of input rows sharing one scale.
const tGroup = 128

// tTile is the column-tile width; tBlock the bytes of one 16-row block
// of a tile.
const (
	tTile  = 32
	tBlock = 4 * tTile // 4 octets of 32 bytes
)

// NewTernaryMatrix allocates the layout for rows x cols, every weight
// zero; the caller fills Q via Set and Scale via TableIndex.
func NewTernaryMatrix(rows, cols int) *TernaryMatrix {
	blocks := (rows + 15) / 16
	groups := (rows + tGroup - 1) / tGroup
	tiles := (cols + tTile - 1) / tTile
	q := &TernaryMatrix{
		Rows:  rows,
		Cols:  cols,
		Q:     make([]uint8, tiles*blocks*tBlock+32),
		Scale: make([]tensai.Float, tiles*groups*tTile),
	}
	for i := range q.Q {
		q.Q[i] = 0x55 // slot code 1 in all four: weight zero
	}
	return q
}

// Index returns the byte in Q carrying row i of column j, and the shift
// of its two-bit slot.
func (q *TernaryMatrix) Index(i, j int) (idx int, shift uint) {
	blocks := (q.Rows + 15) / 16
	idx = (j/tTile)*blocks*tBlock + (i/16)*tBlock + ((j%tTile)/8)*32 + (j%8)*4 + i%4
	return idx, 2 * uint((i%16)/4)
}

// Set stores weight w (-1, 0 or +1) at row i, column j.
func (q *TernaryMatrix) Set(i, j int, w int8) {
	idx, shift := q.Index(i, j)
	q.Q[idx] = q.Q[idx]&^(3<<shift) | uint8(w+1)<<shift
}

// SetGroup stores one scale group of column j, the 128 weights from
// row 128*g down, in one pass over its eight blocks.
func (q *TernaryMatrix) SetGroup(g, j int, w *[128]int8) {
	idx, _ := q.Index(g*tGroup, j)
	for k := 0; k < tGroup/16; k++ {
		b := q.Q[idx+k*tBlock : idx+k*tBlock+4 : idx+k*tBlock+4]
		r := w[16*k : 16*k+16]
		for t := 0; t < 4; t++ {
			b[t] = uint8(r[t]+1) | uint8(r[4+t]+1)<<2 | uint8(r[8+t]+1)<<4 | uint8(r[12+t]+1)<<6
		}
	}
}

// At reads the weight at row i, column j.
func (q *TernaryMatrix) At(i, j int) int8 {
	idx, shift := q.Index(i, j)
	return int8((q.Q[idx]>>shift)&3) - 1
}

// TableIndex returns the position in Scale of group g, column j.
func (q *TernaryMatrix) TableIndex(g, j int) int {
	groups := (q.Rows + tGroup - 1) / tGroup
	return ((j/tTile)*groups+g)*tTile + j%tTile
}

// signedActs quantizes an activation row to signed 7-bit (round(x/sx)
// in [-63,63]), padded to a whole 16-row block, with the sum of each
// 128-row group, which is the correction every column subtracts.
func signedActs(x []tensai.Float, rows int) (xs []int8, sx tensai.Float, gsum []int32) {
	xu, sx := quantizeActs(x)
	padded := (rows + 15) &^ 15
	xs = make([]int8, padded)
	gsum = make([]int32, (rows+tGroup-1)/tGroup)
	for i := range x {
		v := int8(int(xu[i]) - 64)
		xs[i] = v
		gsum[i/tGroup] += int32(v)
	}
	return xs, sx, gsum
}

// MatVec computes out = x @ Q for a single activation row: len(x) must
// be Rows and len(out) Cols.
func (q *TernaryMatrix) MatVec(x, out []tensai.Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: ternary matvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	xs, sx, gsum := signedActs(x, q.Rows)
	xq := xsQuads(xs)
	if matvecWorkerCount(q.Cols, q.Rows) == 1 {
		ternaryMatvecCols(out, xs, xq, sx, gsum, q.Q, q.Scale, q.Rows, q.Cols, 0, q.Cols)
		return nil
	}
	workpool.Run(q.Cols, tTile, func(lo, hi int) {
		ternaryMatvecCols(out, xs, xq, sx, gsum, q.Q, q.Scale, q.Rows, q.Cols, lo, hi)
	})
	return nil
}

// MatMul computes out = x @ Q for a batch of activation rows.
func (q *TernaryMatrix) MatMul(x, out *tensai.Matrix) error {
	if x.Cols != q.Rows || out.Rows != x.Rows || out.Cols != q.Cols {
		return fmt.Errorf("tensai: ternary matmul shape mismatch: x %dx%d out %dx%d, want %dx%d",
			x.Rows, x.Cols, out.Rows, out.Cols, q.Rows, q.Cols)
	}
	rows := x.Rows
	xss := make([][]int8, rows)
	sxs := make([]tensai.Float, rows)
	gsums := make([][]int32, rows)
	for r := 0; r < rows; r++ {
		xss[r], sxs[r], gsums[r] = signedActs(x.Data[r*x.Cols:(r+1)*x.Cols], q.Rows)
	}
	// The row tail pads to one full block of eight, as the other
	// layouts do: zero rows cost nothing but the pass, and only the
	// real rows copy back.
	var pxss [][]int8
	var psxs []tensai.Float
	var pgsums [][]int32
	var scratch *tensai.Matrix
	if rows%8 != 0 {
		pxss, psxs, pgsums = make([][]int8, 8), make([]tensai.Float, 8), make([][]int32, 8)
		zero := make([]int8, len(xss[0]))
		zsum := make([]int32, len(gsums[0]))
		for i := 0; i < 8; i++ {
			if r := rows - rows%8 + i; r < rows {
				pxss[i], psxs[i], pgsums[i] = xss[r], sxs[r], gsums[r]
			} else {
				pxss[i], pgsums[i] = zero, zsum
			}
		}
		scratch = tensai.NewMatrix(8, q.Cols)
	}
	run := func(lo, hi int) {
		var r int
		for ; r+8 <= rows; r += 8 {
			ternaryMatmulRows8(out, xss[r:r+8], sxs[r:r+8], gsums[r:r+8], r, q.Q, q.Scale, q.Rows, q.Cols, lo, hi)
		}
		if r < rows {
			ternaryMatmulRows8(scratch, pxss, psxs, pgsums, 0, q.Q, q.Scale, q.Rows, q.Cols, lo, hi)
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
	workpool.Run(q.Cols, tTile, func(lo, hi int) {
		run(lo, hi)
	})
	return nil
}

// ternaryMatvecColsGeneric accumulates out[lo:hi] in pure Go over the
// same signed activations as the AVX2 kernel, so both builds agree
// exactly.
func ternaryMatvecColsGeneric(out []tensai.Float, xs []int8, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	blocks := (rows + 15) / 16
	groups := (rows + tGroup - 1) / tGroup
	for j := lo; j < hi; j++ {
		tile := qw[(j/tTile)*blocks*tBlock+((j%tTile)/8)*32+(j%8)*4:]
		stab := scale[(j/tTile)*groups*tTile+j%tTile:]
		var sum tensai.Float
		for g := 0; g < groups; g++ {
			var acc int32
			for b := g * 8; b < min(g*8+8, blocks); b++ {
				blk := tile[b*tBlock:]
				for t := 0; t < 4; t++ {
					c := blk[t]
					for s := 0; s < 4; s++ {
						acc += int32((c>>(2*uint(s)))&3) * int32(xs[b*16+4*s+t])
					}
				}
			}
			sum += tensai.Float(acc-gsum[g]) * stab[g*tTile]
		}
		out[j] = sum * sx
	}
}

// ternaryMatmulRows8Generic is the eight-row form of the above.
func ternaryMatmulRows8Generic(out *tensai.Matrix, xss [][]int8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	for r := 0; r < 8; r++ {
		o := out.Data[(r0+r)*cols : (r0+r+1)*cols]
		ternaryMatvecColsGeneric(o, xss[r], sxs[r], gsums[r], qw, scale, rows, cols, lo, hi)
	}
}

// xsQuads packs a signed activation row four bytes to a word, one word
// per row-quad, for the kernels' broadcasts.
func xsQuads(xs []int8) []uint32 {
	xq := make([]uint32, len(xs)/4)
	for i := range xq {
		xq[i] = uint32(uint8(xs[4*i])) | uint32(uint8(xs[4*i+1]))<<8 | uint32(uint8(xs[4*i+2]))<<16 | uint32(uint8(xs[4*i+3]))<<24
	}
	return xq
}

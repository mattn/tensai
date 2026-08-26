package tensai

import (
	"fmt"
)

// q8Group is the number of input rows sharing one scale in a Q8GMatrix —
// GGUF's Q8_0 block length, so its blocks repack without requantizing.
const q8Group = 32

// Q8GMatrix is a weight matrix quantized to int8 with one scale per
// (32-row group, output column) — the granularity llama.cpp's Q8_0 blocks
// carry, so a GGUF checkpoint loads by transposing bytes, never touching
// float32 weights. Rows are stored in the same interleaved quads as
// QMatrix; the kernels are the QMatrix kernels with the group scale and
// activation-offset correction folded in at group boundaries, Q4Matrix
// style.
type Q8GMatrix struct {
	Rows, Cols int
	Q          []int8 // interleaved row quads, padded for 32-byte loads
	Scale      []Float
	ColSum64   []int32 // per (group, column): 64 * sum of the group's weights
	Group      int     // input rows per scale; 0 means the default q8Group (32)
}

// NewQ8GMatrix allocates the layout for rows x cols with `group` input
// rows per scale (0 for the default 32); the caller fills Q (quad
// layout), Scale, and ColSum64 — see the loaders in _example/qwen.
func NewQ8GMatrix(rows, cols, group int) *Q8GMatrix {
	if group == 0 {
		group = q8Group
	}
	quads := (rows + 3) / 4
	groups := (rows + group - 1) / group
	return &Q8GMatrix{
		Rows:     rows,
		Cols:     cols,
		Q:        make([]int8, ((cols+q4Tile-1)/q4Tile)*quads*4*q4Tile+32),
		Scale:    make([]Float, ((cols+q4Tile-1)/q4Tile)*groups*q4Tile),
		ColSum64: make([]int32, ((cols+q4Tile-1)/q4Tile)*groups*q4Tile),
		Group:    group,
	}
}

// group returns the effective scale-group length.
func (q *Q8GMatrix) group() int {
	if q.Group != 0 {
		return q.Group
	}
	return q8Group
}

// Index returns the position in Q of row i, column j: 32-column tiles
// store their row quads back to back (128 bytes apiece), so a kernel
// worker streams sequential memory.
func (q *Q8GMatrix) Index(i, j int) int {
	quads := (q.Rows + 3) / 4
	return (j/q4Tile)*quads*4*q4Tile + (i/4)*4*q4Tile + (j%q4Tile)*4 + i%4
}

// TableIndex returns the position in Scale and ColSum64 of group g,
// column j; tables are tile-major like the weights.
func (q *Q8GMatrix) TableIndex(g, j int) int {
	groups := (q.Rows + q.group() - 1) / q.group()
	return ((j/q4Tile)*groups+g)*q4Tile + j%q4Tile
}

// MatVec computes out = x @ Q for a single activation row: len(x) must be
// Rows and len(out) Cols.
func (q *Q8GMatrix) MatVec(x, out []Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: q8gmatvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	xu, sx := quantizeActs(x)
	grp := q.group()
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		q8gMatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, grp, q.Cols, 0, q.Cols)
		return nil
	}
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		q8gMatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, grp, q.Cols, lo, hi)
	})
	return nil
}

// MatMul computes out = x @ Q for a batch of activation rows, in blocks of
// eight rows per weight stream like QMatrix.MatMul.
func (q *Q8GMatrix) MatMul(x, out *Matrix) error {
	if x.Cols != q.Rows || out.Rows != x.Rows || out.Cols != q.Cols {
		return fmt.Errorf("tensai: q8gmatmul shape mismatch: x %dx%d out %dx%d, want %dx%d",
			x.Rows, x.Cols, out.Rows, out.Cols, q.Rows, q.Cols)
	}
	rows := x.Rows
	grp := q.group()
	xus := make([][]uint8, rows)
	sxs := make([]Float, rows)
	for r := 0; r < rows; r++ {
		xus[r], sxs[r] = quantizeActs(x.Data[r*x.Cols : (r+1)*x.Cols])
	}
	// The row tail (rows%8 leftovers) pads to one full block: zero
	// rows are harmless in the integer kernels, the scratch matrix is
	// shared by every column chunk (they write disjoint ranges), and
	// only the real rows copy back — so the tail streams the weights
	// once instead of once per row.
	var pxus [][]uint8
	var psxs []Float
	var scratch *Matrix
	if rows%8 != 0 {
		pxus, psxs, scratch = padRows8(xus[rows-rows%8:], sxs[rows-rows%8:], len(xus[0]), q.Cols)
	}
	run := func(lo, hi int) {
		var r int
		for ; r+8 <= rows; r += 8 {
			q8gMatmulRows8(out, xus[r:r+8], sxs[r:r+8], r, q.Q, q.Scale, q.ColSum64, grp, q.Cols, lo, hi)
		}
		if r < rows {
			q8gMatmulRows8(scratch, pxus, psxs, 0, q.Q, q.Scale, q.ColSum64, grp, q.Cols, lo, hi)
			for i := 0; i < rows-r; i++ {
				copy(out.Data[(r+i)*q.Cols+lo:(r+i)*q.Cols+hi], scratch.Data[i*q.Cols+lo:i*q.Cols+hi])
			}
		}
	}
	workers := matvecWorkerCount(q.Cols, q.Rows*rows)
	if workers == 1 {
		run(0, q.Cols)
		return nil
	}
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		run(lo, hi)
	})
	return nil
}

// q8gMatvecColsGeneric accumulates out[lo:hi] in pure Go over the same
// 7-bit activations as the AVX2 kernel, so both builds agree exactly.
func q8gMatvecColsGeneric(out []Float, xu []uint8, sx Float, qw []int8, scale []Float, colSum64 []int32, group, cols, lo, hi int) {
	clear(out[lo:hi])
	quads := len(xu) / 4
	groups := (len(xu) + group - 1) / group
	acc := make([]int32, hi-lo)
	for g := 0; g < groups; g++ {
		ib := g * group / 4
		ie := min(ib+group/4, quads)
		clear(acc)
		for i4 := ib; i4 < ie; i4++ {
			x0, x1 := int32(xu[4*i4]), int32(xu[4*i4+1])
			x2, x3 := int32(xu[4*i4+2]), int32(xu[4*i4+3])
			for tile := lo / q4Tile; tile <= (hi-1)/q4Tile; tile++ {
				j0 := max(lo, tile*q4Tile)
				j1 := min(hi, j0+(q4Tile-j0%q4Tile))
				row := qw[tile*quads*4*q4Tile+i4*4*q4Tile+(j0%q4Tile)*4:]
				for j := j0; j < j1; j++ {
					a := j - lo
					acc[a] += x0*int32(row[0]) + x1*int32(row[1]) +
						x2*int32(row[2]) + x3*int32(row[3])
					row = row[4:]
				}
			}
		}
		for j := lo; j < hi; j++ {
			t := ((j/q4Tile)*groups + g) * q4Tile
			out[j] += Float(acc[j-lo]-colSum64[t+j%q4Tile]) * scale[t+j%q4Tile]
		}
	}
	for j := lo; j < hi; j++ {
		out[j] *= sx
	}
}

// q8gMatmulRows8Generic is the eight-row batched body.
func q8gMatmulRows8Generic(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, group, cols, lo, hi int) {
	for r := 0; r < 8; r++ {
		clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+hi])
	}
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + group - 1) / group
	var acc [8][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	for g := 0; g < groups; g++ {
		ib := g * group / 4
		ie := min(ib+group/4, quads)
		for r := range acc {
			clear(acc[r])
		}
		for i4 := ib; i4 < ie; i4++ {
			row := qw[i4*4*q4Tile:]
			for r := 0; r < 8; r++ {
				x0, x1 := int32(xus[r][4*i4]), int32(xus[r][4*i4+1])
				x2, x3 := int32(xus[r][4*i4+2]), int32(xus[r][4*i4+3])
				a := acc[r]
				for j := lo; j < hi; j++ {
					o := (j/q4Tile)*quads*4*q4Tile + (j%q4Tile)*4
					a[j-lo] += x0*int32(row[o]) + x1*int32(row[o+1]) +
						x2*int32(row[o+2]) + x3*int32(row[o+3])
				}
			}
		}
		for r := 0; r < 8; r++ {
			o := out.Data[(r0+r)*cols:]
			for j := lo; j < hi; j++ {
				t := ((j/q4Tile)*groups + g) * q4Tile
				o[j] += Float(acc[r][j-lo]-colSum64[t+j%q4Tile]) * scale[t+j%q4Tile]
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

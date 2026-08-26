package tensai

import (
	"fmt"
	"runtime"
	"sync"
)

// matvecWorkerCount sizes the worker pool for quantized matvecs. Unlike
// dotWorkerCount's cap of 8 (tuned for training matmuls that share the
// machine with other layers), a decode matvec is the only thing running,
// so it may use every CPU.
func matvecWorkerCount(cols, rows int) int {
	if cols*rows < 1<<20 {
		return 1
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > cols {
		workers = cols
	}
	return workers
}

// padRows8 pads a tail of quantized activation rows (fewer than eight)
// to a full block: zero rows are harmless in the integer kernels, and a
// scratch matrix receives the block so callers copy back only the real
// rows. The Q4 path uses the first four entries.
func padRows8(xus [][]uint8, sxs []Float, rowLen, cols int) ([][]uint8, []Float, *Matrix) {
	pxus := make([][]uint8, 8)
	psxs := make([]Float, 8)
	zero := make([]uint8, rowLen)
	for i := 0; i < 8; i++ {
		if i < len(xus) {
			pxus[i], psxs[i] = xus[i], sxs[i]
		} else {
			pxus[i] = zero
		}
	}
	return pxus, psxs, NewMatrix(8, cols)
}

// packQuadsRows8 packs eight quantized activation rows into the
// quad-major interleaved stream the batched kernels broadcast from:
// entry i4*8+r holds row r's quad i4, so the inner loop walks one
// contiguous group of eight per weight load. Packing here runs once
// per row block instead of once per column chunk.
func packQuadsRows8(xus [][]uint8) []uint32 {
	quads := len(xus[0]) / 4
	xq := make([]uint32, 8*quads)
	for r, xu := range xus {
		for i4 := 0; i4 < quads; i4++ {
			xq[i4*8+r] = uint32(xu[4*i4]) | uint32(xu[4*i4+1])<<8 |
				uint32(xu[4*i4+2])<<16 | uint32(xu[4*i4+3])<<24
		}
	}
	return xq
}

// parallelChunks splits [0, n) into align-rounded chunks across the
// workers, the calling goroutine taking the first chunk itself: a
// matvec fan-out spawns workers-1 goroutines and the caller streams
// instead of idling on the join.
func parallelChunks(n, workers, align int, body func(lo, hi int)) {
	chunk := ((n+workers-1)/workers + align - 1) &^ (align - 1)
	var wg sync.WaitGroup
	for lo := chunk; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	body(0, min(chunk, n))
	wg.Wait()
}

// QMatrix is a weight matrix quantized to int8 with one scale per output
// column: W[i][j] ~= Float(q_ij) * Scale[j]. Rows are stored in
// interleaved quads — Q[(i/4)*4*Cols + 4*j + i%4] holds rows i..i+3 of
// column j in four consecutive bytes, zero rows padding the final quad —
// which is the operand layout of the 256-bit u8 x s8 pairwise
// multiply-add followed by the widening i16 pair-add: two instructions
// take a column four rows deep.
//
// Activations quantize per call to 7 bits with a +64 offset (see
// quantizeActs): the unsigned operand of the multiply then stays within
// [0,127], so the i16 pair sums cannot saturate, and the offset folds out
// through ColSum64, the precomputed per-column weight sums times 64.
type QMatrix struct {
	Rows, Cols int
	Q          []int8 // interleaved row quads, padded for 32-byte loads
	Scale      []Float
	ColSum64   []int32
}

// Index returns the position in Q of row i, column j: 32-column tiles
// store their row quads back to back (128 bytes apiece), so a kernel
// worker streams sequential memory.
func (q *QMatrix) Index(i, j int) int {
	quads := (q.Rows + 3) / 4
	return (j/q4Tile)*quads*4*q4Tile + (i/4)*4*q4Tile + (j%q4Tile)*4 + i%4
}

// QuantizeMatrix quantizes column-wise, symmetric around zero with
// round-to-nearest. Columns are independent, so large matrices split
// across CPUs — quantize-at-load of a whole checkpoint is bound by this.
func QuantizeMatrix(m *Matrix) *QMatrix {
	quads := (m.Rows + 3) / 4
	q := &QMatrix{
		Rows:     m.Rows,
		Cols:     m.Cols,
		Q:        make([]int8, ((m.Cols+q4Tile-1)/q4Tile)*quads*4*q4Tile+32),
		Scale:    make([]Float, m.Cols),
		ColSum64: make([]int32, m.Cols+8), // padded for 8-wide loads
	}
	parallelCols(m.Rows, m.Cols, func(lo, hi int) {
		quantizeColumns(m, q, lo, hi)
	})
	return q
}

// parallelCols splits [0, cols) across workers when the matrix is big
// enough to pay for them.
func parallelCols(rows, cols int, f func(lo, hi int)) {
	workers := runtime.NumCPU()
	if rows*cols < 1<<20 || workers <= 1 {
		f(0, cols)
		return
	}
	if workers > cols {
		workers = cols
	}
	chunk := (cols + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < cols; lo += chunk {
		hi := min(lo+chunk, cols)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			f(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func quantizeColumns(m *Matrix, q *QMatrix, colLo, colHi int) {
	for j := colLo; j < colHi; j++ {
		var maxAbs Float
		for i := 0; i < m.Rows; i++ {
			v := m.Data[i*m.Cols+j]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		s := maxAbs / 127
		q.Scale[j] = s
		if s == 0 {
			continue
		}
		inv := 1 / s
		var sum int32
		for i := 0; i < m.Rows; i++ {
			v := m.Data[i*m.Cols+j] * inv
			if v >= 0 {
				v += 0.5
			} else {
				v -= 0.5
			}
			w := int8(v)
			q.Q[q.Index(i, j)] = w
			sum += int32(w)
		}
		q.ColSum64[j] = 64 * sum
	}
}

// quantizeActs maps an activation row to 7-bit offset-binary: round(x/sx)
// clamped to [-63,63], stored +64 so every value sits in [1,127]. The
// range keeps u8 x s8 pair sums inside int16. Pad bytes (which the weight
// layout pairs with zero weight rows) fill the final quad. The AVX2 build
// runs a vectorized twin with the same rounding (see quantacts_simd.go).
func quantizeActs(x []Float) (xu []uint8, sx Float) {
	padded := (len(x) + 3) &^ 3
	xu = make([]uint8, padded)
	for i := len(x); i < padded; i++ {
		xu[i] = 64 // pairs with a zero weight row: contributes 64*0
	}
	sx = quantizeActsInto(x, xu)
	return xu, sx
}

// quantizeActsScalar is the portable body; the round is half away from
// zero, truncation after a signed 0.5 nudge.
func quantizeActsScalar(x []Float, xu []uint8) Float {
	var maxAbs Float
	for _, v := range x {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	sx := maxAbs / 63
	if sx == 0 {
		for i := range x {
			xu[i] = 64
		}
		return 0
	}
	inv := 1 / sx
	for i, v := range x {
		f := v * inv
		if f >= 0 {
			f += 0.5
		} else {
			f -= 0.5
		}
		n := int(f)
		if n < -63 {
			n = -63
		} else if n > 63 {
			n = 63
		}
		xu[i] = uint8(n + 64)
	}
	return sx
}

// MatVec computes out = x @ Q for a single activation row: len(x) must be
// Rows and len(out) Cols. The activation row is quantized once per call;
// output columns split across CPUs.
func (q *QMatrix) MatVec(x, out []Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: qmatvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	xu, sx := quantizeActs(x)
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		qmatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, q.Cols, 0, q.Cols)
		return nil
	}
	// Chunks stay multiples of 8 so only the last one has a scalar tail.
	parallelChunks(q.Cols, workers, q4Tile, func(lo, hi int) {
		qmatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
	})
	return nil
}

// MatMul computes out = x @ Q for a batch of activation rows — the
// prompt-prefill shape. Each row quantizes to the same 7-bit form MatVec
// uses, and the kernel processes rows in blocks of eight against one
// streaming pass over the weights, so the weight traffic that dominates a
// single matvec amortizes across the batch.
func (q *QMatrix) MatMul(x, out *Matrix) error {
	if x.Cols != q.Rows || out.Rows != x.Rows || out.Cols != q.Cols {
		return fmt.Errorf("tensai: qmatmul shape mismatch: x %dx%d out %dx%d, want %dx%d",
			x.Rows, x.Cols, out.Rows, out.Cols, q.Rows, q.Cols)
	}
	rows := x.Rows
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
	xqb := make([][]uint32, rows/8)
	for b := range xqb {
		xqb[b] = packQuadsRows8(xus[b*8 : b*8+8])
	}
	var pxus [][]uint8
	var pxq []uint32
	var psxs []Float
	var scratch *Matrix
	if rows%8 != 0 {
		pxus, psxs, scratch = padRows8(xus[rows-rows%8:], sxs[rows-rows%8:], len(xus[0]), q.Cols)
		pxq = packQuadsRows8(pxus)
	}
	run := func(lo, hi int) {
		var r int
		for ; r+8 <= rows; r += 8 {
			qmatmulRows8(out, xus[r:r+8], xqb[r/8], sxs[r:r+8], r, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
		}
		if r < rows {
			qmatmulRows8(scratch, pxus, pxq, psxs, 0, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
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

// qmatmulRows8Generic is the eight-row batched body: one pass over the
// weights feeds eight output rows.
func qmatmulRows8Generic(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	var acc [8][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	quads := len(xus[0]) / 4
	for i4 := 0; i4 < quads; i4++ {
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
			o[j] = Float(acc[r][j-lo]-colSum64[j]) * scale[j] * sxs[r]
		}
	}
}

// qmatvecColsGeneric accumulates out[lo:hi] of the quantized product in
// pure Go, over the same 7-bit activations as the AVX2 kernel so both
// builds compute identical results. qmatvecCols (see quant_simd.go and
// quant_generic.go) dispatches to the AVX2 kernel when available.
func qmatvecColsGeneric(out []Float, xu []uint8, sx Float, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	acc := make([]int32, hi-lo)
	quads := len(xu) / 4
	for i4 := 0; i4 < quads; i4++ {
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
		out[j] = Float(acc[j-lo]-colSum64[j]) * scale[j] * sx
	}
}

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
	workers := runtime.NumCPU()
	if workers > cols {
		workers = cols
	}
	return workers
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

// QuantizeMatrix quantizes column-wise, symmetric around zero with
// round-to-nearest. Columns are independent, so large matrices split
// across CPUs — quantize-at-load of a whole checkpoint is bound by this.
func QuantizeMatrix(m *Matrix) *QMatrix {
	quads := (m.Rows + 3) / 4
	q := &QMatrix{
		Rows:     m.Rows,
		Cols:     m.Cols,
		Q:        make([]int8, quads*4*m.Cols+32),
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
			q.Q[(i/4)*4*m.Cols+4*j+i%4] = w
			sum += int32(w)
		}
		q.ColSum64[j] = 64 * sum
	}
}

// quantizeActs maps an activation row to 7-bit offset-binary: round(x/sx)
// clamped to [-63,63], stored +64 so every value sits in [1,127]. The
// range keeps u8 x s8 pair sums inside int16. Pad bytes (which the weight
// layout pairs with zero weight rows) fill the final quad.
func quantizeActs(x []Float) (xu []uint8, sx Float) {
	var maxAbs Float
	for _, v := range x {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	padded := (len(x) + 3) &^ 3
	xu = make([]uint8, padded)
	for i := len(x); i < padded; i++ {
		xu[i] = 64 // pairs with a zero weight row: contributes 64*0
	}
	sx = maxAbs / 63
	if sx == 0 {
		for i := range xu {
			xu[i] = 64
		}
		return xu, 0
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
	return xu, sx
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
	chunk := ((q.Cols+workers-1)/workers + 7) &^ 7
	var wg sync.WaitGroup
	for lo := 0; lo < q.Cols; lo += chunk {
		hi := min(lo+chunk, q.Cols)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			qmatvecCols(out, xu, sx, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
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
	run := func(lo, hi int) {
		var r int
		for ; r+8 <= rows; r += 8 {
			qmatmulRows8(out, xus[r:r+8], sxs[r:r+8], r, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
		}
		for ; r < rows; r++ {
			qmatvecCols(out.Data[r*q.Cols:(r+1)*q.Cols], xus[r], sxs[r], q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
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

// qmatmulRows8Generic is the eight-row batched body: one pass over the
// weights feeds eight output rows.
func qmatmulRows8Generic(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	var acc [8][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	quads := len(xus[0]) / 4
	for i4 := 0; i4 < quads; i4++ {
		row := qw[i4*4*cols:]
		for r := 0; r < 8; r++ {
			x0, x1 := int32(xus[r][4*i4]), int32(xus[r][4*i4+1])
			x2, x3 := int32(xus[r][4*i4+2]), int32(xus[r][4*i4+3])
			a := acc[r]
			for j := lo; j < hi; j++ {
				a[j-lo] += x0*int32(row[4*j]) + x1*int32(row[4*j+1]) +
					x2*int32(row[4*j+2]) + x3*int32(row[4*j+3])
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
	for i4 := 0; i4 < len(xu)/4; i4++ {
		x0, x1 := int32(xu[4*i4]), int32(xu[4*i4+1])
		x2, x3 := int32(xu[4*i4+2]), int32(xu[4*i4+3])
		row := qw[i4*4*cols:]
		for j := lo; j < hi; j++ {
			acc[j-lo] += x0*int32(row[4*j]) + x1*int32(row[4*j+1]) +
				x2*int32(row[4*j+2]) + x3*int32(row[4*j+3])
		}
	}
	for j := lo; j < hi; j++ {
		out[j] = Float(acc[j-lo]-colSum64[j]) * scale[j] * sx
	}
}

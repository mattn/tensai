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
// interleaved pairs — Q[(i/2)*2*Cols + 2*j] holds row i (even) and the
// following byte row i+1 of the same column, with a zero row appended when
// Rows is odd — which is exactly the operand layout of the u8 x s8
// pairwise multiply-add the AVX2 kernel is built on.
//
// Activations quantize per call to 7 bits with a +64 offset (see
// quantizeActs): the unsigned operand of the multiply then stays within
// [0,127], so the i16 pair sums cannot saturate, and the offset folds out
// through ColSum64, the precomputed per-column weight sums times 64.
type QMatrix struct {
	Rows, Cols int
	Q          []int8 // interleaved row pairs, padded for 16-byte loads
	Scale      []Float
	ColSum64   []int32
}

// QuantizeMatrix quantizes column-wise, symmetric around zero with
// round-to-nearest.
func QuantizeMatrix(m *Matrix) *QMatrix {
	pairs := (m.Rows + 1) / 2
	q := &QMatrix{
		Rows:     m.Rows,
		Cols:     m.Cols,
		Q:        make([]int8, pairs*2*m.Cols+16),
		Scale:    make([]Float, m.Cols),
		ColSum64: make([]int32, m.Cols+8), // padded for 8-wide loads
	}
	for j := 0; j < m.Cols; j++ {
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
			q.Q[(i/2)*2*m.Cols+2*j+i%2] = w
			sum += int32(w)
		}
		q.ColSum64[j] = 64 * sum
	}
	return q
}

// quantizeActs maps an activation row to 7-bit offset-binary: round(x/sx)
// clamped to [-63,63], stored +64 so every value sits in [1,127]. The
// range keeps u8 x s8 pair sums inside int16. A zero pad byte (which the
// weight layout pairs with a zero weight row) covers odd row counts.
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
	xu = make([]uint8, len(x)+len(x)%2)
	sx = maxAbs / 63
	if sx == 0 {
		for i := range xu {
			xu[i] = 64
		}
		if len(x)%2 == 1 {
			xu[len(x)] = 64
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
	if len(x)%2 == 1 {
		xu[len(x)] = 64 // pairs with the zero weight row: contributes 64*0
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
// uses, and the kernel processes rows in blocks of four against one
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
		for ; r+4 <= rows; r += 4 {
			qmatmulCols4(out, xus[r:r+4], sxs[r:r+4], r, q.Q, q.Scale, q.ColSum64, q.Cols, lo, hi)
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

// qmatmulCols4Generic is the four-row batched body: one pass over the
// weights feeds four output rows.
func qmatmulCols4Generic(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	var acc [4][]int32
	for r := range acc {
		acc[r] = make([]int32, hi-lo)
	}
	pairs := len(xus[0]) / 2
	for i2 := 0; i2 < pairs; i2++ {
		row := qw[i2*2*cols:]
		for r := 0; r < 4; r++ {
			x0, x1 := int32(xus[r][2*i2]), int32(xus[r][2*i2+1])
			a := acc[r]
			for j := lo; j < hi; j++ {
				a[j-lo] += x0*int32(row[2*j]) + x1*int32(row[2*j+1])
			}
		}
	}
	for r := 0; r < 4; r++ {
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
	for i2 := 0; i2 < len(xu)/2; i2++ {
		x0, x1 := int32(xu[2*i2]), int32(xu[2*i2+1])
		row := qw[i2*2*cols:]
		for j := lo; j < hi; j++ {
			acc[j-lo] += x0*int32(row[2*j]) + x1*int32(row[2*j+1])
		}
	}
	for j := lo; j < hi; j++ {
		out[j] = Float(acc[j-lo]-colSum64[j]) * scale[j] * sx
	}
}

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
// column: W[i][j] ~= Float(Q[i*Cols+j]) * Scale[j]. Inference matvecs are
// memory-bandwidth bound, so weight-only quantization — activations stay
// float32 — quarters their traffic while keeping the accumulation in
// float32.
type QMatrix struct {
	Rows, Cols int
	Q          []int8 // row-major, padded so 16-byte loads never run off the end
	Scale      []Float
}

// QuantizeMatrix quantizes column-wise, symmetric around zero with
// round-to-nearest, so each output column keeps its own dynamic range.
func QuantizeMatrix(m *Matrix) *QMatrix {
	q := &QMatrix{
		Rows:  m.Rows,
		Cols:  m.Cols,
		Q:     make([]int8, m.Rows*m.Cols+16),
		Scale: make([]Float, m.Cols),
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
		for i := 0; i < m.Rows; i++ {
			v := m.Data[i*m.Cols+j] * inv
			if v >= 0 {
				v += 0.5
			} else {
				v -= 0.5
			}
			q.Q[i*m.Cols+j] = int8(v)
		}
	}
	return q
}

// MatVec computes out = x @ Q for a single activation row: len(x) must be
// Rows and len(out) Cols. Output columns are split across CPUs for large
// matrices, mirroring DotInto.
func (q *QMatrix) MatVec(x, out []Float) error {
	if len(x) != q.Rows || len(out) != q.Cols {
		return fmt.Errorf("tensai: qmatvec shape mismatch: x=%d out=%d, want %dx%d",
			len(x), len(out), q.Rows, q.Cols)
	}
	workers := matvecWorkerCount(q.Cols, q.Rows)
	if workers == 1 {
		qmatvecCols(out, x, q.Q, q.Scale, q.Cols, 0, q.Cols)
		return nil
	}
	// Chunks stay multiples of 8 so only the last one has a scalar tail.
	chunk := ((q.Cols+workers-1)/workers + 7) &^ 7
	var wg sync.WaitGroup
	for lo := 0; lo < q.Cols; lo += chunk {
		hi := lo + chunk
		if hi > q.Cols {
			hi = q.Cols
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			qmatvecCols(out, x, q.Q, q.Scale, q.Cols, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// qmatvecColsGeneric accumulates out[lo:hi] of out = x @ Q in pure Go and
// applies the column scales. qmatvecCols (see quant_simd.go and
// quant_generic.go) dispatches to the AVX2 kernel when available.
func qmatvecColsGeneric(out, x []Float, qw []int8, scale []Float, cols, lo, hi int) {
	clear(out[lo:hi])
	for i, xi := range x {
		if xi == 0 {
			continue
		}
		row := qw[i*cols : i*cols+cols]
		for j := lo; j < hi; j++ {
			out[j] += xi * Float(row[j])
		}
	}
	for j := lo; j < hi; j++ {
		out[j] *= scale[j]
	}
}

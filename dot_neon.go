//go:build goexperiment.simd && arm64 && go1.27

package tensai

import "runtime"

// The dense float matmuls keep the portable bodies on arm64. They are the
// training path; the decode kernels that carry a chat are in quant and
// internal/kernels, which do have NEON here.

func dotRows(out, a, b *Matrix, lo, hi int) {
	dotRowsGeneric(out, a, b, lo, hi)
}

func dotWorkerCount(rows, inner, cols int) int {
	workers := 1
	if rows*inner*cols >= 1<<20 {
		workers = runtime.NumCPU()
		if workers > 8 {
			workers = 8
		}
		if workers > rows {
			workers = rows
		}
	}
	return workers
}

func dotTATall(out, a, b *Matrix, lo, hi int) {
	dotTARowsGeneric(out, a, b, lo, hi)
}

func dotTARows(out, a, b *Matrix, lo, hi int) {
	dotTARowsGeneric(out, a, b, lo, hi)
}

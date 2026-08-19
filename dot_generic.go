//go:build !goexperiment.simd || !amd64

package tensai

import "runtime"

// dotRows computes rows lo..hi of out = a * b. This is the portable
// fallback; build with GOEXPERIMENT=simd on amd64 to get the AVX2 kernel in
// dot_simd.go.
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

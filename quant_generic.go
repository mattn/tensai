//go:build !goexperiment.simd || !amd64

package tensai

// Portable dispatcher for the quantized matvec kernel; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quant_simd.go.

func qmatvecCols(out, x []Float, qw []int8, scale []Float, cols, lo, hi int) {
	qmatvecColsGeneric(out, x, qw, scale, cols, lo, hi)
}

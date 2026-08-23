//go:build !goexperiment.simd || !amd64

package tensai

// Portable dispatcher for the 4-bit matvec kernel; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quant4_simd.go.

func q4matvecCols(out []Float, xu []uint8, sx Float, gsum []int32, qw []uint8, scale []Float, cols, lo, hi int) {
	q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, cols, lo, hi)
}

func q4matmulCols4(out *Matrix, xus [][]uint8, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, cols, lo, hi int) {
	q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, cols, lo, hi)
}

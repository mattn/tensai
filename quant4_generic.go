//go:build !goexperiment.simd || !amd64

package tensai

// Portable dispatcher for the 4-bit matvec kernel; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quant4_simd.go.

func q4matvecCols(out []Float, xu []uint8, xq []uint32, sx Float, gsum []int32, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, lo, hi)
}

func q4matmulCols4(out *Matrix, xus [][]uint8, _ [][]uint32, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, lo, hi)
}

func q4matmulCols8(out *Matrix, xus [][]uint8, _ [][]uint32, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	q4matmulCols4Generic(out, xus[:4], sxs[:4], gsums[:4], r0, qw, scale, sm, group, cols, lo, hi)
	q4matmulCols4Generic(out, xus[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, sm, group, cols, lo, hi)
}

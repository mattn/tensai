//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package quant

import "github.com/mattn/tensai"

// Portable dispatchers for the ternary kernels; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 versions in ternary_simd.go.

func ternaryMatvecCols(out []tensai.Float, xs []int8, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	ternaryMatvecColsGeneric(out, xs, sx, gsum, qw, scale, rows, cols, lo, hi)
}

func ternaryMatmulRows8(out *tensai.Matrix, xss [][]int8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	ternaryMatmulRows8Generic(out, xss, sxs, gsums, r0, qw, scale, rows, cols, lo, hi)
}

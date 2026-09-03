//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package quant

import "github.com/mattn/tensai"

// Portable dispatchers for the grouped int8 kernels; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 versions in quant8g_simd.go.

func q8gMatvecCols(out []tensai.Float, xu []uint8, sx tensai.Float, qw []int8, scale []tensai.Float, colSum64 []int32, group, cols, lo, hi int) {
	q8gMatvecColsGeneric(out, xu, sx, qw, scale, colSum64, group, cols, lo, hi)
}

func q8gMatmulRows8(out *tensai.Matrix, xus [][]uint8, xq []uint32, sxs []tensai.Float, r0 int, qw []int8, scale []tensai.Float, colSum64 []int32, group, cols, lo, hi int) {
	q8gMatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, group, cols, lo, hi)
}

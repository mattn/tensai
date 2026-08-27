//go:build !goexperiment.simd || !amd64

package quant

import "github.com/mattn/tensai"

// Portable dispatchers for the MXFP4 kernels; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 versions in mxfp4_simd.go.

func mxfp4MatvecCols(out []tensai.Float, xu []uint8, sx tensai.Float, qw []uint8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	mxfp4MatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, lo, hi)
}

func mxfp4MatmulRows8(out *tensai.Matrix, xus [][]uint8, sxs []tensai.Float, r0 int, qw []uint8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	mxfp4MatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
}

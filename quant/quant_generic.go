//go:build !goexperiment.simd || !amd64

package quant

import "github.com/mattn/tensai"

// Portable dispatcher for the quantized matvec kernel; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quant_simd.go.

func qmatvecCols(out []tensai.Float, xu []uint8, sx tensai.Float, qw []int8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	qmatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, lo, hi)
}

func qmatmulRows8(out *tensai.Matrix, xus [][]uint8, xq []uint32, sxs []tensai.Float, r0 int, qw []int8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	qmatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
}

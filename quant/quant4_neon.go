//go:build goexperiment.simd && arm64

package quant

import "github.com/mattn/tensai"

// The 4-bit kernels keep the portable bodies on arm64 for now. Unpacking
// nibbles into the quad layout is the same shape of work the int8 kernel
// in quant_neon.go does, one mask and shift ahead of it; it is the next
// one to vectorize here.

func q4matvecCols(out []tensai.Float, xu []uint8, xq []uint32, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, lo, hi)
}

func q4matmulCols4(out *tensai.Matrix, xus [][]uint8, _ [][]uint32, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, lo, hi)
}

func q4matmulCols8(out *tensai.Matrix, xus [][]uint8, _ [][]uint32, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	q4matmulCols4Generic(out, xus[:4], sxs[:4], gsums[:4], r0, qw, scale, sm, group, cols, lo, hi)
	q4matmulCols4Generic(out, xus[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, sm, group, cols, lo, hi)
}

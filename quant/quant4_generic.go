//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package quant

import "github.com/mattn/tensai"

// Portable dispatcher for the 4-bit matvec kernel; build with
// GOEXPERIMENT=simd for the AVX2 version in quant4_simd.go on amd64 or
// the NEON version in quant4_neon.go on arm64.

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

// packQuad packs a re-centered activation quad in weight-layout order.
// Only the vector kernels read it; the portable bodies take xu directly.
func packQuad(x []uint8) uint32 {
	return recenter(x[0]) | recenter(x[1])<<8 | recenter(x[2])<<16 | recenter(x[3])<<24
}

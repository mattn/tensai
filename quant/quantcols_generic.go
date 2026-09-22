//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package quant

import "github.com/mattn/tensai"

// Portable dispatcher for the column-tile quantizer; build with
// GOEXPERIMENT=simd for the vector versions in quantcols_simd.go and
// quantcols_neon.go.

func maxAbsCols(maxAbs, data []tensai.Float, stride, rows int) {
	maxAbsColsScalar(maxAbs, data, stride, rows)
}

func quantCols(dst []int8, data, inv []tensai.Float, sums []int32, stride, rows, w int) {
	quantColsScalar(dst, data, inv, sums, stride, rows, w)
}

func quant4Cols(dst []int8, data, inv []tensai.Float, stride, rows, w int) {
	quant4ColsScalar(dst, data, inv, stride, rows, w)
}

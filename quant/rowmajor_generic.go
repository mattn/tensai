//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package quant

func packRowQuad(dst []uint32, src []int8, stride, rows, cols int) {
	packRowQuadGeneric(dst, src, stride, rows, cols)
}

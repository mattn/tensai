//go:build goexperiment.simd && amd64

package quant

import "github.com/mattn/tensai/internal/simd"

// Transpose four columns of four row bytes with one PSHUFB.
func packRowQuad(dst []uint32, src []int8, stride, rows, cols int) {
	if !simd.HasAVX2 {
		packRowQuadGeneric(dst, src, stride, rows, cols)
		return
	}
	idx := simd.LoadI8x16([]int8{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15})
	n := cols &^ 3
	for c := 0; c < n; c += 4 {
		v := simd.LoadI8x16(src[4*c:]).AsUint8x16().PermuteOrZero(idx).AsUint32x4()
		dst[c/4] = v.GetElem(0)
		if rows > 1 {
			dst[stride+c/4] = v.GetElem(1)
		}
		if rows > 2 {
			dst[2*stride+c/4] = v.GetElem(2)
		}
		if rows > 3 {
			dst[3*stride+c/4] = v.GetElem(3)
		}
	}
	if n < cols {
		packRowQuadGeneric(dst[n/4:], src[4*n:], stride, rows, cols-n)
	}
}

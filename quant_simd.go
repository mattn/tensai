//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// AVX2 kernel for the quantized matvec: 16-byte int8 loads (the QMatrix
// padding keeps them in bounds), the low 8 lanes widened to int32,
// converted to float32, and FMA'd against the broadcast activation. The
// scalar tail runs after the vector state is cleared, mirroring the
// discipline in dot_simd.go.
func qmatvecCols(out, x []Float, qw []int8, scale []Float, cols, lo, hi int) {
	if !hasAVX2 {
		qmatvecColsGeneric(out, x, qw, scale, cols, lo, hi)
		return
	}
	wide := lo + ((hi - lo) &^ 31)
	vecEnd := lo + ((hi - lo) &^ 7)
	clear(out[lo:hi])
	if vecEnd > lo {
		for i := range x {
			xv := archsimd.BroadcastFloat32x8(x[i])
			row := qw[i*cols:]
			j := lo
			for ; j < wide; j += 32 {
				w0 := loadI8x16(row[j:]).ExtendLo8ToInt32().ConvertToFloat32()
				w1 := loadI8x16(row[j+8:]).ExtendLo8ToInt32().ConvertToFloat32()
				w2 := loadI8x16(row[j+16:]).ExtendLo8ToInt32().ConvertToFloat32()
				w3 := loadI8x16(row[j+24:]).ExtendLo8ToInt32().ConvertToFloat32()
				storeF32x8(w0.MulAdd(xv, loadF32x8(out[j:])), out[j:])
				storeF32x8(w1.MulAdd(xv, loadF32x8(out[j+8:])), out[j+8:])
				storeF32x8(w2.MulAdd(xv, loadF32x8(out[j+16:])), out[j+16:])
				storeF32x8(w3.MulAdd(xv, loadF32x8(out[j+24:])), out[j+24:])
			}
			for ; j < vecEnd; j += 8 {
				w := loadI8x16(row[j:]).ExtendLo8ToInt32().ConvertToFloat32()
				storeF32x8(w.MulAdd(xv, loadF32x8(out[j:])), out[j:])
			}
		}
		for j := lo; j < vecEnd; j += 8 {
			storeF32x8(loadF32x8(out[j:]).Mul(loadF32x8(scale[j:])), out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		qmatvecColsGeneric(out, x, qw, scale, cols, vecEnd, hi)
	}
}

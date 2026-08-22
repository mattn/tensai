//go:build goexperiment.simd && amd64

package tensai

import (
	"unsafe"

	"simd/archsimd"
)

// AVX2 kernel for the 4-bit matvec: an 8-byte load carries 8 column pairs,
// the low nibbles masked out directly and the high nibbles via a 16-bit
// shift (AVX2 has no byte shift), both widened to float32 and FMA'd
// against the broadcast activation. Group scales fold in vectorized at
// each group boundary; the scalar tail runs through the generic body after
// the vector state is cleared.
func q4matvecCols(out, x []Float, qw []byte, scale []Float, cols, lo, hi int, tmp []Float) {
	if !hasAVX2 {
		q4matvecColsGeneric(out, x, qw, scale, cols, lo, hi, tmp)
		return
	}
	half := cols / 2
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		mask := archsimd.BroadcastInt8x16(0x0F)
		eight := archsimd.BroadcastFloat32x8(8)
		clear(out[lo:vecEnd])
		clear(out[half+lo : half+vecEnd])
		groups := (len(x) + q4Group - 1) / q4Group
		for g := 0; g < groups; g++ {
			rlo := g * q4Group
			rhi := min(rlo+q4Group, len(x))
			clear(tmp[lo:vecEnd])
			clear(tmp[half+lo : half+vecEnd])
			for i := rlo; i < rhi; i++ {
				xv := archsimd.BroadcastFloat32x8(x[i])
				row := qw[i*half:]
				for j := lo; j < vecEnd; j += 8 {
					v := loadI8x16(unsafe.Slice((*int8)(unsafe.Pointer(&row[j])), 16))
					loNib := v.And(mask).ExtendLo8ToInt32().ConvertToFloat32().Sub(eight)
					hiNib := v.AsUint16x8().ShiftAllRight(4).AsInt8x16().And(mask).ExtendLo8ToInt32().ConvertToFloat32().Sub(eight)
					storeF32x8(loNib.MulAdd(xv, loadF32x8(tmp[j:])), tmp[j:])
					storeF32x8(hiNib.MulAdd(xv, loadF32x8(tmp[half+j:])), tmp[half+j:])
				}
			}
			srow := scale[g*cols:]
			for j := lo; j < vecEnd; j += 8 {
				storeF32x8(loadF32x8(tmp[j:]).MulAdd(loadF32x8(srow[j:]), loadF32x8(out[j:])), out[j:])
				storeF32x8(loadF32x8(tmp[half+j:]).MulAdd(loadF32x8(srow[half+j:]), loadF32x8(out[half+j:])), out[half+j:])
			}
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q4matvecColsGeneric(out, x, qw, scale, cols, vecEnd, hi, tmp)
	}
}

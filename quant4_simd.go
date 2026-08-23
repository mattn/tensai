//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// AVX2 kernel for the 4-bit matvec, sharing the u8 x s8 pairwise
// multiply-add design of the int8 kernel: a load's low 8 bytes zero-extend
// to u16 lanes, where masks and a shift split each byte's nibbles into the
// adjacent-byte pair layout the multiply wants; the signed operand is the
// broadcast 7-bit activation pair. Nibbles are at most 15, so pair sums
// cannot approach saturation at any activation width. Each group of 32 row
// pairs is one register-accumulated block: eight i32x8 accumulators cover
// a 64-column tile (one cache line per pair row), and the group's scales
// and offset correction fold in vectorized at the block boundary.
func q4matvecCols(out []Float, xu []uint8, sx Float, gsum []int32, qw []uint8, scale []Float, cols, lo, hi int) {
	if !hasAVX2 {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, cols, lo, hi)
		return
	}
	pairs := len(xu) / 2
	vecEnd := lo + ((hi - lo) &^ 63)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x8(0x000F)
		mHi := archsimd.BroadcastUint16x8(0x0F00)
		clear(out[lo:vecEnd])
		for g := 0; g < len(gsum); g++ {
			ib := g * q4Group / 2
			ie := min(ib+q4Group/2, pairs)
			corr := archsimd.BroadcastInt32x8(8 * gsum[g])
			srow := scale[g*cols:]
			for jt := lo; jt < vecEnd; jt += 64 {
				var a [8]archsimd.Int32x8
				for i2 := ib; i2 < ie; i2++ {
					x0 := uint8(int8(int(xu[2*i2]) - 64))
					x1 := uint8(int8(int(xu[2*i2+1]) - 64))
					xp := archsimd.BroadcastUint16x8(uint16(x0) | uint16(x1)<<8).AsUint8x16().AsInt8x16()
					row := qw[i2*cols+jt:]
					for k := 0; k < 8; k++ {
						v := loadU8x16(row[8*k:]).ExtendLo8ToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi))
						a[k] = a[k].Add(pair.AsUint8x16().DotProductPairsSaturated(xp).ExtendToInt32())
					}
				}
				for k := 0; k < 8; k++ {
					j := jt + 8*k
					f := a[k].Sub(corr).ConvertToFloat32().Mul(loadF32x8(srow[j:]))
					storeF32x8(loadF32x8(out[j:]).Add(f), out[j:])
				}
			}
		}
		sxv := archsimd.BroadcastFloat32x8(sx)
		for j := lo; j < vecEnd; j += 8 {
			storeF32x8(loadF32x8(out[j:]).Mul(sxv), out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, cols, vecEnd, hi)
	}
}

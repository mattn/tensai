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

// q4matmulCols4 is the four-row batched form: per 16-column tile each
// pair-row's nibble load and split happen once and feed four broadcast
// activation pairs — eight register accumulators, two per row — so the
// nibble traffic that dominates a single matvec amortizes fourfold. Group
// scale and offset folding runs per row at each block edge.
func q4matmulCols4(out *Matrix, xus [][]uint8, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, cols, lo, hi int) {
	if !hasAVX2 {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, cols, lo, hi)
		return
	}
	pairs := len(xus[0]) / 2
	vecEnd := lo + ((hi - lo) &^ 15)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x8(0x000F)
		mHi := archsimd.BroadcastUint16x8(0x0F00)
		for r := 0; r < 4; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		for g := 0; g < len(gsums[0]); g++ {
			ib := g * q4Group / 2
			ie := min(ib+q4Group/2, pairs)
			srow := scale[g*cols:]
			for jt := lo; jt < vecEnd; jt += 16 {
				var a [4][2]archsimd.Int32x8
				for i2 := ib; i2 < ie; i2++ {
					row := qw[i2*cols+jt:]
					v0 := loadU8x16(row).ExtendLo8ToUint16()
					p0 := v0.And(mLo).Or(v0.ShiftAllLeft(4).And(mHi)).AsUint8x16()
					v1 := loadU8x16(row[8:]).ExtendLo8ToUint16()
					p1 := v1.And(mLo).Or(v1.ShiftAllLeft(4).And(mHi)).AsUint8x16()
					for r := 0; r < 4; r++ {
						x0 := uint8(int8(int(xus[r][2*i2]) - 64))
						x1 := uint8(int8(int(xus[r][2*i2+1]) - 64))
						xp := archsimd.BroadcastUint16x8(uint16(x0) | uint16(x1)<<8).AsUint8x16().AsInt8x16()
						a[r][0] = a[r][0].Add(p0.DotProductPairsSaturated(xp).ExtendToInt32())
						a[r][1] = a[r][1].Add(p1.DotProductPairsSaturated(xp).ExtendToInt32())
					}
				}
				for r := 0; r < 4; r++ {
					corr := archsimd.BroadcastInt32x8(8 * gsums[r][g])
					o := out.Data[(r0+r)*cols:]
					for k := 0; k < 2; k++ {
						j := jt + 8*k
						f := a[r][k].Sub(corr).ConvertToFloat32().Mul(loadF32x8(srow[j:]))
						storeF32x8(loadF32x8(o[j:]).Add(f), o[j:])
					}
				}
			}
		}
		for r := 0; r < 4; r++ {
			sxv := archsimd.BroadcastFloat32x8(sxs[r])
			o := out.Data[(r0+r)*cols:]
			for j := lo; j < vecEnd; j += 8 {
				storeF32x8(loadF32x8(o[j:]).Mul(sxv), o[j:])
			}
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, cols, vecEnd, hi)
	}
}

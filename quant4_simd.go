//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// 256-bit AVX2 kernels for the 4-bit matvec and matmul over the quad-row
// layout: sixteen loaded bytes hold eight columns four rows deep (two
// nibble bytes per column), the zero-extend to u16 lanes plus a mask and
// shift split each byte into adjacent nibble lanes — landing exactly in
// the int8 kernels' quad layout — and the u8 x s8 pairwise multiply-add
// against a broadcast of re-centered signed activations, followed by the
// widening i16 pair-add, takes the column's whole quad in two multiplies.
// Nibbles are at most 15 and |activations| at most 63, so the i16 pair
// sums sit far inside saturation. Group scales and the nibble-offset
// correction (8 times the group's activation sum) fold in per group.

// q4xQuad packs four re-centered activations into the u32 a broadcast
// turns into the signed multiply operand.
func q4xQuad(xu []uint8, i4 int) uint32 {
	x0 := uint32(uint8(int8(int(xu[4*i4]) - 64)))
	x1 := uint32(uint8(int8(int(xu[4*i4+1]) - 64)))
	x2 := uint32(uint8(int8(int(xu[4*i4+2]) - 64)))
	x3 := uint32(uint8(int8(int(xu[4*i4+3]) - 64)))
	return x0 | x1<<8 | x2<<16 | x3<<24
}

func q4matvecCols(out []Float, xu []uint8, sx Float, gsum []int32, qw []uint8, scale []Float, group, cols, lo, hi int) {
	if !hasAVX2 {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, group, cols, lo, hi)
		return
	}
	quads := len(xu) / 4
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		ones := archsimd.BroadcastInt16x16(1)
		clear(out[lo:vecEnd])
		for g := 0; g < len(gsum); g++ {
			ib := g * group / 4
			ie := min(ib+group/4, quads)
			corr := archsimd.BroadcastInt32x8(8 * gsum[g])
			srow := scale[g*cols:]
			for jt := lo; jt < vecEnd; jt += 32 {
				var a [4]archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					xp := archsimd.BroadcastUint32x8(q4xQuad(xu, i4)).AsInt8x32()
					row := qw[i4*2*cols+2*jt:]
					for k := 0; k < 4; k++ {
						v := loadU8x16(row[16*k:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						a[k] = a[k].Add(pair.DotProductPairsSaturated(xp).DotProductPairs(ones))
					}
				}
				for k := 0; k < 4; k++ {
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
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, group, cols, vecEnd, hi)
	}
}

// q4matmulCols4 is the four-row batched form: per eight-column tile each
// sixteen-byte nibble load unpacks once and feeds four broadcast
// activation quads, amortizing the nibble traffic fourfold.
func q4matmulCols4(out *Matrix, xus [][]uint8, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, group, cols, lo, hi int) {
	if !hasAVX2 {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, group, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	var xq [4][]uint32
	for r := 0; r < 4; r++ {
		xq[r] = make([]uint32, quads)
		for i4 := 0; i4 < quads; i4++ {
			xq[r][i4] = q4xQuad(xus[r], i4)
		}
	}
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		ones := archsimd.BroadcastInt16x16(1)
		for r := 0; r < 4; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		for g := 0; g < len(gsums[0]); g++ {
			ib := g * group / 4
			ie := min(ib+group/4, quads)
			srow := scale[g*cols:]
			for jt := lo; jt < vecEnd; jt += 8 {
				var a [4]archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					v := loadU8x16(qw[i4*2*cols+2*jt:]).ExtendToUint16()
					pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
					for r := 0; r < 4; r++ {
						xp := archsimd.BroadcastUint32x8(xq[r][i4]).AsInt8x32()
						a[r] = a[r].Add(pair.DotProductPairsSaturated(xp).DotProductPairs(ones))
					}
				}
				sc := loadF32x8(srow[jt:])
				for r := 0; r < 4; r++ {
					corr := archsimd.BroadcastInt32x8(8 * gsums[r][g])
					o := out.Data[(r0+r)*cols:]
					f := a[r].Sub(corr).ConvertToFloat32().Mul(sc)
					storeF32x8(loadF32x8(o[jt:]).Add(f), o[jt:])
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
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, group, cols, vecEnd, hi)
	}
}

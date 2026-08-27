//go:build goexperiment.simd && amd64

package tensai

import (
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// 256-bit AVX2 kernels for the MXFP4 layout: the Q4 kernels' nibble
// unpack lands the FP4 codes in quad-layout byte lanes, one VPSHUFB
// expands them to the doubled integer grid, and the rest is the
// grouped-int8 chain — raw 7-bit activations as the unsigned operand,
// the 64x column-sum correction folded per 32-row group.

// mxfp4LUT32 doubles the 16-entry code table across both shuffle lanes.
var mxfp4LUT32 = func() [32]int8 {
	var t [32]int8
	copy(t[:16], mxfp4LUT[:])
	copy(t[16:], mxfp4LUT[:])
	return t
}()

func mxfp4MatvecCols(out []Float, xu []uint8, sx Float, qw []uint8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !simd.HasAVX2 {
		mxfp4MatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, lo, hi)
		return
	}
	quads := len(xu) / 4
	groups := (len(xu) + 31) / 32
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		ones := archsimd.BroadcastInt16x16(1)
		lut := simd.LoadI8x32(mxfp4LUT32[:])
		clear(out[lo:vecEnd])
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile:]
			stab := scale[(jt/q4Tile)*groups*q4Tile:]
			ctab := colSum64[(jt/q4Tile)*groups*q4Tile:]
			d0 := out[jt : jt+8 : jt+8]
			d1 := out[jt+8 : jt+16 : jt+16]
			d2 := out[jt+16 : jt+24 : jt+24]
			d3 := out[jt+24 : jt+32 : jt+32]
			for g := 0; g < groups; g++ {
				ib := g * 8
				ie := min(ib+8, quads)
				var a0, a1, a2, a3 archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					xp := archsimd.BroadcastUint32x8(qxQuad(xu, i4)).AsUint8x32()
					row := tile[i4*2*q4Tile:]
					v := simd.LoadU8x16(row).ExtendToUint16()
					w := lut.PermuteOrZeroGrouped(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsInt8x32())
					a0 = a0.Add(xp.DotProductPairsSaturated(w).DotProductPairs(ones))
					v = simd.LoadU8x16(row[16:]).ExtendToUint16()
					w = lut.PermuteOrZeroGrouped(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsInt8x32())
					a1 = a1.Add(xp.DotProductPairsSaturated(w).DotProductPairs(ones))
					v = simd.LoadU8x16(row[32:]).ExtendToUint16()
					w = lut.PermuteOrZeroGrouped(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsInt8x32())
					a2 = a2.Add(xp.DotProductPairsSaturated(w).DotProductPairs(ones))
					v = simd.LoadU8x16(row[48:]).ExtendToUint16()
					w = lut.PermuteOrZeroGrouped(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsInt8x32())
					a3 = a3.Add(xp.DotProductPairsSaturated(w).DotProductPairs(ones))
				}
				tg := stab[g*q4Tile:]
				cg := ctab[g*q4Tile:]
				f := a0.Sub(simd.LoadI32x8(cg)).ConvertToFloat32().Mul(simd.LoadF32x8(tg))
				simd.StoreF32x8(simd.LoadF32x8(d0).Add(f), d0)
				f = a1.Sub(simd.LoadI32x8(cg[8:])).ConvertToFloat32().Mul(simd.LoadF32x8(tg[8:]))
				simd.StoreF32x8(simd.LoadF32x8(d1).Add(f), d1)
				f = a2.Sub(simd.LoadI32x8(cg[16:])).ConvertToFloat32().Mul(simd.LoadF32x8(tg[16:]))
				simd.StoreF32x8(simd.LoadF32x8(d2).Add(f), d2)
				f = a3.Sub(simd.LoadI32x8(cg[24:])).ConvertToFloat32().Mul(simd.LoadF32x8(tg[24:]))
				simd.StoreF32x8(simd.LoadF32x8(d3).Add(f), d3)
			}
		}
		sxv := archsimd.BroadcastFloat32x8(sx)
		for j := lo; j < vecEnd; j += 8 {
			simd.StoreF32x8(simd.LoadF32x8(out[j:]).Mul(sxv), out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		mxfp4MatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

// mxfp4MatmulRows8 is the eight-row batched form: the nibble unpack and
// the shuffle run once per eight-column step and feed eight broadcast
// activation quads.
func mxfp4MatmulRows8(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []uint8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !simd.HasAVX2 {
		mxfp4MatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + 31) / 32
	xq := make([]uint32, 8*quads)
	for r := 0; r < 8; r++ {
		for i4 := 0; i4 < quads; i4++ {
			xq[i4*8+r] = qxQuad(xus[r], i4)
		}
	}
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		ones := archsimd.BroadcastInt16x16(1)
		lut := simd.LoadI8x32(mxfp4LUT32[:])
		for r := 0; r < 8; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		o0 := out.Data[r0*cols:]
		o1 := out.Data[(r0+1)*cols:]
		o2 := out.Data[(r0+2)*cols:]
		o3 := out.Data[(r0+3)*cols:]
		o4 := out.Data[(r0+4)*cols:]
		o5 := out.Data[(r0+5)*cols:]
		o6 := out.Data[(r0+6)*cols:]
		o7 := out.Data[(r0+7)*cols:]
		for jt := lo; jt < vecEnd; jt += 8 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile+(jt%q4Tile)*2:]
			stab := scale[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
			ctab := colSum64[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
			d0 := o0[jt : jt+8 : jt+8]
			d1 := o1[jt : jt+8 : jt+8]
			d2 := o2[jt : jt+8 : jt+8]
			d3 := o3[jt : jt+8 : jt+8]
			d4 := o4[jt : jt+8 : jt+8]
			d5 := o5[jt : jt+8 : jt+8]
			d6 := o6[jt : jt+8 : jt+8]
			d7 := o7[jt : jt+8 : jt+8]
			for g := 0; g < groups; g++ {
				ib := g * 8
				ie := min(ib+8, quads)
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					v := simd.LoadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
					w := lut.PermuteOrZeroGrouped(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsInt8x32())
					xf := xq[i4*8 : i4*8+8]
					a0 = a0.Add(archsimd.BroadcastUint32x8(xf[0]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a1 = a1.Add(archsimd.BroadcastUint32x8(xf[1]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a2 = a2.Add(archsimd.BroadcastUint32x8(xf[2]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a3 = a3.Add(archsimd.BroadcastUint32x8(xf[3]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a4 = a4.Add(archsimd.BroadcastUint32x8(xf[4]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a5 = a5.Add(archsimd.BroadcastUint32x8(xf[5]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a6 = a6.Add(archsimd.BroadcastUint32x8(xf[6]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
					a7 = a7.Add(archsimd.BroadcastUint32x8(xf[7]).AsUint8x32().DotProductPairsSaturated(w).DotProductPairs(ones))
				}
				cs := simd.LoadI32x8(ctab[g*q4Tile:])
				sc := simd.LoadF32x8(stab[g*q4Tile:])
				simd.StoreF32x8(simd.LoadF32x8(d0).Add(a0.Sub(cs).ConvertToFloat32().Mul(sc)), d0)
				simd.StoreF32x8(simd.LoadF32x8(d1).Add(a1.Sub(cs).ConvertToFloat32().Mul(sc)), d1)
				simd.StoreF32x8(simd.LoadF32x8(d2).Add(a2.Sub(cs).ConvertToFloat32().Mul(sc)), d2)
				simd.StoreF32x8(simd.LoadF32x8(d3).Add(a3.Sub(cs).ConvertToFloat32().Mul(sc)), d3)
				simd.StoreF32x8(simd.LoadF32x8(d4).Add(a4.Sub(cs).ConvertToFloat32().Mul(sc)), d4)
				simd.StoreF32x8(simd.LoadF32x8(d5).Add(a5.Sub(cs).ConvertToFloat32().Mul(sc)), d5)
				simd.StoreF32x8(simd.LoadF32x8(d6).Add(a6.Sub(cs).ConvertToFloat32().Mul(sc)), d6)
				simd.StoreF32x8(simd.LoadF32x8(d7).Add(a7.Sub(cs).ConvertToFloat32().Mul(sc)), d7)
			}
		}
		for r := 0; r < 8; r++ {
			sxv := archsimd.BroadcastFloat32x8(sxs[r])
			o := out.Data[(r0+r)*cols:]
			for j := lo; j < vecEnd; j += 8 {
				simd.StoreF32x8(simd.LoadF32x8(o[j:]).Mul(sxv), o[j:])
			}
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		mxfp4MatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

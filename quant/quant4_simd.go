//go:build goexperiment.simd && amd64

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

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

func q4matvecCols(out []tensai.Float, xu []uint8, xq []uint32, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	if !simd.HasAVX2 {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, lo, hi)
		return
	}
	quads := len(xq)
	groups := len(gsum)
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		gsumF := make([]tensai.Float, groups)
		gsum8 := make([]int32, groups)
		for i, v := range gsum {
			gsumF[i] = tensai.Float(v)
			gsum8[i] = 8 * v
		}
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		mMin := archsimd.BroadcastUint32x8(0xFFFF0000)
		ones := archsimd.BroadcastInt16x16(1)
		clear(out[lo:vecEnd])
		// Tiles outermost: the nibble walk and the tile-major table walk
		// both advance strictly sequentially per worker.
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile:]
			d0 := out[jt : jt+8 : jt+8]
			d1 := out[jt+8 : jt+16 : jt+16]
			d2 := out[jt+16 : jt+24 : jt+24]
			d3 := out[jt+24 : jt+32 : jt+32]
			if sm != nil {
				tab := sm[(jt/q4Tile)*groups*q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					gsf := archsimd.BroadcastFloat32x8(gsumF[g])
					var a0, a1, a2, a3 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						xp := archsimd.BroadcastUint32x8(xq[i4]).AsInt8x32()
						row := tile[i4*2*q4Tile:]
						v := simd.LoadU8x16(row).ExtendToUint16()
						a0 = a0.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = simd.LoadU8x16(row[16:]).ExtendToUint16()
						a1 = a1.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = simd.LoadU8x16(row[32:]).ExtendToUint16()
						a2 = a2.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = simd.LoadU8x16(row[48:]).ExtendToUint16()
						a3 = a3.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
					}
					tg := tab[g*q4Tile:]
					u := simd.LoadU32x8(tg)
					f := a0.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					simd.StoreF32x8(simd.LoadF32x8(d0).Add(f), d0)
					u = simd.LoadU32x8(tg[8:])
					f = a1.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					simd.StoreF32x8(simd.LoadF32x8(d1).Add(f), d1)
					u = simd.LoadU32x8(tg[16:])
					f = a2.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					simd.StoreF32x8(simd.LoadF32x8(d2).Add(f), d2)
					u = simd.LoadU32x8(tg[24:])
					f = a3.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					simd.StoreF32x8(simd.LoadF32x8(d3).Add(f), d3)
				}
			} else {
				tab := scale[(jt/q4Tile)*groups*q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					corr := archsimd.BroadcastInt32x8(gsum8[g])
					var a0, a1, a2, a3 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						xp := archsimd.BroadcastUint32x8(xq[i4]).AsInt8x32()
						row := tile[i4*2*q4Tile:]
						v := simd.LoadU8x16(row).ExtendToUint16()
						a0 = a0.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = simd.LoadU8x16(row[16:]).ExtendToUint16()
						a1 = a1.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = simd.LoadU8x16(row[32:]).ExtendToUint16()
						a2 = a2.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = simd.LoadU8x16(row[48:]).ExtendToUint16()
						a3 = a3.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
					}
					tg := tab[g*q4Tile:]
					simd.StoreF32x8(simd.LoadF32x8(d0).Add(a0.Sub(corr).ConvertToFloat32().Mul(simd.LoadF32x8(tg))), d0)
					simd.StoreF32x8(simd.LoadF32x8(d1).Add(a1.Sub(corr).ConvertToFloat32().Mul(simd.LoadF32x8(tg[8:]))), d1)
					simd.StoreF32x8(simd.LoadF32x8(d2).Add(a2.Sub(corr).ConvertToFloat32().Mul(simd.LoadF32x8(tg[16:]))), d2)
					simd.StoreF32x8(simd.LoadF32x8(d3).Add(a3.Sub(corr).ConvertToFloat32().Mul(simd.LoadF32x8(tg[24:]))), d3)
				}
			}
		}
		sxv := archsimd.BroadcastFloat32x8(sx)
		for j := lo; j < vecEnd; j += 8 {
			simd.StoreF32x8(simd.LoadF32x8(out[j:]).Mul(sxv), out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

// q4matmulCols4 is the four-row batched form: per eight-column step each
// sixteen-byte nibble load unpacks once and feeds four broadcast
// activation quads, amortizing the nibble traffic fourfold; steps stay
// outermost so both streams advance sequentially.
func q4matmulCols4(out *tensai.Matrix, xus [][]uint8, xqs [][]uint32, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	if !simd.HasAVX2 {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	groups := len(gsums[0])
	xq0, xq1, xq2, xq3 := xqs[0], xqs[1], xqs[2], xqs[3]
	// Flat copies keep the folds free of nested-slice loads: one indexed
	// read per row instead of a header chase and two bounds checks.
	gsf := make([]tensai.Float, 4*groups)
	gsc := make([]int32, 4*groups)
	for r := 0; r < 4; r++ {
		for g, v := range gsums[r] {
			gsf[4*g+r] = tensai.Float(v)
			gsc[4*g+r] = 8 * v
		}
	}
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		mMin := archsimd.BroadcastUint32x8(0xFFFF0000)
		ones := archsimd.BroadcastInt16x16(1)
		o0 := out.Data[r0*cols:]
		o1 := out.Data[(r0+1)*cols:]
		o2 := out.Data[(r0+2)*cols:]
		o3 := out.Data[(r0+3)*cols:]
		for r := 0; r < 4; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		for jt := lo; jt < vecEnd; jt += 8 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile+(jt%q4Tile)*2:]
			d0 := o0[jt : jt+8 : jt+8]
			d1 := o1[jt : jt+8 : jt+8]
			d2 := o2[jt : jt+8 : jt+8]
			d3 := o3[jt : jt+8 : jt+8]
			if sm != nil {
				tab := sm[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					var a0, a1, a2, a3 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						v := simd.LoadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						a0 = a0.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[i4]).AsInt8x32()).DotProductPairs(ones))
						a1 = a1.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[i4]).AsInt8x32()).DotProductPairs(ones))
						a2 = a2.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[i4]).AsInt8x32()).DotProductPairs(ones))
						a3 = a3.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[i4]).AsInt8x32()).DotProductPairs(ones))
					}
					u := simd.LoadU32x8(tab[g*q4Tile:])
					sc := u.ShiftAllLeft(16).AsFloat32x8()
					mn := u.And(mMin).AsFloat32x8()
					gr := gsf[4*g : 4*g+4 : 4*g+4]
					f := a0.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[0]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d0).Add(f), d0)
					f = a1.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[1]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d1).Add(f), d1)
					f = a2.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[2]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d2).Add(f), d2)
					f = a3.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[3]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d3).Add(f), d3)
				}
			} else {
				tab := scale[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					var a0, a1, a2, a3 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						v := simd.LoadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						a0 = a0.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[i4]).AsInt8x32()).DotProductPairs(ones))
						a1 = a1.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[i4]).AsInt8x32()).DotProductPairs(ones))
						a2 = a2.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[i4]).AsInt8x32()).DotProductPairs(ones))
						a3 = a3.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[i4]).AsInt8x32()).DotProductPairs(ones))
					}
					sc := simd.LoadF32x8(tab[g*q4Tile:])
					gr := gsc[4*g : 4*g+4 : 4*g+4]
					f := a0.Sub(archsimd.BroadcastInt32x8(gr[0])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d0).Add(f), d0)
					f = a1.Sub(archsimd.BroadcastInt32x8(gr[1])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d1).Add(f), d1)
					f = a2.Sub(archsimd.BroadcastInt32x8(gr[2])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d2).Add(f), d2)
					f = a3.Sub(archsimd.BroadcastInt32x8(gr[3])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d3).Add(f), d3)
				}
			}
		}
		for r := 0; r < 4; r++ {
			sxv := archsimd.BroadcastFloat32x8(sxs[r])
			o := out.Data[(r0+r)*cols:]
			for j := lo; j < vecEnd; j += 8 {
				simd.StoreF32x8(simd.LoadF32x8(o[j:]).Mul(sxv), o[j:])
			}
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

// q4matmulCols8 is the eight-row batched body: one nibble unpack and one
// scale-table fold feed eight output rows, halving both against the
// four-row kernel while the multiplies stay one broadcast and two madds
// per row. The eight Int32x8 accumulators plus the unpack temporaries
// still fit the sixteen-register file — named variables only, flat
// tables, hoisted row slices (the spill traps notes on the four-row
// kernel apply verbatim).
func q4matmulCols8(out *tensai.Matrix, xus [][]uint8, xqs [][]uint32, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	if !simd.HasAVX2 {
		q4matmulCols4Generic(out, xus[:4], sxs[:4], gsums[:4], r0, qw, scale, sm, group, cols, lo, hi)
		q4matmulCols4Generic(out, xus[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, sm, group, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	groups := len(gsums[0])
	xq0, xq1, xq2, xq3 := xqs[0], xqs[1], xqs[2], xqs[3]
	xq4, xq5, xq6, xq7 := xqs[4], xqs[5], xqs[6], xqs[7]
	gsf := make([]tensai.Float, 8*groups)
	gsc := make([]int32, 8*groups)
	for r := 0; r < 8; r++ {
		for g, v := range gsums[r] {
			gsf[8*g+r] = tensai.Float(v)
			gsc[8*g+r] = 8 * v
		}
	}
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		mLo := archsimd.BroadcastUint16x16(0x000F)
		mHi := archsimd.BroadcastUint16x16(0x0F00)
		ones := archsimd.BroadcastInt16x16(1)
		o0 := out.Data[r0*cols:]
		o1 := out.Data[(r0+1)*cols:]
		o2 := out.Data[(r0+2)*cols:]
		o3 := out.Data[(r0+3)*cols:]
		o4 := out.Data[(r0+4)*cols:]
		o5 := out.Data[(r0+5)*cols:]
		o6 := out.Data[(r0+6)*cols:]
		o7 := out.Data[(r0+7)*cols:]
		for r := 0; r < 8; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		for jt := lo; jt < vecEnd; jt += 8 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile+(jt%q4Tile)*2:]
			d0 := o0[jt : jt+8 : jt+8]
			d1 := o1[jt : jt+8 : jt+8]
			d2 := o2[jt : jt+8 : jt+8]
			d3 := o3[jt : jt+8 : jt+8]
			d4 := o4[jt : jt+8 : jt+8]
			d5 := o5[jt : jt+8 : jt+8]
			d6 := o6[jt : jt+8 : jt+8]
			d7 := o7[jt : jt+8 : jt+8]
			if sm != nil {
				tab := sm[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						v := simd.LoadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						a0 = a0.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[i4]).AsInt8x32()).DotProductPairs(ones))
						a1 = a1.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[i4]).AsInt8x32()).DotProductPairs(ones))
						a2 = a2.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[i4]).AsInt8x32()).DotProductPairs(ones))
						a3 = a3.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[i4]).AsInt8x32()).DotProductPairs(ones))
						a4 = a4.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq4[i4]).AsInt8x32()).DotProductPairs(ones))
						a5 = a5.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq5[i4]).AsInt8x32()).DotProductPairs(ones))
						a6 = a6.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq6[i4]).AsInt8x32()).DotProductPairs(ones))
						a7 = a7.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq7[i4]).AsInt8x32()).DotProductPairs(ones))
					}
					u := simd.LoadU32x8(tab[g*q4Tile:])
					sc := u.ShiftAllLeft(16).AsFloat32x8()
					mn := u.And(archsimd.BroadcastUint32x8(0xFFFF0000)).AsFloat32x8()
					gr := gsf[8*g : 8*g+8 : 8*g+8]
					f := a0.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[0]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d0).Add(f), d0)
					f = a1.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[1]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d1).Add(f), d1)
					f = a2.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[2]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d2).Add(f), d2)
					f = a3.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[3]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d3).Add(f), d3)
					f = a4.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[4]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d4).Add(f), d4)
					f = a5.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[5]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d5).Add(f), d5)
					f = a6.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[6]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d6).Add(f), d6)
					f = a7.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[7]).Mul(mn))
					simd.StoreF32x8(simd.LoadF32x8(d7).Add(f), d7)
				}
			} else {
				tab := scale[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						v := simd.LoadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						a0 = a0.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[i4]).AsInt8x32()).DotProductPairs(ones))
						a1 = a1.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[i4]).AsInt8x32()).DotProductPairs(ones))
						a2 = a2.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[i4]).AsInt8x32()).DotProductPairs(ones))
						a3 = a3.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[i4]).AsInt8x32()).DotProductPairs(ones))
						a4 = a4.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq4[i4]).AsInt8x32()).DotProductPairs(ones))
						a5 = a5.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq5[i4]).AsInt8x32()).DotProductPairs(ones))
						a6 = a6.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq6[i4]).AsInt8x32()).DotProductPairs(ones))
						a7 = a7.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq7[i4]).AsInt8x32()).DotProductPairs(ones))
					}
					sc := simd.LoadF32x8(tab[g*q4Tile:])
					gr := gsc[8*g : 8*g+8 : 8*g+8]
					f := a0.Sub(archsimd.BroadcastInt32x8(gr[0])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d0).Add(f), d0)
					f = a1.Sub(archsimd.BroadcastInt32x8(gr[1])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d1).Add(f), d1)
					f = a2.Sub(archsimd.BroadcastInt32x8(gr[2])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d2).Add(f), d2)
					f = a3.Sub(archsimd.BroadcastInt32x8(gr[3])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d3).Add(f), d3)
					f = a4.Sub(archsimd.BroadcastInt32x8(gr[4])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d4).Add(f), d4)
					f = a5.Sub(archsimd.BroadcastInt32x8(gr[5])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d5).Add(f), d5)
					f = a6.Sub(archsimd.BroadcastInt32x8(gr[6])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d6).Add(f), d6)
					f = a7.Sub(archsimd.BroadcastInt32x8(gr[7])).ConvertToFloat32().Mul(sc)
					simd.StoreF32x8(simd.LoadF32x8(d7).Add(f), d7)
				}
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
		q4matmulCols4Generic(out, xus[:4], sxs[:4], gsums[:4], r0, qw, scale, sm, group, cols, vecEnd, hi)
		q4matmulCols4Generic(out, xus[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

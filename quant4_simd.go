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

func q4matvecCols(out []Float, xu []uint8, xq []uint32, sx Float, gsum []int32, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	if !hasAVX2 {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, lo, hi)
		return
	}
	quads := len(xq)
	groups := len(gsum)
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		gsumF := make([]Float, groups)
		gsum8 := make([]int32, groups)
		for i, v := range gsum {
			gsumF[i] = Float(v)
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
						v := loadU8x16(row).ExtendToUint16()
						a0 = a0.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = loadU8x16(row[16:]).ExtendToUint16()
						a1 = a1.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = loadU8x16(row[32:]).ExtendToUint16()
						a2 = a2.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = loadU8x16(row[48:]).ExtendToUint16()
						a3 = a3.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
					}
					tg := tab[g*q4Tile:]
					u := loadU32x8(tg)
					f := a0.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					storeF32x8(loadF32x8(d0).Add(f), d0)
					u = loadU32x8(tg[8:])
					f = a1.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					storeF32x8(loadF32x8(d1).Add(f), d1)
					u = loadU32x8(tg[16:])
					f = a2.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					storeF32x8(loadF32x8(d2).Add(f), d2)
					u = loadU32x8(tg[24:])
					f = a3.ConvertToFloat32().Mul(u.ShiftAllLeft(16).AsFloat32x8()).Sub(gsf.Mul(u.And(mMin).AsFloat32x8()))
					storeF32x8(loadF32x8(d3).Add(f), d3)
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
						v := loadU8x16(row).ExtendToUint16()
						a0 = a0.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = loadU8x16(row[16:]).ExtendToUint16()
						a1 = a1.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = loadU8x16(row[32:]).ExtendToUint16()
						a2 = a2.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
						v = loadU8x16(row[48:]).ExtendToUint16()
						a3 = a3.Add(v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32().DotProductPairsSaturated(xp).DotProductPairs(ones))
					}
					tg := tab[g*q4Tile:]
					storeF32x8(loadF32x8(d0).Add(a0.Sub(corr).ConvertToFloat32().Mul(loadF32x8(tg))), d0)
					storeF32x8(loadF32x8(d1).Add(a1.Sub(corr).ConvertToFloat32().Mul(loadF32x8(tg[8:]))), d1)
					storeF32x8(loadF32x8(d2).Add(a2.Sub(corr).ConvertToFloat32().Mul(loadF32x8(tg[16:]))), d2)
					storeF32x8(loadF32x8(d3).Add(a3.Sub(corr).ConvertToFloat32().Mul(loadF32x8(tg[24:]))), d3)
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
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

// q4matmulCols4 is the four-row batched form: per eight-column step each
// sixteen-byte nibble load unpacks once and feeds four broadcast
// activation quads, amortizing the nibble traffic fourfold; steps stay
// outermost so both streams advance sequentially.
func q4matmulCols4(out *Matrix, xus [][]uint8, sxs []Float, gsums [][]int32, r0 int, qw []uint8, scale []Float, sm []uint32, group, cols, lo, hi int) {
	if !hasAVX2 {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	groups := len(gsums[0])
	xq := make([]uint32, 4*quads)
	for r := 0; r < 4; r++ {
		for i4 := 0; i4 < quads; i4++ {
			xq[i4*4+r] = q4xQuad(xus[r], i4)
		}
	}
	// Flat copies keep the folds free of nested-slice loads: one indexed
	// read per row instead of a header chase and two bounds checks.
	gsf := make([]Float, 4*groups)
	gsc := make([]int32, 4*groups)
	for r := 0; r < 4; r++ {
		for g, v := range gsums[r] {
			gsf[4*g+r] = Float(v)
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
						v := loadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						xf := xq[i4*4 : i4*4+4]
						a0 = a0.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[0]).AsInt8x32()).DotProductPairs(ones))
						a1 = a1.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[1]).AsInt8x32()).DotProductPairs(ones))
						a2 = a2.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[2]).AsInt8x32()).DotProductPairs(ones))
						a3 = a3.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[3]).AsInt8x32()).DotProductPairs(ones))
					}
					u := loadU32x8(tab[g*q4Tile:])
					sc := u.ShiftAllLeft(16).AsFloat32x8()
					mn := u.And(mMin).AsFloat32x8()
					gr := gsf[4*g : 4*g+4 : 4*g+4]
					f := a0.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[0]).Mul(mn))
					storeF32x8(loadF32x8(d0).Add(f), d0)
					f = a1.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[1]).Mul(mn))
					storeF32x8(loadF32x8(d1).Add(f), d1)
					f = a2.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[2]).Mul(mn))
					storeF32x8(loadF32x8(d2).Add(f), d2)
					f = a3.ConvertToFloat32().Mul(sc).Sub(archsimd.BroadcastFloat32x8(gr[3]).Mul(mn))
					storeF32x8(loadF32x8(d3).Add(f), d3)
				}
			} else {
				tab := scale[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
				for g := 0; g < groups; g++ {
					ib := g * group / 4
					ie := min(ib+group/4, quads)
					var a0, a1, a2, a3 archsimd.Int32x8
					for i4 := ib; i4 < ie; i4++ {
						v := loadU8x16(tile[i4*2*q4Tile:]).ExtendToUint16()
						pair := v.And(mLo).Or(v.ShiftAllLeft(4).And(mHi)).AsUint8x32()
						xf := xq[i4*4 : i4*4+4]
						a0 = a0.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[0]).AsInt8x32()).DotProductPairs(ones))
						a1 = a1.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[1]).AsInt8x32()).DotProductPairs(ones))
						a2 = a2.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[2]).AsInt8x32()).DotProductPairs(ones))
						a3 = a3.Add(pair.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xf[3]).AsInt8x32()).DotProductPairs(ones))
					}
					sc := loadF32x8(tab[g*q4Tile:])
					gr := gsc[4*g : 4*g+4 : 4*g+4]
					f := a0.Sub(archsimd.BroadcastInt32x8(gr[0])).ConvertToFloat32().Mul(sc)
					storeF32x8(loadF32x8(d0).Add(f), d0)
					f = a1.Sub(archsimd.BroadcastInt32x8(gr[1])).ConvertToFloat32().Mul(sc)
					storeF32x8(loadF32x8(d1).Add(f), d1)
					f = a2.Sub(archsimd.BroadcastInt32x8(gr[2])).ConvertToFloat32().Mul(sc)
					storeF32x8(loadF32x8(d2).Add(f), d2)
					f = a3.Sub(archsimd.BroadcastInt32x8(gr[3])).ConvertToFloat32().Mul(sc)
					storeF32x8(loadF32x8(d3).Add(f), d3)
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
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

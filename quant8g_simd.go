//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// 256-bit AVX2 kernels for the grouped int8 layout: identical to the
// QMatrix quad kernels — one 32-byte load takes eight columns four rows
// deep through the u8 x s8 pairwise multiply-add and the widening i16
// pair-add — except the accumulator folds with its (group, column) scale
// and 64x column-sum correction at each 32-row group boundary, Q4Matrix
// style.

func q8gMatvecCols(out []Float, xu []uint8, sx Float, qw []int8, scale []Float, colSum64 []int32, group, cols, lo, hi int) {
	if !hasAVX2 {
		q8gMatvecColsGeneric(out, xu, sx, qw, scale, colSum64, group, cols, lo, hi)
		return
	}
	quads := len(xu) / 4
	groups := (len(xu) + group - 1) / group
	ones := archsimd.BroadcastInt16x16(1)
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		clear(out[lo:vecEnd])
		// Tiles outermost: weight and table streams both advance
		// strictly sequentially per worker.
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/q4Tile)*quads*4*q4Tile:]
			stab := scale[(jt/q4Tile)*groups*q4Tile:]
			ctab := colSum64[(jt/q4Tile)*groups*q4Tile:]
			d0 := out[jt : jt+8 : jt+8]
			d1 := out[jt+8 : jt+16 : jt+16]
			d2 := out[jt+16 : jt+24 : jt+24]
			d3 := out[jt+24 : jt+32 : jt+32]
			for g := 0; g < groups; g++ {
				ib := g * group / 4
				ie := min(ib+group/4, quads)
				var a0, a1, a2, a3 archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					xp := archsimd.BroadcastUint32x8(qxQuad(xu, i4)).AsUint8x32()
					row := tile[i4*4*q4Tile:]
					a0 = a0.Add(xp.DotProductPairsSaturated(loadI8x32(row)).DotProductPairs(ones))
					a1 = a1.Add(xp.DotProductPairsSaturated(loadI8x32(row[32:])).DotProductPairs(ones))
					a2 = a2.Add(xp.DotProductPairsSaturated(loadI8x32(row[64:])).DotProductPairs(ones))
					a3 = a3.Add(xp.DotProductPairsSaturated(loadI8x32(row[96:])).DotProductPairs(ones))
				}
				sg := stab[g*q4Tile:]
				cg := ctab[g*q4Tile:]
				f := a0.Sub(loadI32x8(cg)).ConvertToFloat32().Mul(loadF32x8(sg))
				storeF32x8(loadF32x8(d0).Add(f), d0)
				f = a1.Sub(loadI32x8(cg[8:])).ConvertToFloat32().Mul(loadF32x8(sg[8:]))
				storeF32x8(loadF32x8(d1).Add(f), d1)
				f = a2.Sub(loadI32x8(cg[16:])).ConvertToFloat32().Mul(loadF32x8(sg[16:]))
				storeF32x8(loadF32x8(d2).Add(f), d2)
				f = a3.Sub(loadI32x8(cg[24:])).ConvertToFloat32().Mul(loadF32x8(sg[24:]))
				storeF32x8(loadF32x8(d3).Add(f), d3)
			}
		}
		sxv := archsimd.BroadcastFloat32x8(sx)
		for j := lo; j < vecEnd; j += 8 {
			storeF32x8(loadF32x8(out[j:]).Mul(sxv), out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q8gMatvecColsGeneric(out, xu, sx, qw, scale, colSum64, group, cols, vecEnd, hi)
	}
}

// q8gMatmulRows8 is the eight-row batched form: per eight-column step
// each 32-byte weight load feeds eight broadcast activation quads, with
// steps outermost so both streams advance sequentially.
func q8gMatmulRows8(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, group, cols, lo, hi int) {
	if !hasAVX2 {
		q8gMatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, group, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + group - 1) / group
	xq := make([]uint32, 8*quads)
	for r := 0; r < 8; r++ {
		for i4 := 0; i4 < quads; i4++ {
			xq[i4*8+r] = qxQuad(xus[r], i4)
		}
	}
	ones := archsimd.BroadcastInt16x16(1)
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
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
			tile := qw[(jt/q4Tile)*quads*4*q4Tile+(jt%q4Tile)*4:]
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
				ib := g * group / 4
				ie := min(ib+group/4, quads)
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					w := loadI8x32(tile[i4*4*q4Tile:])
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
				cs := loadI32x8(ctab[g*q4Tile:])
				sc := loadF32x8(stab[g*q4Tile:])
				storeF32x8(loadF32x8(d0).Add(a0.Sub(cs).ConvertToFloat32().Mul(sc)), d0)
				storeF32x8(loadF32x8(d1).Add(a1.Sub(cs).ConvertToFloat32().Mul(sc)), d1)
				storeF32x8(loadF32x8(d2).Add(a2.Sub(cs).ConvertToFloat32().Mul(sc)), d2)
				storeF32x8(loadF32x8(d3).Add(a3.Sub(cs).ConvertToFloat32().Mul(sc)), d3)
				storeF32x8(loadF32x8(d4).Add(a4.Sub(cs).ConvertToFloat32().Mul(sc)), d4)
				storeF32x8(loadF32x8(d5).Add(a5.Sub(cs).ConvertToFloat32().Mul(sc)), d5)
				storeF32x8(loadF32x8(d6).Add(a6.Sub(cs).ConvertToFloat32().Mul(sc)), d6)
				storeF32x8(loadF32x8(d7).Add(a7.Sub(cs).ConvertToFloat32().Mul(sc)), d7)
			}
		}
		for r := 0; r < 8; r++ {
			sxv := archsimd.BroadcastFloat32x8(sxs[r])
			o := out.Data[(r0+r)*cols:]
			for j := lo; j < vecEnd; j += 8 {
				storeF32x8(loadF32x8(o[j:]).Mul(sxv), o[j:])
			}
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		q8gMatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, group, cols, vecEnd, hi)
	}
}

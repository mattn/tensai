//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// 256-bit AVX2 kernels for the grouped int8 layout: identical to the
// QMatrix quad kernels — one 32-byte load takes eight columns four rows
// deep through the u8 x s8 pairwise multiply-add and the widening i16
// pair-add — except the accumulator folds with its (group, column) scale
// and 64x column-sum correction at each 32-row group boundary, Q4Matrix
// style.

func q8gMatvecCols(out []Float, xu []uint8, sx Float, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !hasAVX2 {
		q8gMatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, lo, hi)
		return
	}
	quads := len(xu) / 4
	groups := (len(xu) + q8Group - 1) / q8Group
	ones := archsimd.BroadcastInt16x16(1)
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		clear(out[lo:vecEnd])
		for g := 0; g < groups; g++ {
			ib := g * q8Group / 4
			ie := min(ib+q8Group/4, quads)
			srow := scale[g*cols:]
			csrow := colSum64[g*cols:]
			for jt := lo; jt < vecEnd; jt += 32 {
				var a [4]archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					xp := archsimd.BroadcastUint32x8(qxQuad(xu, i4)).AsUint8x32()
					row := qw[i4*4*cols+4*jt:]
					a[0] = a[0].Add(xp.DotProductPairsSaturated(loadI8x32(row)).DotProductPairs(ones))
					a[1] = a[1].Add(xp.DotProductPairsSaturated(loadI8x32(row[32:])).DotProductPairs(ones))
					a[2] = a[2].Add(xp.DotProductPairsSaturated(loadI8x32(row[64:])).DotProductPairs(ones))
					a[3] = a[3].Add(xp.DotProductPairsSaturated(loadI8x32(row[96:])).DotProductPairs(ones))
				}
				for k := 0; k < 4; k++ {
					j := jt + 8*k
					f := a[k].Sub(loadI32x8(csrow[j:])).ConvertToFloat32().Mul(loadF32x8(srow[j:]))
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
		q8gMatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

// q8gMatmulRows8 is the eight-row batched form: per eight-column tile each
// 32-byte weight load feeds eight broadcast activation quads.
func q8gMatmulRows8(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !hasAVX2 {
		q8gMatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + q8Group - 1) / q8Group
	var xq [8][]uint32
	for r := 0; r < 8; r++ {
		xq[r] = make([]uint32, quads)
		for i4 := 0; i4 < quads; i4++ {
			xq[r][i4] = qxQuad(xus[r], i4)
		}
	}
	ones := archsimd.BroadcastInt16x16(1)
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		for r := 0; r < 8; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		for g := 0; g < groups; g++ {
			ib := g * q8Group / 4
			ie := min(ib+q8Group/4, quads)
			srow := scale[g*cols:]
			csrow := colSum64[g*cols:]
			for jt := lo; jt < vecEnd; jt += 8 {
				var a [8]archsimd.Int32x8
				for i4 := ib; i4 < ie; i4++ {
					w := loadI8x32(qw[i4*4*cols+4*jt:])
					for r := 0; r < 8; r++ {
						xp := archsimd.BroadcastUint32x8(xq[r][i4]).AsUint8x32()
						a[r] = a[r].Add(xp.DotProductPairsSaturated(w).DotProductPairs(ones))
					}
				}
				cs := loadI32x8(csrow[jt:])
				sc := loadF32x8(srow[jt:])
				for r := 0; r < 8; r++ {
					o := out.Data[(r0+r)*cols:]
					f := a[r].Sub(cs).ConvertToFloat32().Mul(sc)
					storeF32x8(loadF32x8(o[jt:]).Add(f), o[jt:])
				}
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
		q8gMatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

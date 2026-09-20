//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// 128-bit NEON kernels for the grouped int8 layout: the QMatrix quad
// kernels of quant_neon.go, with the accumulator folded against its
// (group, column) scale at each group boundary, Q4Matrix style. The
// matvec keeps the activations signed, so as in quant_neon.go the
// column-sum correction is already inside the integer sum and colSum64 is
// never read; the batched form spreads the offset-binary quads the caller
// packed and folds the correction back out per group, as amd64 does.
//
// The group fold is a fused multiply-add. The portable body writes
// `out += float(acc) * scale`, which the arm64 compiler fuses, and the
// batched form must match the matvec bit for bit across the two paths.

func q8gMatvecCols(out []tensai.Float, xu []uint8, sx tensai.Float, qw []int8, scale []tensai.Float, colSum64 []int32, group, cols, lo, hi int) {
	quads := len(xu) / 4
	groups := (len(xu) + group - 1) / group
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		// Tiles outermost: weight and scale streams both advance
		// strictly sequentially per worker.
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/q4Tile)*quads*4*q4Tile:]
			stab := scale[(jt/q4Tile)*groups*q4Tile:]
			var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Float32x4
			for g := 0; g < groups; g++ {
				ib := g * group / 4
				ie := min(ib+group/4, quads)
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
				for i4 := ib; i4 < ie; i4++ {
					xv := qxSigned8(xu, i4)
					row := tile[i4*4*q4Tile : i4*4*q4Tile+4*q4Tile]
					a0 = a0.Add(quadCols(row, xv))
					a1 = a1.Add(quadCols(row[16:], xv))
					a2 = a2.Add(quadCols(row[32:], xv))
					a3 = a3.Add(quadCols(row[48:], xv))
					a4 = a4.Add(quadCols(row[64:], xv))
					a5 = a5.Add(quadCols(row[80:], xv))
					a6 = a6.Add(quadCols(row[96:], xv))
					a7 = a7.Add(quadCols(row[112:], xv))
				}
				sg := stab[g*q4Tile : g*q4Tile+q4Tile]
				o0 = a0.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg), o0)
				o1 = a1.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[4:]), o1)
				o2 = a2.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[8:]), o2)
				o3 = a3.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[12:]), o3)
				o4 = a4.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[16:]), o4)
				o5 = a5.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[20:]), o5)
				o6 = a6.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[24:]), o6)
				o7 = a7.ConvertToFloat32().MulAdd(simd.LoadF32x4(sg[28:]), o7)
			}
			sxv := archsimd.BroadcastFloat32x4(sx)
			o := out[jt : jt+32 : jt+32]
			simd.StoreF32x4(o0.Mul(sxv), o)
			simd.StoreF32x4(o1.Mul(sxv), o[4:])
			simd.StoreF32x4(o2.Mul(sxv), o[8:])
			simd.StoreF32x4(o3.Mul(sxv), o[12:])
			simd.StoreF32x4(o4.Mul(sxv), o[16:])
			simd.StoreF32x4(o5.Mul(sxv), o[20:])
			simd.StoreF32x4(o6.Mul(sxv), o[24:])
			simd.StoreF32x4(o7.Mul(sxv), o[28:])
		}
	}
	if vecEnd < hi {
		q8gMatvecColsGeneric(out, xu, sx, qw, scale, colSum64, group, cols, vecEnd, hi)
	}
}

// q8gMatmulRows8 is the eight-row batched form: per four-column step
// each 16-byte weight load feeds eight spread activation quads, with
// steps outermost so the weight and table streams advance sequentially.
func q8gMatmulRows8(out *tensai.Matrix, xus [][]uint8, xq []uint32, sxs []tensai.Float, r0 int, qw []int8, scale []tensai.Float, colSum64 []int32, group, cols, lo, hi int) {
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + group - 1) / group
	vecEnd := lo + ((hi - lo) &^ 3)
	if vecEnd > lo {
		for jt := lo; jt < vecEnd; jt += 4 {
			tile := qw[(jt/q4Tile)*quads*4*q4Tile+(jt%q4Tile)*4:]
			stab := scale[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
			ctab := colSum64[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
			// Eight named accumulators per row block: an array of SIMD
			// values would live on the stack and turn every multiply-add
			// into a load-op-store round trip.
			var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Float32x4
			for g := 0; g < groups; g++ {
				ib := g * group / 4
				ie := min(ib+group/4, quads)
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
				for i4 := ib; i4 < ie; i4++ {
					w := simd.LoadI8x16(tile[i4*4*q4Tile:])
					wh := w.HiToLo()
					xf := xq[i4*8 : i4*8+8 : i4*8+8]
					a0 = a0.Add(qdot4Hi(qxSpread(xf[0]), w, wh))
					a1 = a1.Add(qdot4Hi(qxSpread(xf[1]), w, wh))
					a2 = a2.Add(qdot4Hi(qxSpread(xf[2]), w, wh))
					a3 = a3.Add(qdot4Hi(qxSpread(xf[3]), w, wh))
					a4 = a4.Add(qdot4Hi(qxSpread(xf[4]), w, wh))
					a5 = a5.Add(qdot4Hi(qxSpread(xf[5]), w, wh))
					a6 = a6.Add(qdot4Hi(qxSpread(xf[6]), w, wh))
					a7 = a7.Add(qdot4Hi(qxSpread(xf[7]), w, wh))
				}
				cs := simd.LoadI32x4(ctab[g*q4Tile:])
				sc := simd.LoadF32x4(stab[g*q4Tile:])
				o0 = a0.Sub(cs).ConvertToFloat32().MulAdd(sc, o0)
				o1 = a1.Sub(cs).ConvertToFloat32().MulAdd(sc, o1)
				o2 = a2.Sub(cs).ConvertToFloat32().MulAdd(sc, o2)
				o3 = a3.Sub(cs).ConvertToFloat32().MulAdd(sc, o3)
				o4 = a4.Sub(cs).ConvertToFloat32().MulAdd(sc, o4)
				o5 = a5.Sub(cs).ConvertToFloat32().MulAdd(sc, o5)
				o6 = a6.Sub(cs).ConvertToFloat32().MulAdd(sc, o6)
				o7 = a7.Sub(cs).ConvertToFloat32().MulAdd(sc, o7)
			}
			for r, o := range [8]archsimd.Float32x4{o0, o1, o2, o3, o4, o5, o6, o7} {
				simd.StoreF32x4(o.Mul(archsimd.BroadcastFloat32x4(sxs[r])), out.Data[(r0+r)*cols+jt:])
			}
		}
	}
	if vecEnd < hi {
		q8gMatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, group, cols, vecEnd, hi)
	}
}

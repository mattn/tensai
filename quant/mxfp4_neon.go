//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// 128-bit NEON kernels for the MXFP4 layout, the arm64 counterpart of
// mxfp4_simd.go. A 16-byte load is eight columns two nibble bytes deep;
// masking gives rows 0 and 2 of every column and shifting rows 1 and 3,
// and zipping the two puts each column's four codes side by side in the
// quad layout, four columns to a vector. One table lookup (TBL) per
// vector expands the codes to the doubled integer grid, and the rest is
// the grouped int8 chain of quant8g_neon.go: signed activations in the
// matvec, spread offset-binary quads with the column-sum correction in
// the batched form, and a fused multiply-add per 32-row group.

// mxfp4LUT16 is the code table as the byte vector TBL indexes.
var mxfp4LUT16 = func() [16]uint8 {
	var t [16]uint8
	for i, v := range mxfp4LUT {
		t[i] = uint8(v)
	}
	return t
}()

// mxfp4Unpack expands eight columns of packed codes into two quad-layout
// weight vectors of four columns each.
func mxfp4Unpack(row []uint8, lut, m4 archsimd.Uint8x16) (archsimd.Int8x16, archsimd.Int8x16) {
	v := simd.LoadU8x16(row)
	lo := v.And(m4)          // rows 0 and 2, per column
	hi := v.ShiftAllRight(4) // rows 1 and 3
	w0 := lut.LookupOrZero(lo.InterleaveLo(hi)).BitsToInt8()
	w1 := lut.LookupOrZero(lo.InterleaveHi(hi)).BitsToInt8()
	return w0, w1
}

func mxfp4MatvecCols(out []tensai.Float, xu []uint8, sx tensai.Float, qw []uint8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	quads := len(xu) / 4
	groups := (len(xu) + 31) / 32
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		lut := simd.LoadU8x16(mxfp4LUT16[:])
		m4 := archsimd.BroadcastUint8x16(0x0F)
		sxv := archsimd.BroadcastFloat32x4(sx)
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile:]
			stab := scale[(jt/q4Tile)*groups*q4Tile:]
			var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Float32x4
			for g := 0; g < groups; g++ {
				ib := g * 8
				ie := min(ib+8, quads)
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
				for i4 := ib; i4 < ie; i4++ {
					xv := qxSigned8(xu, i4)
					row := tile[i4*2*q4Tile : i4*2*q4Tile+2*q4Tile]
					w0, w1 := mxfp4Unpack(row, lut, m4)
					a0 = a0.Add(quadColsV(w0, xv))
					a1 = a1.Add(quadColsV(w1, xv))
					w0, w1 = mxfp4Unpack(row[16:], lut, m4)
					a2 = a2.Add(quadColsV(w0, xv))
					a3 = a3.Add(quadColsV(w1, xv))
					w0, w1 = mxfp4Unpack(row[32:], lut, m4)
					a4 = a4.Add(quadColsV(w0, xv))
					a5 = a5.Add(quadColsV(w1, xv))
					w0, w1 = mxfp4Unpack(row[48:], lut, m4)
					a6 = a6.Add(quadColsV(w0, xv))
					a7 = a7.Add(quadColsV(w1, xv))
				}
				tg := stab[g*q4Tile : g*q4Tile+q4Tile]
				o0 = a0.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg), o0)
				o1 = a1.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[4:]), o1)
				o2 = a2.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[8:]), o2)
				o3 = a3.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[12:]), o3)
				o4 = a4.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[16:]), o4)
				o5 = a5.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[20:]), o5)
				o6 = a6.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[24:]), o6)
				o7 = a7.ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[28:]), o7)
			}
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
		mxfp4MatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

// mxfp4MatmulRows8 is the eight-row batched form: the unpack and the
// lookup run once per eight-column step and feed eight spread activation
// quads, sixteen accumulators for the two four-column halves.
func mxfp4MatmulRows8(out *tensai.Matrix, xus [][]uint8, sxs []tensai.Float, r0 int, qw []uint8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	quads := len(xus[0]) / 4
	groups := (len(xus[0]) + 31) / 32
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		xq := packQuadsRows8(xus)
		lut := simd.LoadU8x16(mxfp4LUT16[:])
		m4 := archsimd.BroadcastUint8x16(0x0F)
		for jt := lo; jt < vecEnd; jt += 8 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile+(jt%q4Tile)*2:]
			stab := scale[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
			ctab := colSum64[(jt/q4Tile)*groups*q4Tile+jt%q4Tile:]
			var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Float32x4
			var p0, p1, p2, p3, p4, p5, p6, p7 archsimd.Float32x4
			for g := 0; g < groups; g++ {
				ib := g * 8
				ie := min(ib+8, quads)
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
				var b0, b1, b2, b3, b4, b5, b6, b7 archsimd.Int32x4
				for i4 := ib; i4 < ie; i4++ {
					w0, w1 := mxfp4Unpack(tile[i4*2*q4Tile:], lut, m4)
					wh0, wh1 := w0.HiToLo(), w1.HiToLo()
					xf := xq[i4*8 : i4*8+8 : i4*8+8]
					x := qxSpread(xf[0])
					a0 = a0.Add(qdot4Hi(x, w0, wh0))
					b0 = b0.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[1])
					a1 = a1.Add(qdot4Hi(x, w0, wh0))
					b1 = b1.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[2])
					a2 = a2.Add(qdot4Hi(x, w0, wh0))
					b2 = b2.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[3])
					a3 = a3.Add(qdot4Hi(x, w0, wh0))
					b3 = b3.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[4])
					a4 = a4.Add(qdot4Hi(x, w0, wh0))
					b4 = b4.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[5])
					a5 = a5.Add(qdot4Hi(x, w0, wh0))
					b5 = b5.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[6])
					a6 = a6.Add(qdot4Hi(x, w0, wh0))
					b6 = b6.Add(qdot4Hi(x, w1, wh1))
					x = qxSpread(xf[7])
					a7 = a7.Add(qdot4Hi(x, w0, wh0))
					b7 = b7.Add(qdot4Hi(x, w1, wh1))
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
				cs = simd.LoadI32x4(ctab[g*q4Tile+4:])
				sc = simd.LoadF32x4(stab[g*q4Tile+4:])
				p0 = b0.Sub(cs).ConvertToFloat32().MulAdd(sc, p0)
				p1 = b1.Sub(cs).ConvertToFloat32().MulAdd(sc, p1)
				p2 = b2.Sub(cs).ConvertToFloat32().MulAdd(sc, p2)
				p3 = b3.Sub(cs).ConvertToFloat32().MulAdd(sc, p3)
				p4 = b4.Sub(cs).ConvertToFloat32().MulAdd(sc, p4)
				p5 = b5.Sub(cs).ConvertToFloat32().MulAdd(sc, p5)
				p6 = b6.Sub(cs).ConvertToFloat32().MulAdd(sc, p6)
				p7 = b7.Sub(cs).ConvertToFloat32().MulAdd(sc, p7)
			}
			for r, o := range [8]archsimd.Float32x4{o0, o1, o2, o3, o4, o5, o6, o7} {
				simd.StoreF32x4(o.Mul(archsimd.BroadcastFloat32x4(sxs[r])), out.Data[(r0+r)*cols+jt:])
			}
			for r, p := range [8]archsimd.Float32x4{p0, p1, p2, p3, p4, p5, p6, p7} {
				simd.StoreF32x4(p.Mul(archsimd.BroadcastFloat32x4(sxs[r])), out.Data[(r0+r)*cols+jt+4:])
			}
		}
	}
	if vecEnd < hi {
		mxfp4MatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

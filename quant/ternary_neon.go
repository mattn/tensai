//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// 128-bit NEON kernels for the ternary layout, the arm64 counterpart of
// ternary_simd.go. A 16-byte load is four columns of a 16-row block, and
// shifting it right by 2s and masking the low two bits of each byte lands
// rows 4s..4s+3 in the quad layout, so an unpack is a byte shift and an
// and per row-quad. NEON has no unsigned-by-signed pair multiply, so the
// codes meet the spread activations through the signed widening multiply
// (SMULL) and the products accumulate as int16 per (column, row) lane
// until the group ends: codes are at most 3 and |activations| at most
// 63, so a lane's 32 quads reach 6048 and a column's four lanes 24192,
// inside int16. What the chain sums is the dot product plus the group's
// activation sum, taken off once per group, the same for every column.
//
// The group fold is a fused multiply-add, as the arm64 compiler fuses
// the portable body's `sum += float(acc) * scale`, and the vector and
// scalar columns of one matvec must agree bit for bit.

// tCodes unpacks the row-quad `shift` bits up a 16-byte block chunk into
// its codes. Callers spell the shift as a constant: a variable count
// costs more than the multiplies it feeds.
func tCodes(v archsimd.Uint8x16, shift uint64, m3 archsimd.Uint8x16) archsimd.Int8x16 {
	return v.ShiftAllRight(shift).And(m3).BitsToInt8()
}

// tFold closes a chunk's two int16 accumulators (columns 0-1 and 2-3,
// four row lanes each) into four column sums.
func tFold(lo, hi archsimd.Int16x8) archsimd.Int32x4 {
	p := lo.ConcatAddPairs(hi)
	return p.ConcatAddPairs(p).ExtendLo4ToInt32()
}

func ternaryMatvecCols(out []tensai.Float, xs []int8, xq []uint32, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	blocks := (rows + 15) / 16
	groups := (rows + tGroup - 1) / tGroup
	vecEnd := lo + ((hi - lo) &^ 15)
	if vecEnd > lo {
		m3 := archsimd.BroadcastUint8x16(3)
		sxv := archsimd.BroadcastFloat32x4(sx)
		// Sixteen columns per pass: four chunks, two accumulators each,
		// which leaves registers for the unpacked quad and the broadcast.
		for jt := lo; jt < vecEnd; jt += 16 {
			tile := qw[(jt/tTile)*blocks*tBlock+((jt%tTile)/8)*32:]
			stab := scale[(jt/tTile)*groups*tTile+jt%tTile:]
			var o0, o1, o2, o3 archsimd.Float32x4
			for g := 0; g < groups; g++ {
				bb := g * (tGroup / 16)
				be := min(bb+tGroup/16, blocks)
				var a0, b0, a1, b1, a2, b2, a3, b3 archsimd.Int16x8
				for b := bb; b < be; b++ {
					blk := tile[b*tBlock : b*tBlock+64]
					v0 := simd.LoadU8x16(blk)
					v1 := simd.LoadU8x16(blk[16:])
					v2 := simd.LoadU8x16(blk[32:])
					v3 := simd.LoadU8x16(blk[48:])
					x := qxSpread(xq[4*b+0])
					w := tCodes(v0, 0, m3)
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v1, 0, m3)
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v2, 0, m3)
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v3, 0, m3)
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(w.HiToLo().MulWidenLo(x))
					x = qxSpread(xq[4*b+1])
					w = tCodes(v0, 2, m3)
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v1, 2, m3)
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v2, 2, m3)
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v3, 2, m3)
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(w.HiToLo().MulWidenLo(x))
					x = qxSpread(xq[4*b+2])
					w = tCodes(v0, 4, m3)
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v1, 4, m3)
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v2, 4, m3)
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v3, 4, m3)
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(w.HiToLo().MulWidenLo(x))
					x = qxSpread(xq[4*b+3])
					w = tCodes(v0, 6, m3)
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v1, 6, m3)
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v2, 6, m3)
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(w.HiToLo().MulWidenLo(x))
					w = tCodes(v3, 6, m3)
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(w.HiToLo().MulWidenLo(x))
				}
				gs := archsimd.BroadcastInt32x4(gsum[g])
				tg := stab[g*tTile : g*tTile+16]
				o0 = tFold(a0, b0).Sub(gs).ConvertToFloat32().MulAdd(simd.LoadF32x4(tg), o0)
				o1 = tFold(a1, b1).Sub(gs).ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[4:]), o1)
				o2 = tFold(a2, b2).Sub(gs).ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[8:]), o2)
				o3 = tFold(a3, b3).Sub(gs).ConvertToFloat32().MulAdd(simd.LoadF32x4(tg[12:]), o3)
			}
			o := out[jt : jt+16 : jt+16]
			simd.StoreF32x4(o0.Mul(sxv), o)
			simd.StoreF32x4(o1.Mul(sxv), o[4:])
			simd.StoreF32x4(o2.Mul(sxv), o[8:])
			simd.StoreF32x4(o3.Mul(sxv), o[12:])
		}
	}
	if vecEnd < hi {
		ternaryMatvecColsGeneric(out, xs, sx, gsum, qw, scale, rows, cols, vecEnd, hi)
	}
}

// ternaryMatmulRows8 is the eight-row batched form: two passes of four
// rows over the same weights, as on amd64. Four rows keep the
// accumulators at eight, and the second pass reads the block back from
// L1.
func ternaryMatmulRows8(out *tensai.Matrix, xss [][]int8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	ternaryMatmulRows4(out, xss[:4], sxs[:4], gsums[:4], r0, qw, scale, rows, cols, lo, hi)
	ternaryMatmulRows4(out, xss[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, rows, cols, lo, hi)
}

func ternaryMatmulRows4(out *tensai.Matrix, xss [][]int8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	blocks := (rows + 15) / 16
	groups := (rows + tGroup - 1) / tGroup
	xq0, xq1, xq2, xq3 := xsQuads(xss[0]), xsQuads(xss[1]), xsQuads(xss[2]), xsQuads(xss[3])
	gs0, gs1, gs2, gs3 := gsums[0], gsums[1], gsums[2], gsums[3]
	vecEnd := lo + ((hi - lo) &^ 3)
	if vecEnd > lo {
		m3 := archsimd.BroadcastUint8x16(3)
		for jt := lo; jt < vecEnd; jt += 4 {
			tile := qw[(jt/tTile)*blocks*tBlock+((jt%tTile)/8)*32+(jt%8)*4:]
			stab := scale[(jt/tTile)*groups*tTile+jt%tTile:]
			var o0, o1, o2, o3 archsimd.Float32x4
			for g := 0; g < groups; g++ {
				bb := g * (tGroup / 16)
				be := min(bb+tGroup/16, blocks)
				var a0, b0, a1, b1, a2, b2, a3, b3 archsimd.Int16x8
				for b := bb; b < be; b++ {
					v := simd.LoadU8x16(tile[b*tBlock:])
					q := 4 * b
					w := tCodes(v, 0, m3)
					wh := w.HiToLo()
					x := qxSpread(xq0[q+0])
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(wh.MulWidenLo(x))
					x = qxSpread(xq1[q+0])
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(wh.MulWidenLo(x))
					x = qxSpread(xq2[q+0])
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(wh.MulWidenLo(x))
					x = qxSpread(xq3[q+0])
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(wh.MulWidenLo(x))
					w = tCodes(v, 2, m3)
					wh = w.HiToLo()
					x = qxSpread(xq0[q+1])
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(wh.MulWidenLo(x))
					x = qxSpread(xq1[q+1])
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(wh.MulWidenLo(x))
					x = qxSpread(xq2[q+1])
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(wh.MulWidenLo(x))
					x = qxSpread(xq3[q+1])
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(wh.MulWidenLo(x))
					w = tCodes(v, 4, m3)
					wh = w.HiToLo()
					x = qxSpread(xq0[q+2])
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(wh.MulWidenLo(x))
					x = qxSpread(xq1[q+2])
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(wh.MulWidenLo(x))
					x = qxSpread(xq2[q+2])
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(wh.MulWidenLo(x))
					x = qxSpread(xq3[q+2])
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(wh.MulWidenLo(x))
					w = tCodes(v, 6, m3)
					wh = w.HiToLo()
					x = qxSpread(xq0[q+3])
					a0 = a0.Add(w.MulWidenLo(x))
					b0 = b0.Add(wh.MulWidenLo(x))
					x = qxSpread(xq1[q+3])
					a1 = a1.Add(w.MulWidenLo(x))
					b1 = b1.Add(wh.MulWidenLo(x))
					x = qxSpread(xq2[q+3])
					a2 = a2.Add(w.MulWidenLo(x))
					b2 = b2.Add(wh.MulWidenLo(x))
					x = qxSpread(xq3[q+3])
					a3 = a3.Add(w.MulWidenLo(x))
					b3 = b3.Add(wh.MulWidenLo(x))
				}
				sc := simd.LoadF32x4(stab[g*tTile:])
				o0 = tFold(a0, b0).Sub(archsimd.BroadcastInt32x4(gs0[g])).ConvertToFloat32().MulAdd(sc, o0)
				o1 = tFold(a1, b1).Sub(archsimd.BroadcastInt32x4(gs1[g])).ConvertToFloat32().MulAdd(sc, o1)
				o2 = tFold(a2, b2).Sub(archsimd.BroadcastInt32x4(gs2[g])).ConvertToFloat32().MulAdd(sc, o2)
				o3 = tFold(a3, b3).Sub(archsimd.BroadcastInt32x4(gs3[g])).ConvertToFloat32().MulAdd(sc, o3)
			}
			simd.StoreF32x4(o0.Mul(archsimd.BroadcastFloat32x4(sxs[0])), out.Data[r0*cols+jt:])
			simd.StoreF32x4(o1.Mul(archsimd.BroadcastFloat32x4(sxs[1])), out.Data[(r0+1)*cols+jt:])
			simd.StoreF32x4(o2.Mul(archsimd.BroadcastFloat32x4(sxs[2])), out.Data[(r0+2)*cols+jt:])
			simd.StoreF32x4(o3.Mul(archsimd.BroadcastFloat32x4(sxs[3])), out.Data[(r0+3)*cols+jt:])
		}
	}
	if vecEnd < hi {
		for r := 0; r < 4; r++ {
			ternaryMatvecColsGeneric(out.Data[(r0+r)*cols:(r0+r+1)*cols], xss[r], sxs[r], gsums[r], qw, scale, rows, cols, vecEnd, hi)
		}
	}
}

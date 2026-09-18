//go:build goexperiment.simd && amd64

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// 256-bit AVX2 kernels for the ternary layout. A 32-byte block holds 16
// rows of 8 columns, and shifting it right by 2s and masking the low two
// bits of each byte lands rows 4s..4s+3 in the quad layout, so the
// unpack is a shift and an and per row-quad. The codes are the unsigned
// operand of VPMADDUBSW and the activations the signed one: what the
// chain sums is the dot product plus the group's activation sum, taken
// off once per group, the same for every column.

func ternaryMatvecCols(out []tensai.Float, xs []int8, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	if !simd.HasAVX2 {
		ternaryMatvecColsGeneric(out, xs, sx, gsum, qw, scale, rows, cols, lo, hi)
		return
	}
	blocks := (rows + 15) / 16
	groups := (rows + tGroup - 1) / tGroup
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		m3 := archsimd.BroadcastUint8x32(3)
		ones := archsimd.BroadcastInt16x16(1)
		clear(out[lo:vecEnd])
		xq := xsQuads(xs)
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/tTile)*blocks*tBlock:]
			stab := scale[(jt/tTile)*groups*tTile:]
			d0 := out[jt : jt+8 : jt+8]
			d1 := out[jt+8 : jt+16 : jt+16]
			d2 := out[jt+16 : jt+24 : jt+24]
			d3 := out[jt+24 : jt+32 : jt+32]
			for g := 0; g < groups; g++ {
				bb := g * (tGroup / 16)
				be := min(bb+tGroup/16, blocks)
				var a0, a1, a2, a3 archsimd.Int32x8
				for b := bb; b < be; b++ {
					blk := tile[b*tBlock:]
					x0 := archsimd.BroadcastUint32x8(xq[4*b]).AsInt8x32()
					x1 := archsimd.BroadcastUint32x8(xq[4*b+1]).AsInt8x32()
					x2 := archsimd.BroadcastUint32x8(xq[4*b+2]).AsInt8x32()
					x3 := archsimd.BroadcastUint32x8(xq[4*b+3]).AsInt8x32()
					v := simd.LoadU8x32(blk)
					a0 = a0.Add(v.And(m3).DotProductPairsSaturated(x0).DotProductPairs(ones))
					a0 = a0.Add(v.AsUint16x16().ShiftAllRight(2).AsUint8x32().And(m3).DotProductPairsSaturated(x1).DotProductPairs(ones))
					a0 = a0.Add(v.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(m3).DotProductPairsSaturated(x2).DotProductPairs(ones))
					a0 = a0.Add(v.AsUint16x16().ShiftAllRight(6).AsUint8x32().And(m3).DotProductPairsSaturated(x3).DotProductPairs(ones))
					v = simd.LoadU8x32(blk[32:])
					a1 = a1.Add(v.And(m3).DotProductPairsSaturated(x0).DotProductPairs(ones))
					a1 = a1.Add(v.AsUint16x16().ShiftAllRight(2).AsUint8x32().And(m3).DotProductPairsSaturated(x1).DotProductPairs(ones))
					a1 = a1.Add(v.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(m3).DotProductPairsSaturated(x2).DotProductPairs(ones))
					a1 = a1.Add(v.AsUint16x16().ShiftAllRight(6).AsUint8x32().And(m3).DotProductPairsSaturated(x3).DotProductPairs(ones))
					v = simd.LoadU8x32(blk[64:])
					a2 = a2.Add(v.And(m3).DotProductPairsSaturated(x0).DotProductPairs(ones))
					a2 = a2.Add(v.AsUint16x16().ShiftAllRight(2).AsUint8x32().And(m3).DotProductPairsSaturated(x1).DotProductPairs(ones))
					a2 = a2.Add(v.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(m3).DotProductPairsSaturated(x2).DotProductPairs(ones))
					a2 = a2.Add(v.AsUint16x16().ShiftAllRight(6).AsUint8x32().And(m3).DotProductPairsSaturated(x3).DotProductPairs(ones))
					v = simd.LoadU8x32(blk[96:])
					a3 = a3.Add(v.And(m3).DotProductPairsSaturated(x0).DotProductPairs(ones))
					a3 = a3.Add(v.AsUint16x16().ShiftAllRight(2).AsUint8x32().And(m3).DotProductPairsSaturated(x1).DotProductPairs(ones))
					a3 = a3.Add(v.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(m3).DotProductPairsSaturated(x2).DotProductPairs(ones))
					a3 = a3.Add(v.AsUint16x16().ShiftAllRight(6).AsUint8x32().And(m3).DotProductPairsSaturated(x3).DotProductPairs(ones))
				}
				gs := archsimd.BroadcastInt32x8(gsum[g])
				tg := stab[g*tTile:]
				simd.StoreF32x8(simd.LoadF32x8(d0).Add(a0.Sub(gs).ConvertToFloat32().Mul(simd.LoadF32x8(tg))), d0)
				simd.StoreF32x8(simd.LoadF32x8(d1).Add(a1.Sub(gs).ConvertToFloat32().Mul(simd.LoadF32x8(tg[8:]))), d1)
				simd.StoreF32x8(simd.LoadF32x8(d2).Add(a2.Sub(gs).ConvertToFloat32().Mul(simd.LoadF32x8(tg[16:]))), d2)
				simd.StoreF32x8(simd.LoadF32x8(d3).Add(a3.Sub(gs).ConvertToFloat32().Mul(simd.LoadF32x8(tg[24:]))), d3)
			}
		}
		sxv := archsimd.BroadcastFloat32x8(sx)
		for j := lo; j < vecEnd; j += 8 {
			simd.StoreF32x8(simd.LoadF32x8(out[j:]).Mul(sxv), out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		ternaryMatvecColsGeneric(out, xs, sx, gsum, qw, scale, rows, cols, vecEnd, hi)
	}
}

// xsQuads packs a signed activation row four bytes to a word, one word
// per row-quad, for the broadcasts.
func xsQuads(xs []int8) []uint32 {
	xq := make([]uint32, len(xs)/4)
	for i := range xq {
		xq[i] = uint32(uint8(xs[4*i])) | uint32(uint8(xs[4*i+1]))<<8 | uint32(uint8(xs[4*i+2]))<<16 | uint32(uint8(xs[4*i+3]))<<24
	}
	return xq
}

// ternaryMatmulRows8 is the eight-row batched form: two passes of four
// rows over the same weights. Four accumulators leave room for the
// unpacked quad, the masks and a broadcast without spilling, and the
// second pass reads the block back from L1. Each block unpacks its four
// row-quads with constant shifts; a variable shift count costs more
// than the multiply-adds it feeds.
func ternaryMatmulRows8(out *tensai.Matrix, xss [][]int8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	if !simd.HasAVX2 {
		ternaryMatmulRows8Generic(out, xss, sxs, gsums, r0, qw, scale, rows, cols, lo, hi)
		return
	}
	ternaryMatmulRows4(out, xss[:4], sxs[:4], gsums[:4], r0, qw, scale, rows, cols, lo, hi)
	ternaryMatmulRows4(out, xss[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, rows, cols, lo, hi)
}

func ternaryMatmulRows4(out *tensai.Matrix, xss [][]int8, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, rows, cols, lo, hi int) {
	blocks := (rows + 15) / 16
	groups := (rows + tGroup - 1) / tGroup
	xq0, xq1, xq2, xq3 := xsQuads(xss[0]), xsQuads(xss[1]), xsQuads(xss[2]), xsQuads(xss[3])
	gs0, gs1, gs2, gs3 := gsums[0], gsums[1], gsums[2], gsums[3]
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		m3 := archsimd.BroadcastUint8x32(3)
		ones := archsimd.BroadcastInt16x16(1)
		for r := 0; r < 4; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		o0 := out.Data[r0*cols:]
		o1 := out.Data[(r0+1)*cols:]
		o2 := out.Data[(r0+2)*cols:]
		o3 := out.Data[(r0+3)*cols:]
		for jt := lo; jt < vecEnd; jt += 8 {
			tile := qw[(jt/tTile)*blocks*tBlock+((jt%tTile)/8)*32:]
			stab := scale[(jt/tTile)*groups*tTile+jt%tTile:]
			d0 := o0[jt : jt+8 : jt+8]
			d1 := o1[jt : jt+8 : jt+8]
			d2 := o2[jt : jt+8 : jt+8]
			d3 := o3[jt : jt+8 : jt+8]
			for g := 0; g < groups; g++ {
				bb := g * (tGroup / 16)
				be := min(bb+tGroup/16, blocks)
				var a0, a1, a2, a3 archsimd.Int32x8
				for b := bb; b < be; b++ {
					v := simd.LoadU8x32(tile[b*tBlock:])
					q := 4 * b
					w := v.And(m3)
					a0 = a0.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[q]).AsInt8x32()).DotProductPairs(ones))
					a1 = a1.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[q]).AsInt8x32()).DotProductPairs(ones))
					a2 = a2.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[q]).AsInt8x32()).DotProductPairs(ones))
					a3 = a3.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[q]).AsInt8x32()).DotProductPairs(ones))
					w = v.AsUint16x16().ShiftAllRight(2).AsUint8x32().And(m3)
					a0 = a0.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[q+1]).AsInt8x32()).DotProductPairs(ones))
					a1 = a1.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[q+1]).AsInt8x32()).DotProductPairs(ones))
					a2 = a2.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[q+1]).AsInt8x32()).DotProductPairs(ones))
					a3 = a3.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[q+1]).AsInt8x32()).DotProductPairs(ones))
					w = v.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(m3)
					a0 = a0.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[q+2]).AsInt8x32()).DotProductPairs(ones))
					a1 = a1.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[q+2]).AsInt8x32()).DotProductPairs(ones))
					a2 = a2.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[q+2]).AsInt8x32()).DotProductPairs(ones))
					a3 = a3.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[q+2]).AsInt8x32()).DotProductPairs(ones))
					w = v.AsUint16x16().ShiftAllRight(6).AsUint8x32().And(m3)
					a0 = a0.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq0[q+3]).AsInt8x32()).DotProductPairs(ones))
					a1 = a1.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq1[q+3]).AsInt8x32()).DotProductPairs(ones))
					a2 = a2.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq2[q+3]).AsInt8x32()).DotProductPairs(ones))
					a3 = a3.Add(w.DotProductPairsSaturated(archsimd.BroadcastUint32x8(xq3[q+3]).AsInt8x32()).DotProductPairs(ones))
				}
				sc := simd.LoadF32x8(stab[g*tTile:])
				simd.StoreF32x8(simd.LoadF32x8(d0).Add(a0.Sub(archsimd.BroadcastInt32x8(gs0[g])).ConvertToFloat32().Mul(sc)), d0)
				simd.StoreF32x8(simd.LoadF32x8(d1).Add(a1.Sub(archsimd.BroadcastInt32x8(gs1[g])).ConvertToFloat32().Mul(sc)), d1)
				simd.StoreF32x8(simd.LoadF32x8(d2).Add(a2.Sub(archsimd.BroadcastInt32x8(gs2[g])).ConvertToFloat32().Mul(sc)), d2)
				simd.StoreF32x8(simd.LoadF32x8(d3).Add(a3.Sub(archsimd.BroadcastInt32x8(gs3[g])).ConvertToFloat32().Mul(sc)), d3)
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
		for r := 0; r < 4; r++ {
			ternaryMatvecColsGeneric(out.Data[(r0+r)*cols:(r0+r+1)*cols], xss[r], sxs[r], gsums[r], qw, scale, rows, cols, vecEnd, hi)
		}
	}
}

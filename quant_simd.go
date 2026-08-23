//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// AVX2 kernel for the quantized matvec, built on the u8 x s8 pairwise
// multiply-add: the 7-bit offset activations of one row pair broadcast as
// repeated (x_even, x_odd) bytes, one 16-byte load of the interleaved
// weight layout covers 8 columns of the pair, and a single
// DotProductPairsSaturated yields the 8 per-column pair sums — no
// widening, masking, or shuffling of weights at all. Pair sums stay under
// 127*127*2 so the saturating i16 never saturates; they widen to an i32
// accumulator strip.
//
// The loop nest keeps weight reads streaming: an outer block of 32 row
// pairs, a middle loop over 32-column tiles (64 interleaved bytes — one
// cache line per pair-row), and the pair rows inner with four i32x8
// accumulators live in registers, flushed to the strip once per tile.
func qmatvecCols(out []Float, xu []uint8, sx Float, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !hasAVX2 {
		qmatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, lo, hi)
		return
	}
	pairs := len(xu) / 2
	acc := make([]int32, hi-lo)
	vecEnd := lo + ((hi - lo) &^ 31)
	const iBlock = 32
	for ib := 0; ib < pairs; ib += iBlock {
		ie := min(ib+iBlock, pairs)
		for jt := lo; jt < vecEnd; jt += 32 {
			var a0, a1, a2, a3 archsimd.Int32x8
			for i2 := ib; i2 < ie; i2++ {
				xp := archsimd.BroadcastUint16x8(uint16(xu[2*i2]) | uint16(xu[2*i2+1])<<8).AsUint8x16()
				row := qw[i2*2*cols+2*jt:]
				a0 = a0.Add(xp.DotProductPairsSaturated(loadI8x16(row)).ExtendToInt32())
				a1 = a1.Add(xp.DotProductPairsSaturated(loadI8x16(row[16:])).ExtendToInt32())
				a2 = a2.Add(xp.DotProductPairsSaturated(loadI8x16(row[32:])).ExtendToInt32())
				a3 = a3.Add(xp.DotProductPairsSaturated(loadI8x16(row[48:])).ExtendToInt32())
			}
			s := acc[jt-lo:]
			storeI32x8(loadI32x8(s).Add(a0), s)
			storeI32x8(loadI32x8(s[8:]).Add(a1), s[8:])
			storeI32x8(loadI32x8(s[16:]).Add(a2), s[16:])
			storeI32x8(loadI32x8(s[24:]).Add(a3), s[24:])
		}
		// Scalar tail columns: integer-only, safe while the vector upper
		// state is dirty.
		for i2 := ib; i2 < ie; i2++ {
			x0, x1 := int32(xu[2*i2]), int32(xu[2*i2+1])
			row := qw[i2*2*cols:]
			for j := vecEnd; j < hi; j++ {
				acc[j-lo] += x0*int32(row[2*j]) + x1*int32(row[2*j+1])
			}
		}
	}
	sxv := archsimd.BroadcastFloat32x8(sx)
	for j := lo; j < vecEnd; j += 8 {
		v := loadI32x8(acc[j-lo:]).Sub(loadI32x8(colSum64[j:])).ConvertToFloat32()
		storeF32x8(v.Mul(loadF32x8(scale[j:])).Mul(sxv), out[j:])
	}
	archsimd.ClearAVXUpperBits()
	for j := vecEnd; j < hi; j++ {
		out[j] = Float(acc[j-lo]-colSum64[j]) * scale[j] * sx
	}
}

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

// qmatmulCols4 is the batched kernel: two weight loads per 16-column step
// feed eight multiply-adds, one pair per activation row, so the streaming
// weight traffic amortizes fourfold.
func qmatmulCols4(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !hasAVX2 {
		qmatmulCols4Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
		return
	}
	pairs := len(xus[0]) / 2
	width := hi - lo
	accs := make([]int32, 4*width)
	vecEnd := lo + (width &^ 15)
	const iBlock = 32
	for ib := 0; ib < pairs; ib += iBlock {
		ie := min(ib+iBlock, pairs)
		for jt := lo; jt < vecEnd; jt += 16 {
			var a00, a01, a10, a11, a20, a21, a30, a31 archsimd.Int32x8
			for i2 := ib; i2 < ie; i2++ {
				row := qw[i2*2*cols+2*jt:]
				w0 := loadI8x16(row)
				w1 := loadI8x16(row[16:])
				xp := archsimd.BroadcastUint16x8(uint16(xus[0][2*i2]) | uint16(xus[0][2*i2+1])<<8).AsUint8x16()
				a00 = a00.Add(xp.DotProductPairsSaturated(w0).ExtendToInt32())
				a01 = a01.Add(xp.DotProductPairsSaturated(w1).ExtendToInt32())
				xp = archsimd.BroadcastUint16x8(uint16(xus[1][2*i2]) | uint16(xus[1][2*i2+1])<<8).AsUint8x16()
				a10 = a10.Add(xp.DotProductPairsSaturated(w0).ExtendToInt32())
				a11 = a11.Add(xp.DotProductPairsSaturated(w1).ExtendToInt32())
				xp = archsimd.BroadcastUint16x8(uint16(xus[2][2*i2]) | uint16(xus[2][2*i2+1])<<8).AsUint8x16()
				a20 = a20.Add(xp.DotProductPairsSaturated(w0).ExtendToInt32())
				a21 = a21.Add(xp.DotProductPairsSaturated(w1).ExtendToInt32())
				xp = archsimd.BroadcastUint16x8(uint16(xus[3][2*i2]) | uint16(xus[3][2*i2+1])<<8).AsUint8x16()
				a30 = a30.Add(xp.DotProductPairsSaturated(w0).ExtendToInt32())
				a31 = a31.Add(xp.DotProductPairsSaturated(w1).ExtendToInt32())
			}
			for r, pair := range [4][2]archsimd.Int32x8{{a00, a01}, {a10, a11}, {a20, a21}, {a30, a31}} {
				s := accs[r*width+jt-lo:]
				storeI32x8(loadI32x8(s).Add(pair[0]), s)
				storeI32x8(loadI32x8(s[8:]).Add(pair[1]), s[8:])
			}
		}
		for i2 := ib; i2 < ie; i2++ {
			row := qw[i2*2*cols:]
			for r := 0; r < 4; r++ {
				x0, x1 := int32(xus[r][2*i2]), int32(xus[r][2*i2+1])
				a := accs[r*width:]
				for j := vecEnd; j < hi; j++ {
					a[j-lo] += x0*int32(row[2*j]) + x1*int32(row[2*j+1])
				}
			}
		}
	}
	for r := 0; r < 4; r++ {
		o := out.Data[(r0+r)*cols:]
		sxv := archsimd.BroadcastFloat32x8(sxs[r])
		a := accs[r*width:]
		for j := lo; j < vecEnd; j += 8 {
			v := loadI32x8(a[j-lo:]).Sub(loadI32x8(colSum64[j:])).ConvertToFloat32()
			storeF32x8(v.Mul(loadF32x8(scale[j:])).Mul(sxv), o[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	for r := 0; r < 4; r++ {
		o := out.Data[(r0+r)*cols:]
		for j := vecEnd; j < hi; j++ {
			o[j] = Float(accs[r*width+j-lo]-colSum64[j]) * scale[j] * sxs[r]
		}
	}
}

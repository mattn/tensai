//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// 256-bit AVX2 kernels for the int8 matvec and matmul over the quad-row
// layout: a 32-byte weight load holds eight columns four rows deep, the
// u8 x s8 pairwise multiply-add (VPMADDUBSW) against a broadcast
// activation quad gives sixteen i16 pair sums, and the widening i16
// pair-add (VPMADDWD by ones) folds them into eight per-column i32 lanes.
// Two multiplies per 32 weight bytes, and no i16 saturation window to
// manage: every quad widens to i32 immediately (pair sums stay under
// 127*127*2, and the pair-add doubles that far inside i32).
//
// Scalar float instructions (SSE encodings) would pay AVX-SSE transition
// penalties while the 256-bit registers are dirty, so integer scalar tails
// run before ClearAVXUpperBits and float tails after.

// qxQuad packs four consecutive 7-bit activations into the u32 a
// broadcast turns into the unsigned multiply operand.
func qxQuad(xu []uint8, i4 int) uint32 {
	return uint32(xu[4*i4]) | uint32(xu[4*i4+1])<<8 | uint32(xu[4*i4+2])<<16 | uint32(xu[4*i4+3])<<24
}

func qmatvecCols(out []Float, xu []uint8, sx Float, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !hasAVX2 {
		qmatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, lo, hi)
		return
	}
	quads := len(xu) / 4
	ones := archsimd.BroadcastInt16x16(1)
	vecEnd := lo + ((hi - lo) &^ 31)
	for jt := lo; jt < vecEnd; jt += 32 {
		var a0, a1, a2, a3 archsimd.Int32x8
		for i4 := 0; i4 < quads; i4++ {
			xp := archsimd.BroadcastUint32x8(qxQuad(xu, i4)).AsUint8x32()
			row := qw[i4*4*cols+4*jt:]
			a0 = a0.Add(xp.DotProductPairsSaturated(loadI8x32(row)).DotProductPairs(ones))
			a1 = a1.Add(xp.DotProductPairsSaturated(loadI8x32(row[32:])).DotProductPairs(ones))
			a2 = a2.Add(xp.DotProductPairsSaturated(loadI8x32(row[64:])).DotProductPairs(ones))
			a3 = a3.Add(xp.DotProductPairsSaturated(loadI8x32(row[96:])).DotProductPairs(ones))
		}
		sxv := archsimd.BroadcastFloat32x8(sx)
		for k, a := range [4]archsimd.Int32x8{a0, a1, a2, a3} {
			j := jt + 8*k
			f := a.Sub(loadI32x8(colSum64[j:])).ConvertToFloat32().Mul(loadF32x8(scale[j:])).Mul(sxv)
			storeF32x8(f, out[j:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		qmatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

// qmatmulRows8 is the eight-row batched form: per eight-column tile each
// 32-byte weight load feeds eight broadcast activation quads, so the
// weight stream that dominates a single matvec amortizes eightfold while
// every row still costs just the two multiplies per load.
func qmatmulRows8(out *Matrix, xus [][]uint8, sxs []Float, r0 int, qw []int8, scale []Float, colSum64 []int32, cols, lo, hi int) {
	if !hasAVX2 {
		qmatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
		return
	}
	quads := len(xus[0]) / 4
	// The packed activation quads, interleaved per quad row so the inner
	// loop walks one contiguous stream. Accumulators are eight named
	// variables: an array of SIMD values would live on the stack and turn
	// every multiply-add into a load-op-store round trip.
	xq := make([]uint32, 8*quads)
	for r := 0; r < 8; r++ {
		for i4 := 0; i4 < quads; i4++ {
			xq[i4*8+r] = qxQuad(xus[r], i4)
		}
	}
	ones := archsimd.BroadcastInt16x16(1)
	vecEnd := lo + ((hi - lo) &^ 7)
	for jt := lo; jt < vecEnd; jt += 8 {
		var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x8
		for i4 := 0; i4 < quads; i4++ {
			w := loadI8x32(qw[i4*4*cols+4*jt:])
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
		cs := loadI32x8(colSum64[jt:])
		sc := loadF32x8(scale[jt:])
		for r, a := range [8]archsimd.Int32x8{a0, a1, a2, a3, a4, a5, a6, a7} {
			f := a.Sub(cs).ConvertToFloat32().Mul(sc).Mul(archsimd.BroadcastFloat32x8(sxs[r]))
			storeF32x8(f, out.Data[(r0+r)*cols+jt:])
		}
	}
	archsimd.ClearAVXUpperBits()
	if vecEnd < hi {
		qmatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

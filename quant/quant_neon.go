//go:build goexperiment.simd && arm64

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// 128-bit NEON kernels for the int8 matvec over the quad-row layout, the
// arm64 counterpart of quant_simd.go. NEON has no unsigned-by-signed
// pairwise multiply-add (VPMADDUBSW) and no widening pair-add (VPMADDWD),
// so the fold is built by hand: sign-extend the bytes to int16, multiply,
// and let two pairwise adds (VADDP) collapse a column's four rows into
// one lane. Sixteen weight bytes hold four columns four rows deep, so one
// pass over 128 bytes covers a 32-column tile.
//
// Because the widening is explicit, the activations can stay signed
// rather than carrying the +64 bias the u8-by-s8 multiply forces on
// amd64. That drops the column-sum correction: the accumulator is already
// the value the amd64 kernel reaches only after subtracting it, so the
// two agree exactly and this side never reads the table.
//
// Overflow: |activation| <= 63 and |weight| <= 127, so a product fits
// 8001, a pair 16002, and a column's four rows 32004 -- inside int16 with
// room to spare. The per-quad column sums widen to int32 before they
// accumulate.

// qxSigned8 broadcasts one activation quad as sixteen signed bytes,
// [x0 x1 x2 x3] four times over, with the storage bias removed.
func qxSigned8(xu []uint8, i4 int) archsimd.Int16x8 {
	q := uint32(xu[4*i4]) | uint32(xu[4*i4+1])<<8 |
		uint32(xu[4*i4+2])<<16 | uint32(xu[4*i4+3])<<24
	v := archsimd.BroadcastUint32x4(q).ReshapeToUint8s().BitsToInt8()
	return v.ExtendLo8ToInt16().Sub(archsimd.BroadcastInt16x8(64))
}

// quadCols folds one 16-byte group (four columns, four rows) against the
// broadcast activations into four int32 column sums.
func quadCols(row []int8, xv archsimd.Int16x8) archsimd.Int32x4 {
	wv := simd.LoadI8x16(row)
	w0 := wv.ExtendLo8ToInt16()                // columns 0 and 1
	w1 := wv.HiToLo().ExtendLo8ToInt16()       // columns 2 and 3
	p := w0.Mul(xv).ConcatAddPairs(w1.Mul(xv)) // two halves per column
	return p.ConcatAddPairs(p).ExtendLo4ToInt32()
}

func qmatvecCols(out []tensai.Float, xu []uint8, sx tensai.Float, qw []int8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	quads := len(xu) / 4
	vecEnd := lo + ((hi - lo) &^ 31)
	for jt := lo; jt < vecEnd; jt += 32 {
		tile := qw[(jt/q4Tile)*quads*4*q4Tile:]
		var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
		for i4 := 0; i4 < quads; i4++ {
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
		sxv := archsimd.BroadcastFloat32x4(sx)
		o := out[jt : jt+32 : jt+32]
		sc := scale[jt : jt+32 : jt+32]
		store := func(a archsimd.Int32x4, k int) {
			f := a.ConvertToFloat32().Mul(simd.LoadF32x4(sc[k:])).Mul(sxv)
			simd.StoreF32x4(f, o[k:])
		}
		store(a0, 0)
		store(a1, 4)
		store(a2, 8)
		store(a3, 12)
		store(a4, 16)
		store(a5, 20)
		store(a6, 24)
		store(a7, 28)
	}
	if vecEnd < hi {
		qmatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

// qmatmulRows8 keeps the portable body for now: prefill amortizes the
// weight stream over eight rows already, and the batched fold wants a
// different shape from the matvec's. See PERF-INVESTIGATION.md.
func qmatmulRows8(out *tensai.Matrix, xus [][]uint8, xq []uint32, sxs []tensai.Float, r0 int, qw []int8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	qmatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, lo, hi)
}

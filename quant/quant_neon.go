//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// 128-bit NEON kernels for the int8 matvec and the batched matmul over
// the quad-row layout, the arm64 counterpart of quant_simd.go. The two
// fold differently, so each carries its own note; this one covers the
// matvec. NEON has no unsigned-by-signed
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
// two agree exactly and the matvec never reads colSum64.
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
	if hasDotProd && quads > 0 && vecEnd > lo {
		qmatvecColsDot(out, xu, sx, qw, scale, quads, lo, vecEnd)
		if vecEnd < hi {
			qmatvecColsGeneric(out, xu, sx, qw, scale, colSum64, cols, vecEnd, hi)
		}
		return
	}
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

// qmatvecColsDot is the same matvec over SDOT, which closes a column's
// four rows in one instruction where the widening path above spends a
// sign-extend, a multiply and two pair-adds. The weights walk a tile
// exactly as the loop above does, so the two produce the same integers:
// SDOT accumulates the four products of a lane in one step, and the
// widening path adds them pairwise, which for exact integers is the same
// sum. The float epilogue is shared, byte for byte.
func qmatvecColsDot(out []tensai.Float, xu []uint8, sx tensai.Float, qw []int8, scale []tensai.Float, quads, lo, vecEnd int) {
	sxv := archsimd.BroadcastFloat32x4(sx)
	var acc [32]int32
	for jt := lo; jt < vecEnd; jt += 32 {
		tile := qw[(jt/q4Tile)*quads*4*q4Tile:]
		sdotTile32(&acc[0], &tile[0], &xu[0], quads)
		o := out[jt : jt+32 : jt+32]
		sc := scale[jt : jt+32 : jt+32]
		for k := 0; k < 32; k += 4 {
			f := simd.LoadI32x4(acc[k:]).ConvertToFloat32().Mul(simd.LoadF32x4(sc[k:])).Mul(sxv)
			simd.StoreF32x4(f, o[k:])
		}
	}
}

// The batched prefill form below folds differently. Its weight load is
// shared by eight activation rows, so the multiply runs eight times per
// load and the cheaper operand matters more than the bias: the signed
// widening multiply (SMULL/SMULL2) takes the weight bytes as they are,
// where the matvec above first sign-extends them. Activations stay in
// their 7-bit offset-binary form, which the +64 correction in colSum64
// folds back out, exactly as the amd64 kernel does.
//
// Overflow: products reach 127*127, so a column's four rows can hit
// 64516. The i16 pair-add still holds the two-row partials, but the sums
// widen to i32 before the second one.

// qxSpread broadcasts a packed activation quad across all sixteen byte
// lanes so both halves of a weight load meet the same four activations.
func qxSpread(quad uint32) archsimd.Int8x16 {
	return archsimd.BroadcastUint32x4(quad).ReshapeToUint8s().BitsToInt8()
}

// qdot4Hi folds one 16-byte weight load -- four columns four rows deep --
// against a spread activation quad into four per-column i32 sums. The
// load's high half arrives already shifted down so a row block pays that
// shift once for all eight rows.
func qdot4Hi(xp, w, wh archsimd.Int8x16) archsimd.Int32x4 {
	p := xp.MulWidenLo(w).ConcatAddPairs(xp.MulWidenLo(wh))
	return p.ExtendLo4ToInt32().ConcatAddPairs(p.HiToLo().ExtendLo4ToInt32())
}

// qscale turns a column's accumulated i32 sums into the output floats:
// the +64 activation offset folds out through colSum64, and the two
// scales bring the product back to the original units.
func qscale(a archsimd.Int32x4, cs []int32, sc []tensai.Float, sxv archsimd.Float32x4, out []tensai.Float) {
	f := a.Sub(simd.LoadI32x4(cs)).ConvertToFloat32().Mul(simd.LoadF32x4(sc)).Mul(sxv)
	simd.StoreF32x4(f, out)
}

// qmatmulRows8 is the eight-row batched form: per four-column tile each
// 16-byte weight load feeds eight spread activation quads, so the weight
// stream that dominates a single matvec amortizes eightfold.
func qmatmulRows8(out *tensai.Matrix, xus [][]uint8, xq []uint32, sxs []tensai.Float, r0 int, qw []int8, scale []tensai.Float, colSum64 []int32, cols, lo, hi int) {
	quads := len(xus[0]) / 4
	// xq holds the caller-packed activation quads, interleaved per quad
	// row so the inner loop walks one contiguous stream.
	vecEnd := lo + ((hi - lo) &^ 3)
	for jt := lo; jt < vecEnd; jt += 4 {
		tile := qw[(jt/q4Tile)*quads*4*q4Tile+(jt%q4Tile)*4:]
		// Eight named accumulators, one per row: an array of SIMD values
		// would live on the stack and turn every multiply-add into a
		// load-op-store round trip.
		var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
		for i4 := 0; i4 < quads; i4++ {
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
		cs := colSum64[jt:]
		sc := scale[jt:]
		for r, a := range [8]archsimd.Int32x4{a0, a1, a2, a3, a4, a5, a6, a7} {
			qscale(a, cs, sc, archsimd.BroadcastFloat32x4(sxs[r]), out.Data[(r0+r)*cols+jt:])
		}
	}
	if vecEnd < hi {
		qmatmulRows8Generic(out, xus, sxs, r0, qw, scale, colSum64, cols, vecEnd, hi)
	}
}

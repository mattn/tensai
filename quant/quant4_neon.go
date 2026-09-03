//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
)

// 128-bit NEON kernels for the 4-bit matvec and matmul over the quad-row
// layout, the arm64 counterpart of the AVX2 kernels in quant4_simd.go.
// Sixteen loaded bytes hold eight columns four rows deep, two nibble
// bytes per column. Expanding them into the int8 kernel's quad layout
// would need two registers where AVX2 needs one, so the nibbles stay
// where the mask and the shift leave them: masking puts column k's rows
// 0 and 2 in lanes 2k and 2k+1, shifting puts its rows 1 and 3 there.
// Each plane then meets a broadcast activation pair — (x0,x2) for the
// low plane, (x1,x3) for the high — and four signed widening multiplies
// plus a pair-add per plane land all eight columns' quad sums in one i16
// vector.
//
// Nibbles are at most 15 and |activations| at most 63, so a column's four
// products sum to 3780 at worst: the quad closes inside i16 and only the
// accumulator widens, one i32 vector per four columns. Group scales and
// the nibble-offset correction (8 times the group's activation sum) fold
// in per group, as on amd64.

// packQuad packs a re-centered activation quad as the two byte pairs the
// nibble planes meet: (x0,x2) in the low half, (x1,x3) in the high.
func packQuad(x []uint8) uint32 {
	return recenter(x[0]) | recenter(x[2])<<8 | recenter(x[1])<<16 | recenter(x[3])<<24
}

// q4Acts spreads a packed quad's two byte pairs over their own vectors.
func q4Acts(quad uint32) (xa, xb archsimd.Int8x16) {
	return q4Spread(uint16(quad)), q4Spread(uint16(quad >> 16))
}

func q4Spread(pair uint16) archsimd.Int8x16 {
	return archsimd.BroadcastUint16x8(pair).ReshapeToUint8s().BitsToInt8()
}

// q4Planes splits a sixteen-byte nibble load into the low and the high
// nibble plane, each next to its own high half shifted down so a row
// block pays that shift once.
func q4Planes(row []uint8, mask archsimd.Uint8x16) (l0, l1, h0, h1 archsimd.Int8x16) {
	w := archsimd.LoadUint8x16(row)
	l := w.And(mask).BitsToInt8()
	h := w.ShiftAllRight(4).BitsToInt8()
	return l, l.HiToLo(), h, h.HiToLo()
}

// q4dot8 folds both nibble planes against one row's activation pairs into
// eight per-column i16 quad sums.
func q4dot8(l0, l1, h0, h1, xa, xb archsimd.Int8x16) archsimd.Int16x8 {
	return xa.MulWidenLo(l0).ConcatAddPairs(xa.MulWidenLo(l1)).
		Add(xb.MulWidenLo(h0).ConcatAddPairs(xb.MulWidenLo(h1)))
}

// q4rowAdd widens one nibble load's eight column sums into a row's pair
// of i32 accumulators.
func q4rowAdd(a0, a1 archsimd.Int32x4, l0, l1, h0, h1, xa, xb archsimd.Int8x16) (archsimd.Int32x4, archsimd.Int32x4) {
	s := q4dot8(l0, l1, h0, h1, xa, xb)
	return a0.Add(s.ExtendLo4ToInt32()), a1.Add(s.HiToLo().ExtendLo4ToInt32())
}

// q4foldSym adds a group's sums into the output in the symmetric form:
// the nibble offset folds out through 8 times the group's activation sum.
func q4foldSym(a, corr archsimd.Int32x4, tg, d []tensai.Float) {
	f := a.Sub(corr).ConvertToFloat32().Mul(archsimd.LoadFloat32x4(tg))
	archsimd.LoadFloat32x4(d).Add(f).Store(d)
}

// q4foldMin is the asymmetric twin: one bfloat16 scale/min pair per
// (group, column), scale in the low half and min in the high.
func q4foldMin(a archsimd.Int32x4, gsf archsimd.Float32x4, mMin archsimd.Uint32x4, tg []uint32, d []tensai.Float) {
	u := archsimd.LoadUint32x4(tg)
	f := a.ConvertToFloat32().Mul(u.ShiftAllLeft(16).BitsToFloat32()).Sub(gsf.Mul(u.And(mMin).BitsToFloat32()))
	archsimd.LoadFloat32x4(d).Add(f).Store(d)
}

// q4scaleRow applies a row's activation scale to a finished column span,
// whose length the callers keep a multiple of four.
func q4scaleRow(out []tensai.Float, sx tensai.Float, lo, hi int) {
	sxv := archsimd.BroadcastFloat32x4(sx)
	for j := lo; j < hi; j += 4 {
		archsimd.LoadFloat32x4(out[j:]).Mul(sxv).Store(out[j:])
	}
}

func q4matvecCols(out []tensai.Float, xu []uint8, xq []uint32, sx tensai.Float, gsum []int32, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	quads := len(xq)
	groups := len(gsum)
	vecEnd := lo + ((hi - lo) &^ 31)
	if vecEnd > lo {
		gsumF := make([]tensai.Float, groups)
		gsum8 := make([]int32, groups)
		for i, v := range gsum {
			gsumF[i] = tensai.Float(v)
			gsum8[i] = 8 * v
		}
		mask := archsimd.BroadcastUint8x16(0x0F)
		mMin := archsimd.BroadcastUint32x4(0xFFFF0000)
		clear(out[lo:vecEnd])
		// Tiles outermost: the nibble walk and the tile-major table walk
		// both advance strictly sequentially per worker.
		for jt := lo; jt < vecEnd; jt += 32 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile:]
			d := out[jt : jt+32 : jt+32]
			for g := 0; g < groups; g++ {
				ib := g * group / 4
				ie := min(ib+group/4, quads)
				// Eight named accumulators, one per four columns: an
				// array of SIMD values would live on the stack and turn
				// every multiply-add into a load-op-store round trip.
				var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Int32x4
				for i4 := ib; i4 < ie; i4++ {
					xa, xb := q4Acts(xq[i4])
					row := tile[i4*2*q4Tile:]
					l0, l1, h0, h1 := q4Planes(row, mask)
					a0, a1 = q4rowAdd(a0, a1, l0, l1, h0, h1, xa, xb)
					l0, l1, h0, h1 = q4Planes(row[16:], mask)
					a2, a3 = q4rowAdd(a2, a3, l0, l1, h0, h1, xa, xb)
					l0, l1, h0, h1 = q4Planes(row[32:], mask)
					a4, a5 = q4rowAdd(a4, a5, l0, l1, h0, h1, xa, xb)
					l0, l1, h0, h1 = q4Planes(row[48:], mask)
					a6, a7 = q4rowAdd(a6, a7, l0, l1, h0, h1, xa, xb)
				}
				acc := [8]archsimd.Int32x4{a0, a1, a2, a3, a4, a5, a6, a7}
				base := (jt/q4Tile)*groups*q4Tile + g*q4Tile
				if sm != nil {
					gsf := archsimd.BroadcastFloat32x4(gsumF[g])
					for k, a := range acc {
						q4foldMin(a, gsf, mMin, sm[base+4*k:], d[4*k:])
					}
				} else {
					corr := archsimd.BroadcastInt32x4(gsum8[g])
					for k, a := range acc {
						q4foldSym(a, corr, scale[base+4*k:], d[4*k:])
					}
				}
			}
		}
		q4scaleRow(out, sx, lo, vecEnd)
	}
	if vecEnd < hi {
		q4matvecColsGeneric(out, xu, sx, gsum, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

// q4matmulCols4 is the four-row batched form: per eight-column step each
// sixteen-byte nibble load splits into planes once and feeds four rows'
// activation pairs, amortizing the nibble traffic fourfold; steps stay
// outermost so both streams advance sequentially.
func q4matmulCols4(out *tensai.Matrix, xus [][]uint8, xqs [][]uint32, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	quads := len(xus[0]) / 4
	groups := len(gsums[0])
	xq0, xq1, xq2, xq3 := xqs[0], xqs[1], xqs[2], xqs[3]
	// Flat copies keep the folds free of nested-slice loads: one indexed
	// read per row instead of a header chase and two bounds checks.
	gsf := make([]tensai.Float, 4*groups)
	gsc := make([]int32, 4*groups)
	for r := 0; r < 4; r++ {
		for g, v := range gsums[r] {
			gsf[4*g+r] = tensai.Float(v)
			gsc[4*g+r] = 8 * v
		}
	}
	vecEnd := lo + ((hi - lo) &^ 7)
	if vecEnd > lo {
		mask := archsimd.BroadcastUint8x16(0x0F)
		mMin := archsimd.BroadcastUint32x4(0xFFFF0000)
		o0 := out.Data[r0*cols:]
		o1 := out.Data[(r0+1)*cols:]
		o2 := out.Data[(r0+2)*cols:]
		o3 := out.Data[(r0+3)*cols:]
		for r := 0; r < 4; r++ {
			clear(out.Data[(r0+r)*cols+lo : (r0+r)*cols+vecEnd])
		}
		for jt := lo; jt < vecEnd; jt += 8 {
			tile := qw[(jt/q4Tile)*quads*2*q4Tile+(jt%q4Tile)*2:]
			d0 := o0[jt : jt+8 : jt+8]
			d1 := o1[jt : jt+8 : jt+8]
			d2 := o2[jt : jt+8 : jt+8]
			d3 := o3[jt : jt+8 : jt+8]
			for g := 0; g < groups; g++ {
				ib := g * group / 4
				ie := min(ib+group/4, quads)
				var a0, a1, b0, b1, c0, c1, e0, e1 archsimd.Int32x4
				for i4 := ib; i4 < ie; i4++ {
					l0, l1, h0, h1 := q4Planes(tile[i4*2*q4Tile:], mask)
					xa, xb := q4Acts(xq0[i4])
					a0, a1 = q4rowAdd(a0, a1, l0, l1, h0, h1, xa, xb)
					xa, xb = q4Acts(xq1[i4])
					b0, b1 = q4rowAdd(b0, b1, l0, l1, h0, h1, xa, xb)
					xa, xb = q4Acts(xq2[i4])
					c0, c1 = q4rowAdd(c0, c1, l0, l1, h0, h1, xa, xb)
					xa, xb = q4Acts(xq3[i4])
					e0, e1 = q4rowAdd(e0, e1, l0, l1, h0, h1, xa, xb)
				}
				base := (jt/q4Tile)*groups*q4Tile + jt%q4Tile + g*q4Tile
				rows := [4][2]archsimd.Int32x4{{a0, a1}, {b0, b1}, {c0, c1}, {e0, e1}}
				dst := [4][]tensai.Float{d0, d1, d2, d3}
				if sm != nil {
					gr := gsf[4*g : 4*g+4 : 4*g+4]
					for r, a := range rows {
						gv := archsimd.BroadcastFloat32x4(gr[r])
						q4foldMin(a[0], gv, mMin, sm[base:], dst[r])
						q4foldMin(a[1], gv, mMin, sm[base+4:], dst[r][4:])
					}
				} else {
					gr := gsc[4*g : 4*g+4 : 4*g+4]
					for r, a := range rows {
						corr := archsimd.BroadcastInt32x4(gr[r])
						q4foldSym(a[0], corr, scale[base:], dst[r])
						q4foldSym(a[1], corr, scale[base+4:], dst[r][4:])
					}
				}
			}
		}
		for r := 0; r < 4; r++ {
			q4scaleRow(out.Data[(r0+r)*cols:], sxs[r], lo, vecEnd)
		}
	}
	if vecEnd < hi {
		q4matmulCols4Generic(out, xus, sxs, gsums, r0, qw, scale, sm, group, cols, vecEnd, hi)
	}
}

// q4matmulCols8 is the eight-row batched body, which runs the four-row
// kernel twice. Eight rows want sixteen i32 accumulators next to the four
// nibble planes; that spills the register file and benchmarked even with
// this, because the plane split the second pass repeats is three
// instructions against the eleven every row already pays.
func q4matmulCols8(out *tensai.Matrix, xus [][]uint8, xqs [][]uint32, sxs []tensai.Float, gsums [][]int32, r0 int, qw []uint8, scale []tensai.Float, sm []uint32, group, cols, lo, hi int) {
	q4matmulCols4(out, xus[:4], xqs[:4], sxs[:4], gsums[:4], r0, qw, scale, sm, group, cols, lo, hi)
	q4matmulCols4(out, xus[4:8], xqs[4:8], sxs[4:8], gsums[4:8], r0+4, qw, scale, sm, group, cols, lo, hi)
}

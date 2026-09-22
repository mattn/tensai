//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// The column tile's two passes at four lanes, the arm64 counterpart of
// quantcols_simd.go. A lane is a column, so neither pass reduces across
// lanes, the running state stays in registers for a whole row sweep,
// and the tail columns take the portable body.

// maxAbsCols keeps the running per-column maximum magnitude over rows
// rows of a tile.
func maxAbsCols(maxAbs, data []tensai.Float, stride, rows int) {
	w := len(maxAbs)
	if w < 4 {
		maxAbsColsScalar(maxAbs, data, stride, rows)
		return
	}
	n := w &^ 3
	for c := 0; c < n; c += 4 {
		m := simd.LoadF32x4(maxAbs[c:])
		for i := 0; i < rows; i++ {
			m = m.Max(simd.LoadF32x4(data[i*stride+c:]).Abs())
		}
		simd.StoreF32x4(m, maxAbs[c:])
	}
	if n < w {
		maxAbsColsScalar(maxAbs[n:], data[n:], stride, rows)
	}
}

// quantCols rounds rows rows of a tile half away from zero into dst,
// row-major and w wide, and accumulates the per-column sums, with the
// scalar body's rounding: the magnitude nudged by a half and truncated,
// the sign restored through an arithmetic-shift mask. The product and
// the nudge stay two separate roundings, which is what keeps arm64 from
// fusing them into an FMA and answering something the portable build
// does not.
func quantCols(dst []int8, data, inv []tensai.Float, sums []int32, stride, rows, w int) {
	if w < 4 {
		quantColsScalar(dst, data, inv, sums, stride, rows, w)
		return
	}
	half := archsimd.BroadcastFloat32x4(0.5)
	n := w &^ 3
	var buf [4]int32
	for c := 0; c < n; c += 4 {
		ivv := simd.LoadF32x4(inv[c:])
		acc := simd.LoadI32x4(sums[c:])
		for i := 0; i < rows; i++ {
			f := simd.LoadF32x4(data[i*stride+c:]).Mul(ivv)
			q := f.Abs().Add(half).ConvertToInt32()
			s := f.ToBits().BitsToInt32().ShiftAllRight(31)
			q = q.Xor(s).Sub(s)
			acc = acc.Add(q)
			simd.StoreI32x4(q, buf[:])
			out := dst[i*w+c : i*w+c+4 : i*w+c+4]
			out[0] = int8(buf[0])
			out[1] = int8(buf[1])
			out[2] = int8(buf[2])
			out[3] = int8(buf[3])
		}
		simd.StoreI32x4(acc, sums[c:])
	}
	if n < w {
		quantColsScalar(dst[n:], data[n:], inv[n:], sums[n:], stride, rows, w)
	}
}

// quant4Cols is quantCols for the nibble range: the same rounding, then
// the asymmetric clamp the four-bit codes need, and no column sums.
func quant4Cols(dst []int8, data, inv []tensai.Float, stride, rows, w int) {
	if w < 4 {
		quant4ColsScalar(dst, data, inv, stride, rows, w)
		return
	}
	half := archsimd.BroadcastFloat32x4(0.5)
	lo := archsimd.BroadcastInt32x4(-8)
	hi := archsimd.BroadcastInt32x4(7)
	n := w &^ 3
	var buf [4]int32
	for c := 0; c < n; c += 4 {
		ivv := simd.LoadF32x4(inv[c:])
		for i := 0; i < rows; i++ {
			f := simd.LoadF32x4(data[i*stride+c:]).Mul(ivv)
			q := f.Abs().Add(half).ConvertToInt32()
			s := f.ToBits().BitsToInt32().ShiftAllRight(31)
			simd.StoreI32x4(q.Xor(s).Sub(s).Max(lo).Min(hi), buf[:])
			out := dst[i*w+c : i*w+c+4 : i*w+c+4]
			out[0] = int8(buf[0])
			out[1] = int8(buf[1])
			out[2] = int8(buf[2])
			out[3] = int8(buf[3])
		}
	}
	if n < w {
		quant4ColsScalar(dst[n:], data[n:], inv[n:], stride, rows, w)
	}
}

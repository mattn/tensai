//go:build goexperiment.simd && amd64

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// The column tile's two passes, eight columns at a time. A lane is a
// column here rather than a position in a row, so neither pass reduces
// across lanes: the running maximum and the running sum stay in place,
// the rows stream past, and the tail columns take the scalar body the
// portable build runs.

// maxAbsCols keeps the running per-column maximum magnitude over rows
// rows of a tile. Clearing the sign bit is the absolute value, as in
// quantizeActsInto. Four columns' worth of accumulator covers the full
// tile, so the maxima stay in registers for the whole row sweep.
func maxAbsCols(maxAbs, data []tensai.Float, stride, rows int) {
	w := len(maxAbs)
	if !simd.HasAVX2 || w < 8 {
		maxAbsColsScalar(maxAbs, data, stride, rows)
		return
	}
	mSign := archsimd.BroadcastInt32x8(0x7fffffff)
	n := w &^ 7
	for c := 0; c < n; c += 8 {
		m := simd.LoadF32x8(maxAbs[c:])
		for i := 0; i < rows; i++ {
			v := simd.LoadF32x8(data[i*stride+c:]).AsInt32x8().And(mSign).AsFloat32x8()
			m = m.Max(v)
		}
		simd.StoreF32x8(m, maxAbs[c:])
	}
	archsimd.ClearAVXUpperBits()
	if n < w {
		maxAbsColsScalar(maxAbs[n:], data[n:], stride, rows)
	}
}

// quantCols rounds rows rows of a tile half away from zero into dst,
// row-major and w wide, and accumulates the per-column sums. The
// rounding is the scalar body's: the magnitude times the reciprocal,
// nudged by a half and truncated, with the sign put back through an
// arithmetic-shift mask, which is what keeps the vector bytes equal to
// the ones the column-at-a-time loop wrote.
func quantCols(dst []int8, data, inv []tensai.Float, sums []int32, stride, rows, w int) {
	if !simd.HasAVX2 || w < 8 {
		quantColsScalar(dst, data, inv, sums, stride, rows, w)
		return
	}
	mSign := archsimd.BroadcastInt32x8(0x7fffffff)
	half := archsimd.BroadcastFloat32x8(0.5)
	n := w &^ 7
	var buf [8]int32
	for c := 0; c < n; c += 8 {
		ivv := simd.LoadF32x8(inv[c:])
		acc := simd.LoadI32x8(sums[c:])
		for i := 0; i < rows; i++ {
			f := simd.LoadF32x8(data[i*stride+c:]).Mul(ivv)
			q := f.AsInt32x8().And(mSign).AsFloat32x8().Add(half).ConvertToInt32()
			s := f.AsInt32x8().ShiftAllRight(31)
			q = q.Xor(s).Sub(s)
			acc = acc.Add(q)
			simd.StoreI32x8(q, buf[:])
			out := dst[i*w+c : i*w+c+8 : i*w+c+8]
			out[0] = int8(buf[0])
			out[1] = int8(buf[1])
			out[2] = int8(buf[2])
			out[3] = int8(buf[3])
			out[4] = int8(buf[4])
			out[5] = int8(buf[5])
			out[6] = int8(buf[6])
			out[7] = int8(buf[7])
		}
		simd.StoreI32x8(acc, sums[c:])
	}
	archsimd.ClearAVXUpperBits()
	if n < w {
		quantColsScalar(dst[n:], data[n:], inv[n:], sums[n:], stride, rows, w)
	}
}

// quant4Cols is quantCols for the nibble range: the same rounding, then
// the asymmetric clamp the four-bit codes need, and no column sums --
// the nibble kernels fold the activation offset through the group sums
// instead.
func quant4Cols(dst []int8, data, inv []tensai.Float, stride, rows, w int) {
	if !simd.HasAVX2 || w < 8 {
		quant4ColsScalar(dst, data, inv, stride, rows, w)
		return
	}
	mSign := archsimd.BroadcastInt32x8(0x7fffffff)
	half := archsimd.BroadcastFloat32x8(0.5)
	lo := archsimd.BroadcastInt32x8(-8)
	hi := archsimd.BroadcastInt32x8(7)
	n := w &^ 7
	var buf [8]int32
	for c := 0; c < n; c += 8 {
		ivv := simd.LoadF32x8(inv[c:])
		for i := 0; i < rows; i++ {
			f := simd.LoadF32x8(data[i*stride+c:]).Mul(ivv)
			q := f.AsInt32x8().And(mSign).AsFloat32x8().Add(half).ConvertToInt32()
			s := f.AsInt32x8().ShiftAllRight(31)
			simd.StoreI32x8(q.Xor(s).Sub(s).Max(lo).Min(hi), buf[:])
			out := dst[i*w+c : i*w+c+8 : i*w+c+8]
			out[0] = int8(buf[0])
			out[1] = int8(buf[1])
			out[2] = int8(buf[2])
			out[3] = int8(buf[3])
			out[4] = int8(buf[4])
			out[5] = int8(buf[5])
			out[6] = int8(buf[6])
			out[7] = int8(buf[7])
		}
	}
	archsimd.ClearAVXUpperBits()
	if n < w {
		quant4ColsScalar(dst[n:], data[n:], inv[n:], stride, rows, w)
	}
}

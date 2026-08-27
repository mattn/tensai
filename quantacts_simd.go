//go:build goexperiment.simd && amd64

package tensai

import (
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// quantizeActsInto vectorizes the 7-bit activation quantizer with the
// scalar path's exact rounding: |v|*inv + 0.5 truncates (Go's float to
// int conversion and CVTTPS2DQ agree), the sign restores through the
// arithmetic-shift mask, and the clamp caps the magnitude at 63. The
// max-abs scan and the quantizing pass each run eight lanes wide; only
// the byte narrowing stays scalar (the API has no saturating pack).
func quantizeActsInto(x []Float, xu []uint8) Float {
	if !simd.HasAVX2 || len(x) < 16 {
		return quantizeActsScalar(x, xu)
	}
	vecEnd := len(x) &^ 7
	// Clearing the sign bit is Abs; the method only exists from Go 1.27.
	mSign := archsimd.BroadcastInt32x8(0x7fffffff)
	m := archsimd.BroadcastFloat32x8(0)
	for i := 0; i < vecEnd; i += 8 {
		m = m.Max(simd.LoadF32x8(x[i:]).AsInt32x8().And(mSign).AsFloat32x8())
	}
	var mb [8]Float
	simd.StoreF32x8(m, mb[:])
	var maxAbs Float
	for _, v := range mb {
		if v > maxAbs {
			maxAbs = v
		}
	}
	for _, v := range x[vecEnd:] {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	sx := maxAbs / 63
	if sx == 0 {
		for i := range x {
			xu[i] = 64
		}
		archsimd.ClearAVXUpperBits()
		return 0
	}
	inv := archsimd.BroadcastFloat32x8(1 / sx)
	half := archsimd.BroadcastFloat32x8(0.5)
	c63 := archsimd.BroadcastInt32x8(63)
	c64 := archsimd.BroadcastInt32x8(64)
	var buf [8]int32
	for i := 0; i < vecEnd; i += 8 {
		f := simd.LoadF32x8(x[i:])
		iv := f.AsInt32x8().And(mSign).AsFloat32x8().Mul(inv).Add(half).ConvertToInt32().Min(c63)
		s := f.AsInt32x8().ShiftAllRight(31)
		simd.StoreI32x8(iv.Xor(s).Sub(s).Add(c64), buf[:])
		xu[i] = uint8(buf[0])
		xu[i+1] = uint8(buf[1])
		xu[i+2] = uint8(buf[2])
		xu[i+3] = uint8(buf[3])
		xu[i+4] = uint8(buf[4])
		xu[i+5] = uint8(buf[5])
		xu[i+6] = uint8(buf[6])
		xu[i+7] = uint8(buf[7])
	}
	archsimd.ClearAVXUpperBits()
	invs := 1 / sx
	for i, v := range x[vecEnd:] {
		f := v * invs
		if f >= 0 {
			f += 0.5
		} else {
			f -= 0.5
		}
		n := int(f)
		if n < -63 {
			n = -63
		} else if n > 63 {
			n = 63
		}
		xu[vecEnd+i] = uint8(n + 64)
	}
	return sx
}

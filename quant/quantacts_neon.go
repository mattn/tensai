//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"simd/archsimd"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/simd"
)

// quantizeActsInto is the NEON activation quantizer, the arm64 counterpart
// of quantacts_simd.go: one pass for the row's largest magnitude, one to
// round each value half away from zero. The round is the scalar body's,
// truncation after a 0.5 nudge on the magnitude, with the sign put back
// through the float's sign bit, so the bytes match it exactly.
func quantizeActsInto(x []tensai.Float, xu []uint8) tensai.Float {
	if len(x) < 16 {
		return quantizeActsScalar(x, xu)
	}
	vecEnd := len(x) &^ 15
	m0 := archsimd.BroadcastFloat32x4(0)
	m1 := m0
	for i := 0; i < vecEnd; i += 16 {
		m0 = m0.Max(simd.LoadF32x4(x[i:]).Abs()).Max(simd.LoadF32x4(x[i+4:]).Abs())
		m1 = m1.Max(simd.LoadF32x4(x[i+8:]).Abs()).Max(simd.LoadF32x4(x[i+12:]).Abs())
	}
	maxAbs := m0.Max(m1).ReduceMax()
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
		return 0
	}
	inv := archsimd.BroadcastFloat32x4(1 / sx)
	half := archsimd.BroadcastFloat32x4(0.5)
	c63 := archsimd.BroadcastInt32x4(63)
	c64 := archsimd.BroadcastInt32x4(64)
	round := func(f archsimd.Float32x4) archsimd.Int32x4 {
		iv := f.Abs().Mul(inv).Add(half).ConvertToInt32().Min(c63)
		s := f.ToBits().BitsToInt32().ShiftAllRight(31)
		return iv.Xor(s).Sub(s).Add(c64)
	}
	var buf [16]int32
	for i := 0; i < vecEnd; i += 16 {
		simd.StoreI32x4(round(simd.LoadF32x4(x[i:])), buf[:])
		simd.StoreI32x4(round(simd.LoadF32x4(x[i+4:])), buf[4:])
		simd.StoreI32x4(round(simd.LoadF32x4(x[i+8:])), buf[8:])
		simd.StoreI32x4(round(simd.LoadF32x4(x[i+12:])), buf[12:])
		o := xu[i : i+16 : i+16]
		for k, v := range buf {
			o[k] = uint8(v)
		}
	}
	invs := 1 / sx
	for i, v := range x[vecEnd:] {
		// The conversion keeps the product and the nudge as two roundings,
		// which the vector bodies also do; arm64 would otherwise fuse them.
		f := tensai.Float(v * invs)
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

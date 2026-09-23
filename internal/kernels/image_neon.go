//go:build goexperiment.simd && arm64 && go1.27

package kernels

import (
	"math"
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// Sum64 adds float32 values with float64 accumulation.
func Sum64(x []float32) float64 {
	var a, b archsimd.Float64x2
	n := len(x) &^ 3
	for i := 0; i < n; i += 4 {
		v := simd.LoadF32x4(x[i:])
		a = a.Add(v.ConvertLo2ToFloat64())
		b = b.Add(v.HiToLo().ConvertLo2ToFloat64())
	}
	c := a.Add(b)
	return c.GetElem(0) + c.GetElem(1) + sum64Generic(x[n:])
}

// SquaredDeviations64 sums (x-mean)^2 in float64; mean=0 gives squared norm.
func SquaredDeviations64(x []float32, mean float64) float64 {
	var a, b archsimd.Float64x2
	m := archsimd.BroadcastFloat64x2(mean)
	n := len(x) &^ 3
	for i := 0; i < n; i += 4 {
		v := simd.LoadF32x4(x[i:])
		lo, hi := v.ConvertLo2ToFloat64().Sub(m), v.HiToLo().ConvertLo2ToFloat64().Sub(m)
		a = a.Add(lo.Mul(lo))
		b = b.Add(hi.Mul(hi))
	}
	c := a.Add(b)
	return c.GetElem(0) + c.GetElem(1) + squaredDeviations64Generic(x[n:], mean)
}
func normalize64(dst, src []float32, mean, inv float64) {
	m, s := archsimd.BroadcastFloat64x2(mean), archsimd.BroadcastFloat64x2(inv)
	n := len(src) &^ 3
	for i := 0; i < n; i += 4 {
		v := simd.LoadF32x4(src[i:])
		lo := v.ConvertLo2ToFloat64().Sub(m).Mul(s).ConvertToFloat32().ToBits().ReshapeToUint64s()
		hi := v.HiToLo().ConvertLo2ToFloat64().Sub(m).Mul(s).ConvertToFloat32().ToBits().ReshapeToUint64s()
		simd.StoreF32x4(lo.InterleaveLo(hi).ReshapeToUint32s().BitsToFloat32(), dst[i:])
	}
	normalize64Generic(dst[n:], src[n:], mean, inv)
}

// ScaleWeights computes x[i] = x[i] * scale * (w[i] + bias), in place.
func ScaleWeights(x, w []float32, scale, bias float32) {
	s, b := archsimd.BroadcastFloat32x4(scale), archsimd.BroadcastFloat32x4(bias)
	n := len(x) &^ 3
	for i := 0; i < n; i += 4 {
		simd.StoreF32x4(simd.LoadF32x4(x[i:]).Mul(s).Mul(simd.LoadF32x4(w[i:]).Add(b)), x[i:])
	}
	scaleWeightsGeneric(x[n:], w[n:], scale, bias)
}

// MulAddSlice computes dst += x*y; dst may alias either input.
func MulAddSlice(dst, x, y []float32) {
	n := len(dst) &^ 3
	for i := 0; i < n; i += 4 {
		simd.StoreF32x4(simd.LoadF32x4(x[i:]).Mul(simd.LoadF32x4(y[i:])).Add(simd.LoadF32x4(dst[i:])), dst[i:])
	}
	mulAddSliceGeneric(dst[n:], x[n:], y[n:])
}

// RopePairs rotates adjacent pairs. cos and sin each have len(x)/2 elements.
func RopePairs(x, cos, sin []float32) {
	signs := simd.LoadF32x4([]float32{-1, 1, -1, 1})
	n := len(cos) &^ 1
	for j := 0; j < n; j += 2 {
		c0 := simd.LoadF32x4Part(cos[j:]).ToBits()
		s0 := simd.LoadF32x4Part(sin[j:]).ToBits()
		c := c0.InterleaveLo(c0).BitsToFloat32()
		s := s0.InterleaveLo(s0).BitsToFloat32().Mul(signs)
		v := simd.LoadF32x4(x[2*j:])
		bits := v.ToBits()
		swapped := bits.ConcatOdd(bits).InterleaveLo(bits.ConcatEven(bits)).BitsToFloat32()
		simd.StoreF32x4(v.Mul(c).Add(swapped.Mul(s)), x[2*j:])
	}
	ropePairsGeneric(x[2*n:], cos[n:], sin[n:])
}

// RopeSplit rotates the two halves of x against each other.
func RopeSplit(x, cos, sin []float32) {
	half := len(x) / 2
	n := half &^ 3
	for j := 0; j < n; j += 4 {
		a, b := simd.LoadF32x4(x[j:]), simd.LoadF32x4(x[half+j:])
		c, s := simd.LoadF32x4(cos[j:]), simd.LoadF32x4(sin[j:])
		simd.StoreF32x4(a.Mul(c).Sub(b.Mul(s)), x[j:])
		simd.StoreF32x4(b.Mul(c).Add(a.Mul(s)), x[half+j:])
	}
	for j := n; j < half; j++ {
		a, b := x[j], x[half+j]
		x[j] = a*cos[j] - b*sin[j]
		x[half+j] = b*cos[j] + a*sin[j]
	}
}

// GeluTanh applies the tanh approximation of GELU in place.
func GeluTanh(x []float32) {
	one := archsimd.BroadcastFloat32x4(1)
	cube, inner := archsimd.BroadcastFloat32x4(geluTanhCube), archsimd.BroadcastFloat32x4(-geluTanhInner)
	map4(x, x, func(v archsimd.Float32x4) archsimd.Float32x4 {
		y := v.Mul(v).Mul(v).Mul(cube).Add(v).Mul(inner)
		return v.Div(one.Add(vexpf4(y)))
	})
}
func maxSlice(x []float32) float32 {
	a := archsimd.BroadcastFloat32x4(float32(math.Inf(-1)))
	n := len(x) &^ 3
	for i := 0; i < n; i += 4 {
		a = a.Max(simd.LoadF32x4(x[i:]))
	}
	var lanes [4]float32
	simd.StoreF32x4(a, lanes[:])
	return max(maxSliceGeneric(lanes[:]), maxSliceGeneric(x[n:]))
}

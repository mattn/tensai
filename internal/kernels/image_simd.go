//go:build goexperiment.simd && amd64

package kernels

import (
	"math"
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// Sum64 adds float32 values with float64 accumulation.
func Sum64(x []float32) float64 {
	if !simd.HasAVX2 {
		return sum64Generic(x)
	}
	var a archsimd.Float64x4
	n := len(x) &^ 3
	for i := 0; i < n; i += 4 {
		a = a.Add(simd.LoadF32x4(x[i:]).ConvertToFloat64())
	}
	var lanes [4]float64
	simd.StoreF64x4(a, lanes[:])
	archsimd.ClearAVXUpperBits()
	return lanes[0] + lanes[1] + lanes[2] + lanes[3] + sum64Generic(x[n:])
}

// SquaredDeviations64 sums (x-mean)^2 in float64; mean=0 gives squared norm.
func SquaredDeviations64(x []float32, mean float64) float64 {
	if !simd.HasAVX2 {
		return squaredDeviations64Generic(x, mean)
	}
	var a archsimd.Float64x4
	m := archsimd.BroadcastFloat64x4(mean)
	n := len(x) &^ 3
	for i := 0; i < n; i += 4 {
		d := simd.LoadF32x4(x[i:]).ConvertToFloat64().Sub(m)
		a = a.Add(d.Mul(d))
	}
	var lanes [4]float64
	simd.StoreF64x4(a, lanes[:])
	archsimd.ClearAVXUpperBits()
	return lanes[0] + lanes[1] + lanes[2] + lanes[3] + squaredDeviations64Generic(x[n:], mean)
}
func normalize64(dst, src []float32, mean, inv float64) {
	if !simd.HasAVX2 {
		normalize64Generic(dst, src, mean, inv)
		return
	}
	m, s := archsimd.BroadcastFloat64x4(mean), archsimd.BroadcastFloat64x4(inv)
	n := len(src) &^ 3
	for i := 0; i < n; i += 4 {
		simd.StoreF32x4(simd.LoadF32x4(src[i:]).ConvertToFloat64().Sub(m).Mul(s).ConvertToFloat32(), dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	normalize64Generic(dst[n:], src[n:], mean, inv)
}

// ScaleWeights computes x[i] = x[i] * scale * (w[i] + bias), in place.
func ScaleWeights(x, w []float32, scale, bias float32) {
	if !simd.HasAVX2 {
		scaleWeightsGeneric(x, w, scale, bias)
		return
	}
	s, b := archsimd.BroadcastFloat32x8(scale), archsimd.BroadcastFloat32x8(bias)
	n := len(x) &^ 7
	for i := 0; i < n; i += 8 {
		simd.StoreF32x8(simd.LoadF32x8(x[i:]).Mul(s).Mul(simd.LoadF32x8(w[i:]).Add(b)), x[i:])
	}
	archsimd.ClearAVXUpperBits()
	scaleWeightsGeneric(x[n:], w[n:], scale, bias)
}

// MulAddSlice computes dst += x*y; dst may alias either input.
func MulAddSlice(dst, x, y []float32) {
	if !simd.HasAVX2 {
		mulAddSliceGeneric(dst, x, y)
		return
	}
	n := len(dst) &^ 7
	for i := 0; i < n; i += 8 {
		simd.StoreF32x8(simd.LoadF32x8(x[i:]).Mul(simd.LoadF32x8(y[i:])).Add(simd.LoadF32x8(dst[i:])), dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	mulAddSliceGeneric(dst[n:], x[n:], y[n:])
}

// RopePairs rotates adjacent pairs. cos and sin each have len(x)/2 elements.
func RopePairs(x, cos, sin []float32) {
	if !simd.HasAVX2 {
		ropePairsGeneric(x, cos, sin)
		return
	}
	dup := simd.LoadU32x8([]uint32{0, 0, 1, 1, 2, 2, 3, 3})
	swap := simd.LoadU32x8([]uint32{1, 0, 3, 2, 5, 4, 7, 6})
	signs := simd.LoadF32x8([]float32{-1, 1, -1, 1, -1, 1, -1, 1})
	n := len(cos) &^ 3
	for j := 0; j < n; j += 4 {
		c := simd.LoadF32x8Part(cos[j:]).Permute(dup)
		s := simd.LoadF32x8Part(sin[j:]).Permute(dup).Mul(signs)
		v := simd.LoadF32x8(x[2*j:])
		simd.StoreF32x8(v.Mul(c).Add(v.Permute(swap).Mul(s)), x[2*j:])
	}
	archsimd.ClearAVXUpperBits()
	ropePairsGeneric(x[2*n:], cos[n:], sin[n:])
}

// RopeSplit rotates the two halves of x against each other.
func RopeSplit(x, cos, sin []float32) {
	if !simd.HasAVX2 {
		ropeSplitGeneric(x, cos, sin)
		return
	}
	half := len(x) / 2
	n := half &^ 7
	for j := 0; j < n; j += 8 {
		a, b := simd.LoadF32x8(x[j:]), simd.LoadF32x8(x[half+j:])
		c, s := simd.LoadF32x8(cos[j:]), simd.LoadF32x8(sin[j:])
		simd.StoreF32x8(a.Mul(c).Sub(b.Mul(s)), x[j:])
		simd.StoreF32x8(b.Mul(c).Add(a.Mul(s)), x[half+j:])
	}
	archsimd.ClearAVXUpperBits()
	for j := n; j < half; j++ {
		a, b := x[j], x[half+j]
		x[j] = a*cos[j] - b*sin[j]
		x[half+j] = b*cos[j] + a*sin[j]
	}
}

// GeluTanh applies the tanh approximation of GELU in place.
func GeluTanh(x []float32) {
	if !simd.HasAVX2 {
		geluTanhGeneric(x)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	cube, inner := archsimd.BroadcastFloat32x8(geluTanhCube), archsimd.BroadcastFloat32x8(-geluTanhInner)
	mapSlices(x, x, func(v archsimd.Float32x8) archsimd.Float32x8 {
		y := v.Mul(v).Mul(v).Mul(cube).Add(v).Mul(inner)
		return v.Div(one.Add(vexpf(y)))
	})
}
func maxSlice(x []float32) float32 {
	if !simd.HasAVX2 {
		return maxSliceGeneric(x)
	}
	a := archsimd.BroadcastFloat32x8(float32(math.Inf(-1)))
	n := len(x) &^ 7
	for i := 0; i < n; i += 8 {
		a = a.Max(simd.LoadF32x8(x[i:]))
	}
	var lanes [8]float32
	simd.StoreF32x8(a, lanes[:])
	archsimd.ClearAVXUpperBits()
	return max(maxSliceGeneric(lanes[:]), maxSliceGeneric(x[n:]))
}

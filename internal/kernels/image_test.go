package kernels

import (
	"fmt"
	"math"
	"testing"
)

func closeImageSlices(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	for i, v := range want {
		d := math.Abs(float64(got[i]) - float64(v))
		if math.IsNaN(d) || d > tol*(1+math.Abs(float64(v))) {
			t.Fatalf("element %d: got %g want %g (difference %g)", i, got[i], v, d)
		}
	}
}
func TestImageKernels(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 127, 128, 129, 4096} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			src, w := make([]float32, n), make([]float32, n)
			for i := range src {
				src[i] = float32(math.Sin(float64(i)*0.37) * 6)
				w[i] = float32(math.Cos(float64(i) * 0.23))
			}
			got, want := make([]float32, n), make([]float32, n)
			check := func(f, g func([]float32)) {
				t.Helper()
				copy(got, src)
				copy(want, src)
				f(got)
				g(want)
				closeImageSlices(t, got, want, 2e-6)
			}
			check(func(x []float32) { ScaleWeights(x, w, 0.37, 1) }, func(x []float32) { scaleWeightsGeneric(x, w, 0.37, 1) })
			check(func(x []float32) { ScaleWeights(x, w, 0.37, 0) }, func(x []float32) { scaleWeightsGeneric(x, w, 0.37, 0) })
			check(func(x []float32) { MulAddSlice(x, x, w) }, func(x []float32) { mulAddSliceGeneric(x, x, w) })
			check(GeluTanh, geluTanhGeneric)
			if n == 0 {
				return
			}
			for _, offset := range []float32{0, 10000} {
				for i := range src {
					src[i] += offset
				}
				mean := sum64Generic(src) / float64(n)
				variance := squaredDeviations64Generic(src, mean)
				if math.Abs(Sum64(src)-sum64Generic(src)) > 1e-10*(1+math.Abs(sum64Generic(src))) {
					t.Fatal("sum lost precision")
				}
				if math.Abs(SquaredDeviations64(src, mean)-variance) > 1e-10*(1+variance) {
					t.Fatal("variance lost precision")
				}
				inv := 1 / math.Sqrt(variance/float64(n)+1e-6)
				normalize64Generic(want, src, mean, inv)
				LayerNorm64(got, src, 1e-6)
				closeImageSlices(t, got, want, 2e-6)
				copy(got, src)
				LayerNorm64(got, got, 1e-6)
				closeImageSlices(t, got, want, 2e-6)
			}
			copy(got, src)
			Softmax(got)
			m := maxSliceGeneric(src)
			var sum float64
			for i, v := range src {
				want[i] = float32(math.Exp(float64(v - m)))
				sum += float64(want[i])
			}
			for i := range want {
				want[i] /= float32(sum)
			}
			closeImageSlices(t, got, want, 2e-6)
			if math.Abs(Sum64(got)-1) > 1e-6 {
				t.Fatal("softmax does not sum to one")
			}
		})
	}
}
func TestImageNormDegenerate(t *testing.T) {
	for _, value := range []float32{0, 1, 1e20, 1e-20} {
		x := make([]float32, 129)
		for i := range x {
			x[i] = value
		}
		LayerNorm64(x, x, 1e-6)
		for _, v := range x {
			if v != 0 {
				t.Fatalf("constant input %g normalized to %g", value, v)
			}
		}
	}
}
func TestImageRope(t *testing.T) {
	for _, pairs := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 64, 65} {
		for _, split := range []bool{false, true} {
			t.Run(fmt.Sprintf("pairs%d_split%v", pairs, split), func(t *testing.T) {
				got, want := make([]float32, 2*pairs), make([]float32, 2*pairs)
				cos, sin := make([]float32, pairs), make([]float32, pairs)
				for i := range got {
					got[i] = float32(math.Sin(float64(i)*0.13) * 3)
				}
				copy(want, got)
				for j := range cos {
					cos[j] = float32(math.Cos(float64(j) * 0.3))
					sin[j] = float32(math.Sin(float64(j) * 0.3))
				}
				if split {
					RopeSplit(got, cos, sin)
					ropeSplitGeneric(want, cos, sin)
				} else {
					RopePairs(got, cos, sin)
					ropePairsGeneric(want, cos, sin)
				}
				closeImageSlices(t, got, want, 2e-6)
			})
		}
	}
}

var imageBenchmarkSum float64

func BenchmarkImageKernels(b *testing.B) {
	const n = 4096
	src, w := make([]float32, n), make([]float32, n)
	for i := range src {
		src[i] = float32(math.Sin(float64(i) * 0.13))
		w[i] = float32(math.Cos(float64(i) * 0.07))
	}
	dst := make([]float32, n)
	cases := []struct {
		name           string
		scalar, vector func()
	}{
		{"LayerNorm", func() {
			mean := sum64Generic(src) / n
			inv := 1 / math.Sqrt(squaredDeviations64Generic(src, mean)/n+1e-6)
			normalize64Generic(dst, src, mean, inv)
		}, func() { LayerNorm64(dst, src, 1e-6) }},
		{"RMSStats", func() { imageBenchmarkSum = squaredDeviations64Generic(src, 0) }, func() { imageBenchmarkSum = SquaredDeviations64(src, 0) }},
		{"Modulate", func() { copy(dst, src); scaleWeightsGeneric(dst, w, 1, 1) }, func() { copy(dst, src); ScaleWeights(dst, w, 1, 1) }},
		{"Gate", func() { copy(dst, src); mulAddSliceGeneric(dst, src, w) }, func() { copy(dst, src); MulAddSlice(dst, src, w) }},
		{"GELU", func() { copy(dst, src); geluTanhGeneric(dst) }, func() { copy(dst, src); GeluTanh(dst) }},
		{"Softmax", func() {
			copy(dst, src)
			m := maxSliceGeneric(dst)
			expShiftGeneric(dst, dst, m)
			scaleSliceGeneric(dst, float32(1/sum64Generic(dst)))
		}, func() { copy(dst, src); Softmax(dst) }},
		{"RoPE", func() { copy(dst, src); ropePairsGeneric(dst, src[:n/2], w[:n/2]) }, func() { copy(dst, src); RopePairs(dst, src[:n/2], w[:n/2]) }},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			for _, c := range []struct {
				name string
				f    func()
			}{{"scalar", tc.scalar}, {"SIMD", tc.vector}} {
				b.Run(c.name, func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						c.f()
					}
				})
			}
		})
	}
}

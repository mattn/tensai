package tensai

import (
	"math"
	"testing"
)

// TestKernelAccuracy compares the element-wise kernels (which may be the
// AVX2 polynomial versions, depending on the build) against float64
// references over a wide input range.
func TestKernelAccuracy(t *testing.T) {
	var src []Float
	for x := -30.0; x <= 30.0; x += 0.037 {
		src = append(src, Float(x))
	}
	dst := make([]Float, len(src))

	check := func(name string, got []Float, ref func(float64) float64, tol float64) {
		t.Helper()
		for i, x := range src {
			want := ref(float64(x))
			diff := math.Abs(float64(got[i]) - want)
			if diff > tol*(1+math.Abs(want)) {
				t.Fatalf("%s(%g): got %g want %g", name, x, got[i], want)
			}
		}
	}

	expShift(dst, src, 0)
	check("exp", dst, math.Exp, 2e-6)

	sigmoidFwd(dst, src)
	check("sigmoid", dst, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }, 2e-6)

	tanhFwd(dst, src)
	check("tanh", dst, math.Tanh, 2e-6)

	reluFwd(dst, src)
	check("relu", dst, func(x float64) float64 { return math.Max(0, x) }, 0)

	leakyFwd(dst, src, 0.1)
	check("leakyrelu", dst, func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0.1 * x
	}, 1e-7)
}

// TestKernelTails makes sure the masked tail path (length not a multiple of
// the vector width) writes exactly the requested elements.
func TestKernelTails(t *testing.T) {
	for _, n := range []int{1, 3, 7, 8, 9, 31} {
		src := make([]Float, n)
		for i := range src {
			src[i] = Float(i) - 2
		}
		dst := make([]Float, n+1)
		dst[n] = 42 // canary just past the writable range
		reluFwd(dst[:n], src)
		if dst[n] != 42 {
			t.Fatalf("n=%d: kernel wrote past the slice end", n)
		}
		for i := range src {
			want := src[i]
			if want < 0 {
				want = 0
			}
			if dst[i] != want {
				t.Fatalf("n=%d: dst[%d]=%g want %g", n, i, dst[i], want)
			}
		}
	}
}

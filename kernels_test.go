package tensai

import (
	"math"
	"math/rand"
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

// TestGELUKernelAccuracy compares the (possibly vectorized) GELU kernels
// against a float64 reference.
func TestGELUKernelAccuracy(t *testing.T) {
	var src []Float
	for x := -8.0; x <= 8.0; x += 0.011 {
		src = append(src, Float(x))
	}
	dst := make([]Float, len(src))
	grad := make([]Float, len(src))
	for i := range grad {
		grad[i] = 1
	}

	geluFwd(dst, src)
	for i, x := range src {
		want := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		if diff := math.Abs(float64(dst[i]) - want); diff > 2e-6*(1+math.Abs(want)) {
			t.Fatalf("gelu(%g): got %g want %g", x, dst[i], want)
		}
	}

	geluBwd(dst, grad, src)
	for i, x := range src {
		xf := float64(x)
		want := 0.5*(1+math.Erf(xf/math.Sqrt2)) + xf*math.Exp(-0.5*xf*xf)/math.Sqrt(2*math.Pi)
		if diff := math.Abs(float64(dst[i]) - want); diff > 2e-6*(1+math.Abs(want)) {
			t.Fatalf("gelu'(%g): got %g want %g", x, dst[i], want)
		}
	}
}

// TestLayerNormKernelMatchesGeneric compares the dispatched LayerNorm row
// kernels against the scalar reference, including non-multiple-of-8 tails.
func TestLayerNormKernelMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	for _, cols := range []int{3, 8, 13, 64, 100} {
		src := make([]Float, cols)
		g := make([]Float, cols)
		gamma := make([]Float, cols)
		beta := make([]Float, cols)
		for i := range src {
			src[i] = Float(rng.NormFloat64())
			g[i] = Float(rng.NormFloat64())
			gamma[i] = Float(0.5 + rng.Float64())
			beta[i] = Float(rng.NormFloat64() * 0.1)
		}
		out := make([]Float, cols)
		xhat := make([]Float, cols)
		wantOut := make([]Float, cols)
		wantXhat := make([]Float, cols)

		invStd := lnFwdRow(out, xhat, src, gamma, beta, 1e-5)
		wantInvStd := lnFwdRowGeneric(wantOut, wantXhat, src, gamma, beta, 1e-5)
		if math.Abs(float64(invStd-wantInvStd)) > 1e-4*float64(wantInvStd) {
			t.Fatalf("cols=%d invStd: got %g want %g", cols, invStd, wantInvStd)
		}
		for i := range out {
			if math.Abs(float64(out[i]-wantOut[i])) > 1e-4*(1+math.Abs(float64(wantOut[i]))) {
				t.Fatalf("cols=%d fwd out[%d]: got %g want %g", cols, i, out[i], wantOut[i])
			}
		}

		gradGamma := make([]Float, cols)
		gradBeta := make([]Float, cols)
		wantGradGamma := make([]Float, cols)
		wantGradBeta := make([]Float, cols)
		bwdOut := make([]Float, cols)
		wantBwdOut := make([]Float, cols)
		lnBwdRow(bwdOut, g, wantXhat, gamma, gradGamma, gradBeta, wantInvStd)
		lnBwdRowGeneric(wantBwdOut, g, wantXhat, gamma, wantGradGamma, wantGradBeta, wantInvStd)
		for i := range bwdOut {
			if math.Abs(float64(bwdOut[i]-wantBwdOut[i])) > 1e-4*(1+math.Abs(float64(wantBwdOut[i]))) {
				t.Fatalf("cols=%d bwd out[%d]: got %g want %g", cols, i, bwdOut[i], wantBwdOut[i])
			}
			if math.Abs(float64(gradGamma[i]-wantGradGamma[i])) > 1e-4*(1+math.Abs(float64(wantGradGamma[i]))) {
				t.Fatalf("cols=%d gradGamma[%d]: got %g want %g", cols, i, gradGamma[i], wantGradGamma[i])
			}
			if gradBeta[i] != wantGradBeta[i] {
				t.Fatalf("cols=%d gradBeta[%d]: got %g want %g", cols, i, gradBeta[i], wantGradBeta[i])
			}
		}
	}
}

func TestSiluMul(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, n := range []int{1, 7, 8, 33, 1000} {
		gate := make([]Float, n)
		up := make([]Float, n)
		want := make([]Float, n)
		for i := range gate {
			gate[i] = Float(rng.NormFloat64() * 4)
			up[i] = Float(rng.NormFloat64())
			g := float64(gate[i])
			want[i] = Float(g/(1+math.Exp(-g))) * up[i]
		}
		SiluMul(gate, up)
		for i := range gate {
			if diff := math.Abs(float64(gate[i] - want[i])); diff > 2e-6*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("n=%d idx %d: got %v want %v", n, i, gate[i], want[i])
			}
		}
	}
}

// TestDotVecsAxpys pins the grouped attention helpers to their per-row
// twins bit for bit, across group widths, head sizes, and both the
// vector and sub-16 generic paths.
func TestDotVecsAxpys(t *testing.T) {
	rng := rand.New(rand.NewSource(63))
	for _, d := range []int{8, 16, 64, 128, 130} {
		for nq := 1; nq <= 8; nq++ {
			qs := make([]Float, nq*d)
			k := make([]Float, d)
			for i := range qs {
				qs[i] = Float(rng.NormFloat64())
			}
			for i := range k {
				k[i] = Float(rng.NormFloat64())
			}
			out := make([]Float, nq)
			DotVecs(qs, k, out)
			for i := 0; i < nq; i++ {
				if want := DotVec(qs[i*d:(i+1)*d], k); out[i] != want {
					t.Fatalf("DotVecs d=%d nq=%d row %d: got %v want %v", d, nq, i, out[i], want)
				}
			}

			ws := make([]Float, nq)
			for i := range ws {
				ws[i] = Float(rng.NormFloat64())
			}
			outs := make([]Float, nq*d)
			ref := make([]Float, nq*d)
			for i := range outs {
				outs[i] = Float(rng.NormFloat64())
				ref[i] = outs[i]
			}
			Axpys(ws, k, outs)
			for i := 0; i < nq; i++ {
				Axpy(ws[i], k, ref[i*d:(i+1)*d])
			}
			for i := range outs {
				if outs[i] != ref[i] {
					t.Fatalf("Axpys d=%d nq=%d elem %d: got %v want %v", d, nq, i, outs[i], ref[i])
				}
			}
		}
	}
}

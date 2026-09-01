package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// TestKernelAccuracy compares the element-wise kernels (which may be the
// AVX2 polynomial versions, depending on the build) against float64
// references over a wide input range.
func TestKernelAccuracy(t *testing.T) {
	var src []float32
	for x := -30.0; x <= 30.0; x += 0.037 {
		src = append(src, float32(x))
	}
	dst := make([]float32, len(src))

	check := func(name string, got []float32, ref func(float64) float64, tol float64) {
		t.Helper()
		for i, x := range src {
			want := ref(float64(x))
			diff := math.Abs(float64(got[i]) - want)
			if diff > tol*(1+math.Abs(want)) {
				t.Fatalf("%s(%g): got %g want %g", name, x, got[i], want)
			}
		}
	}

	ExpShift(dst, src, 0)
	check("exp", dst, math.Exp, 2e-6)

	SigmoidFwd(dst, src)
	check("sigmoid", dst, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }, 2e-6)

	TanhFwd(dst, src)
	check("tanh", dst, math.Tanh, 2e-6)

	ReluFwd(dst, src)
	check("relu", dst, func(x float64) float64 { return math.Max(0, x) }, 0)

	LeakyFwd(dst, src, 0.1)
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
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i) - 2
		}
		dst := make([]float32, n+1)
		dst[n] = 42 // canary just past the writable range
		ReluFwd(dst[:n], src)
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
	var src []float32
	for x := -8.0; x <= 8.0; x += 0.011 {
		src = append(src, float32(x))
	}
	dst := make([]float32, len(src))
	grad := make([]float32, len(src))
	for i := range grad {
		grad[i] = 1
	}

	GeluFwd(dst, src)
	for i, x := range src {
		want := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		if diff := math.Abs(float64(dst[i]) - want); diff > 2e-6*(1+math.Abs(want)) {
			t.Fatalf("gelu(%g): got %g want %g", x, dst[i], want)
		}
	}

	GeluBwd(dst, grad, src)
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
		src := make([]float32, cols)
		g := make([]float32, cols)
		gamma := make([]float32, cols)
		beta := make([]float32, cols)
		for i := range src {
			src[i] = float32(rng.NormFloat64())
			g[i] = float32(rng.NormFloat64())
			gamma[i] = float32(0.5 + rng.Float64())
			beta[i] = float32(rng.NormFloat64() * 0.1)
		}
		out := make([]float32, cols)
		xhat := make([]float32, cols)
		wantOut := make([]float32, cols)
		wantXhat := make([]float32, cols)

		invStd := LnFwdRow(out, xhat, src, gamma, beta, 1e-5)
		wantInvStd := lnFwdRowGeneric(wantOut, wantXhat, src, gamma, beta, 1e-5)
		if math.Abs(float64(invStd-wantInvStd)) > 1e-4*float64(wantInvStd) {
			t.Fatalf("cols=%d invStd: got %g want %g", cols, invStd, wantInvStd)
		}
		for i := range out {
			if math.Abs(float64(out[i]-wantOut[i])) > 1e-4*(1+math.Abs(float64(wantOut[i]))) {
				t.Fatalf("cols=%d fwd out[%d]: got %g want %g", cols, i, out[i], wantOut[i])
			}
		}

		gradGamma := make([]float32, cols)
		gradBeta := make([]float32, cols)
		wantGradGamma := make([]float32, cols)
		wantGradBeta := make([]float32, cols)
		bwdOut := make([]float32, cols)
		wantBwdOut := make([]float32, cols)
		LnBwdRow(bwdOut, g, wantXhat, gamma, gradGamma, gradBeta, wantInvStd)
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
		gate := make([]float32, n)
		up := make([]float32, n)
		want := make([]float32, n)
		for i := range gate {
			gate[i] = float32(rng.NormFloat64() * 4)
			up[i] = float32(rng.NormFloat64())
			g := float64(gate[i])
			want[i] = float32(g/(1+math.Exp(-g))) * up[i]
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
			qs := make([]float32, nq*d)
			k := make([]float32, d)
			for i := range qs {
				qs[i] = float32(rng.NormFloat64())
			}
			for i := range k {
				k[i] = float32(rng.NormFloat64())
			}
			out := make([]float32, nq)
			DotVecs(qs, k, out)
			for i := 0; i < nq; i++ {
				if want := DotVec(qs[i*d:(i+1)*d], k); out[i] != want {
					t.Fatalf("DotVecs d=%d nq=%d row %d: got %v want %v", d, nq, i, out[i], want)
				}
			}

			ws := make([]float32, nq)
			for i := range ws {
				ws[i] = float32(rng.NormFloat64())
			}
			outs := make([]float32, nq*d)
			ref := make([]float32, nq*d)
			for i := range outs {
				outs[i] = float32(rng.NormFloat64())
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

func TestSoftmaxBwdAdd(t *testing.T) {
	rng := rand.New(rand.NewSource(81))
	for _, n := range []int{1, 7, 8, 15, 16, 127, 128, 130, 4096} {
		dst := make([]float32, n)
		grad := make([]float32, n)
		y := make([]float32, n)
		var sum float32
		for i := range dst {
			dst[i] = float32(rng.NormFloat64())
			grad[i] = float32(rng.NormFloat64())
			y[i] = float32(rng.Float64())
			sum += y[i]
		}
		for i := range y {
			y[i] /= sum
		}
		want := append([]float32(nil), dst...)
		softmaxBwdAddGeneric(want, grad, y)
		SoftmaxBwdAdd(dst, grad, y)
		for i := range dst {
			if diff := math.Abs(float64(dst[i] - want[i])); diff > 2e-5*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("n=%d idx=%d: got %g want %g", n, i, dst[i], want[i])
			}
		}
	}
}

func TestSGDStepKernelMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	for _, n := range []int{1, 3, 7, 8, 9, 31, 64} {
		w := make([]float32, n)
		vel := make([]float32, n)
		wantW := make([]float32, n)
		wantVel := make([]float32, n)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
			wantW[i] = w[i]
		}
		g := make([]float32, n)
		for step := 0; step < 3; step++ {
			for i := range g {
				g[i] = float32(rng.NormFloat64())
			}
			SGDStep(w, g, vel, 0.9, 0.05)
			sgdStepGeneric(wantW, g, wantVel, 0.9, 0.05)
		}
		for i := range w {
			if math.Abs(float64(w[i]-wantW[i])) > 1e-6*(1+math.Abs(float64(wantW[i]))) {
				t.Fatalf("n=%d w[%d]: got %g want %g", n, i, w[i], wantW[i])
			}
			if math.Abs(float64(vel[i]-wantVel[i])) > 1e-6*(1+math.Abs(float64(wantVel[i]))) {
				t.Fatalf("n=%d vel[%d]: got %g want %g", n, i, vel[i], wantVel[i])
			}
		}
	}
}

func BenchmarkSGDStep4096(b *testing.B) {
	w := make([]float32, 4096)
	g := make([]float32, 4096)
	vel := make([]float32, 4096)
	for i := range w {
		w[i] = float32(i%17) / 17
		g[i] = float32(i%31) / 31
	}
	b.SetBytes(int64(len(w) * 4 * 3))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SGDStep(w, g, vel, 0.9, 0.001)
	}
}

func BenchmarkSoftmaxBackward4096(b *testing.B) {
	dst := make([]float32, 4096)
	grad := make([]float32, 4096)
	y := make([]float32, 4096)
	for i := range y {
		grad[i] = float32(i%17) / 17
		y[i] = (float32(i%31) / 31) / 2048
	}
	b.SetBytes(int64(len(y) * 4 * 3))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxBwdAdd(dst, grad, y)
	}
}

// TestBinarySlices checks the element-wise binary kernels against the
// scalar definition, including the masked tail and the canary just past the
// destination.
func TestBinarySlices(t *testing.T) {
	ops := []struct {
		name string
		fn   func(dst, x, y []float32)
		want func(a, b float32) float32
	}{
		{"add", AddSlices, func(a, b float32) float32 { return a + b }},
		{"sub", SubSlices, func(a, b float32) float32 { return a - b }},
		{"mul", MulSlices, func(a, b float32) float32 { return a * b }},
		{"div", DivSlices, func(a, b float32) float32 { return a / b }},
	}
	for _, op := range ops {
		for _, n := range []int{1, 3, 7, 8, 9, 31, 64} {
			x := make([]float32, n)
			y := make([]float32, n)
			for i := range x {
				x[i] = float32(i) - 3
				y[i] = float32(i)*0.5 + 1 // never zero, so div stays finite
			}
			dst := make([]float32, n+1)
			dst[n] = 42 // canary just past the writable range
			op.fn(dst[:n], x, y)
			if dst[n] != 42 {
				t.Fatalf("%s n=%d: kernel wrote past the slice end", op.name, n)
			}
			for i := range x {
				want := op.want(x[i], y[i])
				if diff := float64(dst[i] - want); diff > 1e-6 || diff < -1e-6 {
					t.Fatalf("%s n=%d: dst[%d]=%g want %g", op.name, n, i, dst[i], want)
				}
			}
		}
	}
}

// GeluMul has to agree with the tanh gelu written out longhand, which is
// what the Gemma models were trained on and what llama.cpp computes.
func TestGeluMulMatchesTanhForm(t *testing.T) {
	gate := []float32{-6, -2.5, -1, -0.25, 0, 0.25, 1, 2.5, 6, 11.5, -11.5}
	up := []float32{1, -2, 0.5, 3, 1, -1, 2, 0.25, -0.5, 1, 1}
	want := make([]float32, len(gate))
	for i, g := range gate {
		const c = 0.7978845608028654 // sqrt(2/pi)
		d := float64(g)
		want[i] = float32(0.5*d*(1+math.Tanh(c*(d+0.044715*d*d*d)))) * up[i]
	}
	got := append([]float32(nil), gate...)
	GeluMul(got, up)
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 3e-6*(1+math.Abs(float64(want[i]))) {
			t.Errorf("GeluMul(%v, %v) = %v, want %v", gate[i], up[i], got[i], want[i])
		}
	}
	// The generic path is the same function on machines without AVX2.
	plain := append([]float32(nil), gate...)
	geluMulGeneric(plain, up)
	for i := range want {
		if diff := math.Abs(float64(plain[i] - want[i])); diff > 3e-6*(1+math.Abs(float64(want[i]))) {
			t.Errorf("geluMulGeneric(%v) = %v, want %v", gate[i], plain[i], want[i])
		}
	}
}

//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// checkClose compares a downloaded device result against a CPU reference.
func checkClose(t *testing.T, name string, got *tensai.Tensor, want []tensai.Float, tol float64) {
	t.Helper()
	if len(got.Data) != len(want) {
		t.Fatalf("%s: got %d elements, want %d", name, len(got.Data), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got.Data[i] - want[i])); diff > tol*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s element %d: gpu=%v cpu=%v", name, i, got.Data[i], want[i])
		}
	}
}

// TestGPUBinaryOps checks the element-wise arithmetic, including the
// cyclic broadcast a bias row uses.
func TestGPUBinaryOps(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(31))

	x := randTensor(rng, 6, 8)
	y := randTensor(rng, 6, 8)
	row := randTensor(rng, 1, 8) // broadcasts over the rows
	for i := range row.Data {
		row.Data[i] += 2 // keep division well conditioned
	}
	gx, gy, grow := upload3(t, g, x, y, row)
	defer gx.Free()
	defer gy.Free()
	defer grow.Free()

	ops := []struct {
		name string
		op   BinOp
		fn   func(a, b tensai.Float) tensai.Float
	}{
		{"add", OpAdd, func(a, b tensai.Float) tensai.Float { return a + b }},
		{"sub", OpSub, func(a, b tensai.Float) tensai.Float { return a - b }},
		{"mul", OpMul, func(a, b tensai.Float) tensai.Float { return a * b }},
		{"div", OpDiv, func(a, b tensai.Float) tensai.Float { return a / b }},
	}
	for _, o := range ops {
		for _, rhs := range []struct {
			name string
			gt   *Tensor
			ct   *tensai.Tensor
		}{{"same", gy, y}, {"broadcast", grow, row}} {
			out, err := gx.Binary(o.op, rhs.gt)
			if err != nil {
				t.Fatalf("%s/%s: %v", o.name, rhs.name, err)
			}
			got, err := out.Download()
			out.Free()
			if err != nil {
				t.Fatal(err)
			}
			want := make([]tensai.Float, len(x.Data))
			for i := range want {
				want[i] = o.fn(x.Data[i], rhs.ct.Data[i%len(rhs.ct.Data)])
			}
			checkClose(t, o.name+"/"+rhs.name, got, want, 1e-5)
		}
	}

	// A shape that does not divide the output cannot be broadcast.
	odd, err := g.Upload(randTensor(rng, 5))
	if err != nil {
		t.Fatal(err)
	}
	defer odd.Free()
	if _, err := gx.Binary(OpAdd, odd); err == nil {
		t.Error("expected a broadcast error")
	}
}

// TestGPUActivations checks each activation and its gradient against the
// CPU kernels the rest of tensai uses.
func TestGPUActivations(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(37))

	x := randTensor(rng, 4, 16)
	grad := randTensor(rng, 4, 16)
	gx, ggrad := upload2(t, g, x, grad)
	defer gx.Free()
	defer ggrad.Free()

	acts := []struct {
		name string
		act  Act
		fwd  func(f tensai.Float) tensai.Float
		bwd  func(f tensai.Float) tensai.Float
	}{
		{"relu", ActReLU,
			func(f tensai.Float) tensai.Float {
				if f > 0 {
					return f
				}
				return 0
			},
			func(f tensai.Float) tensai.Float {
				if f > 0 {
					return 1
				}
				return 0
			}},
		{"tanh", ActTanh,
			kernels.TanhF,
			func(f tensai.Float) tensai.Float { y := kernels.TanhF(f); return 1 - y*y }},
		{"sigmoid", ActSigmoid,
			func(f tensai.Float) tensai.Float { return 1 / (1 + kernels.ExpF(-f)) },
			func(f tensai.Float) tensai.Float {
				y := 1 / (1 + kernels.ExpF(-f))
				return y * (1 - y)
			}},
		{"gelu", ActGELU, kernels.GeluF, kernels.GeluGrad},
	}
	for _, a := range acts {
		out, err := gx.Activate(a.act)
		if err != nil {
			t.Fatalf("%s forward: %v", a.name, err)
		}
		got, err := out.Download()
		out.Free()
		if err != nil {
			t.Fatal(err)
		}
		want := make([]tensai.Float, len(x.Data))
		for i, v := range x.Data {
			want[i] = a.fwd(v)
		}
		checkClose(t, a.name+" forward", got, want, 1e-5)

		dOut, err := gx.ActivateGrad(a.act, ggrad)
		if err != nil {
			t.Fatalf("%s backward: %v", a.name, err)
		}
		gotG, err := dOut.Download()
		dOut.Free()
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range x.Data {
			want[i] = grad.Data[i] * a.bwd(v)
		}
		checkClose(t, a.name+" backward", gotG, want, 1e-5)
	}
}

// TestGPUSumCols checks the reduction a broadcast operand's gradient uses.
func TestGPUSumCols(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(41))

	x := randTensor(rng, 33, 12) // rows not a multiple of the workgroup
	gx, err := g.Upload(x)
	if err != nil {
		t.Fatal(err)
	}
	defer gx.Free()
	out, err := gx.SumCols()
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape[0] != 1 || got.Shape[1] != 12 {
		t.Fatalf("shape %v, want [1 12]", got.Shape)
	}
	want := make([]tensai.Float, 12)
	for r := 0; r < 33; r++ {
		for c := 0; c < 12; c++ {
			want[c] += x.Data[r*12+c]
		}
	}
	checkClose(t, "sumcols", got, want, 1e-5)
}

// TestGPUAdamStep runs several updates on the device and on the CPU kernel
// and checks the weights stay together.
func TestGPUAdamStep(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(43))

	const n = 40
	w := randTensor(rng, n)
	grad := randTensor(rng, n)
	cpuW := append([]tensai.Float(nil), w.Data...)
	cpuM := make([]tensai.Float, n)
	cpuV := make([]tensai.Float, n)

	gw, err := g.Upload(w)
	if err != nil {
		t.Fatal(err)
	}
	defer gw.Free()
	ggrad, err := g.Upload(grad)
	if err != nil {
		t.Fatal(err)
	}
	defer ggrad.Free()
	gm, err := g.Upload(tensai.NewTensor(n))
	if err != nil {
		t.Fatal(err)
	}
	defer gm.Free()
	gv, err := g.Upload(tensai.NewTensor(n))
	if err != nil {
		t.Fatal(err)
	}
	defer gv.Free()

	const beta1, beta2, lr, eps = 0.9, 0.999, 0.01, 1e-8
	for step := 1; step <= 5; step++ {
		rc1 := 1 / (1 - kernels.PowF(beta1, tensai.Float(step)))
		rc2 := 1 / (1 - kernels.PowF(beta2, tensai.Float(step)))
		if err := gw.AdamStep(ggrad, gm, gv, lr, beta1, beta2, rc1, rc2, eps, 0); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		kernels.AdamStep(cpuW, grad.Data, cpuM, cpuV, beta1, beta2, rc1, rc2, lr, eps, 0)
	}
	got, err := gw.Download()
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "adam", got, cpuW, 1e-5)
}

func upload2(t *testing.T, g *Device, a, b *tensai.Tensor) (*Tensor, *Tensor) {
	t.Helper()
	ga, err := g.Upload(a)
	if err != nil {
		t.Fatal(err)
	}
	gb, err := g.Upload(b)
	if err != nil {
		t.Fatal(err)
	}
	return ga, gb
}

func upload3(t *testing.T, g *Device, a, b, c *tensai.Tensor) (*Tensor, *Tensor, *Tensor) {
	t.Helper()
	ga, gb := upload2(t, g, a, b)
	gc, err := g.Upload(c)
	if err != nil {
		t.Fatal(err)
	}
	return ga, gb, gc
}

// TestGPULayerNorm checks the normalization and both of its gradients
// against the CPU kernels the layer package uses.
func TestGPULayerNorm(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(53))

	const rows, cols = 5, 40
	x := randTensor(rng, rows, cols)
	grad := randTensor(rng, rows, cols)
	gain := randTensor(rng, cols)
	bias := randTensor(rng, cols)
	const eps = 1e-5

	gx, ggrad := upload2(t, g, x, grad)
	defer gx.Free()
	defer ggrad.Free()
	ggain, gbias := upload2(t, g, gain, bias)
	defer ggain.Free()
	defer gbias.Free()

	// Forward, against LnFwdRow.
	out, err := gx.LayerNorm(ggain, gbias, eps)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]tensai.Float, rows*cols)
	xhat := make([]tensai.Float, rows*cols)
	invStd := make([]tensai.Float, rows)
	for r := 0; r < rows; r++ {
		lo, hi := r*cols, (r+1)*cols
		invStd[r] = kernels.LnFwdRow(want[lo:hi], xhat[lo:hi], x.Data[lo:hi], gain.Data, bias.Data, eps)
	}
	checkClose(t, "layernorm", got, want, 1e-4)

	// The normalized values on their own.
	xh, err := gx.LayerNormXhat(eps)
	if err != nil {
		t.Fatal(err)
	}
	defer xh.Free()
	gotXhat, err := xh.Download()
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "layernorm xhat", gotXhat, xhat, 1e-4)

	// Backward, against LnBwdRow.
	dx, err := gx.LayerNormGrad(ggrad, ggain, eps)
	if err != nil {
		t.Fatal(err)
	}
	defer dx.Free()
	gotDx, err := dx.Download()
	if err != nil {
		t.Fatal(err)
	}
	wantDx := make([]tensai.Float, rows*cols)
	gradGamma, gradBeta := make([]tensai.Float, cols), make([]tensai.Float, cols)
	for r := 0; r < rows; r++ {
		lo, hi := r*cols, (r+1)*cols
		kernels.LnBwdRow(wantDx[lo:hi], grad.Data[lo:hi], xhat[lo:hi], gain.Data, gradGamma, gradBeta, invStd[r])
	}
	checkClose(t, "layernorm backward", gotDx, wantDx, 1e-4)
}

// TestGPUSoftmaxGrad checks the softmax backward pass against the CPU
// kernel, on a 4-D stack the way attention produces it.
func TestGPUSoftmaxGrad(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(59))

	const b, h, seq = 2, 3, 12
	scores := randTensor(rng, b, h, seq, seq)
	gscores, err := g.Upload(scores)
	if err != nil {
		t.Fatal(err)
	}
	defer gscores.Free()
	y, err := gscores.Softmax()
	if err != nil {
		t.Fatal(err)
	}
	defer y.Free()
	hostY, err := y.Download()
	if err != nil {
		t.Fatal(err)
	}

	grad := randTensor(rng, b, h, seq, seq)
	ggrad, err := g.Upload(grad)
	if err != nil {
		t.Fatal(err)
	}
	defer ggrad.Free()
	dx, err := y.SoftmaxGrad(ggrad)
	if err != nil {
		t.Fatal(err)
	}
	defer dx.Free()
	got, err := dx.Download()
	if err != nil {
		t.Fatal(err)
	}

	want := make([]tensai.Float, len(grad.Data))
	for pos := 0; pos < len(want); pos += seq {
		kernels.SoftmaxBwdAdd(want[pos:pos+seq], grad.Data[pos:pos+seq], hostY.Data[pos:pos+seq])
	}
	checkClose(t, "softmax backward", got, want, 1e-4)
}

// TestGPUPermute checks the axis permutation against the CPU transpose,
// including the (batch, seq, head, dim) -> (batch, head, seq, dim) move
// attention makes.
func TestGPUPermute(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(61))

	cases := []struct {
		shape []int
		perm  []int
	}{
		{[]int{3, 5}, []int{1, 0}},
		{[]int{2, 3, 4}, []int{1, 0, 2}},
		{[]int{2, 6, 3, 8}, []int{0, 2, 1, 3}},
		{[]int{2, 6, 3, 8}, []int{3, 1, 0, 2}},
	}
	for _, c := range cases {
		x := randTensor(rng, c.shape...)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatal(err)
		}
		out, err := gx.Permute(c.perm...)
		if err != nil {
			t.Fatalf("permute %v by %v: %v", c.shape, c.perm, err)
		}
		got, err := out.Download()
		gx.Free()
		out.Free()
		if err != nil {
			t.Fatal(err)
		}
		want, err := x.Transpose(c.perm...)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.Shape {
			if got.Shape[i] != want.Shape[i] {
				t.Fatalf("permute %v by %v: shape %v, want %v", c.shape, c.perm, got.Shape, want.Shape)
			}
		}
		checkClose(t, "permute", got, want.Data, 1e-6)
	}

	// Rank five is beyond what the kernel expresses.
	big, err := g.Upload(randTensor(rng, 2, 2, 2, 2, 2))
	if err != nil {
		t.Fatal(err)
	}
	defer big.Free()
	if _, err := big.Permute(0, 1, 2, 4, 3); err == nil {
		t.Error("expected an unsupported-rank error")
	}
}

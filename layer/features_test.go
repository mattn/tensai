package layer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

func TestDenseForwardBackward(t *testing.T) {
	d := NewDense(2)
	// Manual weights: 2x2 identity-like
	d.weights = tensai.NewMatrix(2, 2)
	d.weights.Set(0, 0, 1)
	d.weights.Set(1, 1, 1)
	d.bias = []tensai.Float{0, 0}
	d.gradW = tensai.NewMatrix(2, 2)
	d.gradB = make([]tensai.Float, 2)

	input := tensai.NewMatrix(1, 2)
	input.Set(0, 0, 3)
	input.Set(0, 1, 5)

	out, err := d.Forward(input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out.At(0, 0) != 3 || out.At(0, 1) != 5 {
		t.Fatalf("forward = [%g %g], want [3 5]", out.At(0, 0), out.At(0, 1))
	}

	grad := tensai.NewMatrix(1, 2)
	grad.Set(0, 0, 1)
	grad.Set(0, 1, 1)
	gradIn, err := d.Backward(grad)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	// gradInput = gradOutput * W^T = [1,1] * I = [1,1]
	if gradIn.At(0, 0) != 1 || gradIn.At(0, 1) != 1 {
		t.Errorf("gradInput = [%g %g], want [1 1]", gradIn.At(0, 0), gradIn.At(0, 1))
	}
	// gradW = input^T * gradOutput = [[3],[5]] * [[1,1]] = [[3,3],[5,5]]
	gw, _ := d.Grads()
	if gw.At(0, 0) != 3 || gw.At(1, 1) != 5 {
		t.Errorf("gradW mismatch: got %g %g", gw.At(0, 0), gw.At(1, 1))
	}
}

func checkLayerGrad(t *testing.T, layer Layer, input *tensai.Matrix, tol float64) {
	t.Helper()
	out, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	coeff := tensai.NewMatrix(out.Rows, out.Cols)
	for i := range coeff.Data {
		coeff.Data[i] = tensai.Float(math.Sin(float64(i)*0.7) + 0.3)
	}
	gradIn, err := layer.Backward(coeff)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	weighted := func(m *tensai.Matrix) float64 {
		var sum float64
		for j := range m.Data {
			sum += float64(coeff.Data[j]) * float64(m.Data[j])
		}
		return sum
	}
	// Consume each forward's output before the next call: layers may reuse
	// their output buffer between training-mode forwards.
	const h = 5e-3
	for i := range input.Data {
		orig := input.Data[i]
		input.Data[i] = orig + h
		outP, err := layer.Forward(input)
		if err != nil {
			t.Fatalf("forward(+h): %v", err)
		}
		lp := weighted(outP)
		input.Data[i] = orig - h
		outM, err := layer.Forward(input)
		if err != nil {
			t.Fatalf("forward(-h): %v", err)
		}
		lm := weighted(outM)
		input.Data[i] = orig
		num := (lp - lm) / (2 * h)
		if diff := math.Abs(num - float64(gradIn.Data[i])); diff > tol*(1+math.Abs(num)) {
			t.Errorf("input grad %d: numeric=%.8f analytic=%.8f", i, num, gradIn.Data[i])
		}
	}
}

func randomInput(rows, cols int, seed int64) *tensai.Matrix {
	rng := rand.New(rand.NewSource(seed))
	m := tensai.NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return m
}

func TestLeakyReLUGradient(t *testing.T) {
	l := NewLeakyReLU(0.1)
	if _, err := l.Init(6, nil); err != nil {
		t.Fatal(err)
	}
	checkLayerGrad(t, l, randomInput(4, 6, 1), 5e-3)
}

func TestGELUGradient(t *testing.T) {
	l := &GELU{}
	if _, err := l.Init(6, nil); err != nil {
		t.Fatal(err)
	}
	checkLayerGrad(t, l, randomInput(4, 6, 31), 5e-3)
}

func TestSoftmaxGradient(t *testing.T) {
	l := &Softmax{}
	if _, err := l.Init(5, nil); err != nil {
		t.Fatal(err)
	}
	checkLayerGrad(t, l, randomInput(3, 5, 2), 5e-3)

	out, err := l.Forward(randomInput(3, 5, 3))
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < out.Rows; r++ {
		var sum tensai.Float
		for c := 0; c < out.Cols; c++ {
			sum += out.At(r, c)
		}
		if math.Abs(float64(sum-1)) > 1e-6 {
			t.Errorf("row %d does not sum to 1: %g", r, sum)
		}
	}
}

func TestBatchNormGradient(t *testing.T) {
	bn := NewBatchNorm()
	if _, err := bn.Init(4, nil); err != nil {
		t.Fatal(err)
	}
	bn.SetTraining(true)
	// Non-trivial gamma/beta so the scale path is exercised too.
	for i := range bn.gamma.Data {
		bn.gamma.Data[i] = 1.5
		bn.beta[i] = -0.25
	}
	checkLayerGrad(t, bn, randomInput(6, 4, 4), 1e-2)
}

func TestLayerNormGradient(t *testing.T) {
	ln := NewLayerNorm()
	if _, err := ln.Init(4, nil); err != nil {
		t.Fatal(err)
	}
	for i := range ln.gamma.Data {
		ln.gamma.Data[i] = tensai.Float(0.8 + 0.2*float64(i))
		ln.beta[i] = tensai.Float(-0.3 + 0.1*float64(i))
	}
	checkLayerGrad(t, ln, randomInput(6, 4, 41), 1e-2)
}

func TestBatchNormEvalUsesRunningStats(t *testing.T) {
	bn := NewBatchNorm()
	if _, err := bn.Init(2, nil); err != nil {
		t.Fatal(err)
	}
	in := randomInput(8, 2, 5)
	bn.SetTraining(true)
	if _, err := bn.Forward(in); err != nil {
		t.Fatal(err)
	}
	bn.SetTraining(false)
	out1, err := bn.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	// Eval output must be deterministic and independent of batch composition.
	single := &tensai.Matrix{Rows: 1, Cols: 2, Data: in.Data[:2]}
	out2, err := bn.Forward(single)
	if err != nil {
		t.Fatal(err)
	}
	for c := 0; c < 2; c++ {
		if math.Abs(float64(out1.At(0, c)-out2.At(0, c))) > 1e-12 {
			t.Errorf("eval output depends on batch: %g vs %g", out1.At(0, c), out2.At(0, c))
		}
	}
}

func TestConv2DGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	conv := NewConv2D(5, 5, 2, 3, 3, 1, 1)
	outCols, err := conv.Init(5*5*2, rng)
	if err != nil {
		t.Fatal(err)
	}
	if outCols != 5*5*3 {
		t.Fatalf("expected out cols %d, got %d", 5*5*3, outCols)
	}
	checkLayerGrad(t, conv, randomInput(2, 5*5*2, 8), 1e-2)

	// Weight gradient against finite differences.
	in := randomInput(2, 5*5*2, 9)
	out, err := conv.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	coeff := tensai.NewMatrix(out.Rows, out.Cols)
	for i := range coeff.Data {
		coeff.Data[i] = tensai.Float(math.Cos(float64(i) * 0.3))
	}
	if _, err := conv.Backward(coeff); err != nil {
		t.Fatal(err)
	}
	gradW, _ := conv.Grads()
	const h = 5e-3
	for _, i := range []int{0, 7, len(conv.weights.Data) - 1} {
		orig := conv.weights.Data[i]
		conv.weights.Data[i] = orig + h
		outP, _ := conv.Forward(in)
		conv.weights.Data[i] = orig - h
		outM, _ := conv.Forward(in)
		conv.weights.Data[i] = orig
		var lp, lm float64
		for j := range outP.Data {
			lp += float64(coeff.Data[j]) * float64(outP.Data[j])
			lm += float64(coeff.Data[j]) * float64(outM.Data[j])
		}
		num := (lp - lm) / (2 * h)
		if math.Abs(num-float64(gradW.Data[i])) > 1e-2*(1+math.Abs(num)) {
			t.Errorf("weight grad %d: numeric=%.8f analytic=%.8f", i, num, gradW.Data[i])
		}
	}
}

func TestMaxPool2DGradient(t *testing.T) {
	pool := NewMaxPool2D(4, 4, 2, 2)
	outCols, err := pool.Init(4*4*2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if outCols != 2*2*2 {
		t.Fatalf("expected out cols %d, got %d", 2*2*2, outCols)
	}
	// Distinct values avoid ties, where max is not differentiable.
	in := tensai.NewMatrix(2, 4*4*2)
	perm := rand.New(rand.NewSource(3)).Perm(len(in.Data))
	for i, p := range perm {
		in.Data[i] = tensai.Float(p)
	}
	checkLayerGrad(t, pool, in, 1e-2)
}

func TestDropout(t *testing.T) {
	d := NewDropout(0.5)
	if _, err := d.Init(100, rand.New(rand.NewSource(11))); err != nil {
		t.Fatal(err)
	}
	in := tensai.NewMatrix(10, 100)
	for i := range in.Data {
		in.Data[i] = 1
	}

	// Eval mode: exact pass-through.
	out, err := d.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	if out != in {
		t.Error("eval-mode dropout should pass input through unchanged")
	}

	// Train mode: survivors scaled by 1/(1-rate), drop fraction near rate.
	d.SetTraining(true)
	out, err = d.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	zeros := 0
	for _, v := range out.Data {
		switch v {
		case 0:
			zeros++
		case 2:
		default:
			t.Fatalf("unexpected dropout output: %g", v)
		}
	}
	frac := tensai.Float(zeros) / tensai.Float(len(out.Data))
	if math.Abs(float64(frac-0.5)) > 0.05 {
		t.Errorf("drop fraction %g too far from 0.5", frac)
	}

	// Backward applies the same mask.
	grad := tensai.NewMatrix(10, 100)
	for i := range grad.Data {
		grad.Data[i] = 3
	}
	gin, err := d.Backward(grad)
	if err != nil {
		t.Fatal(err)
	}
	for i := range gin.Data {
		var want tensai.Float
		if out.Data[i] != 0 {
			want = 6
		}
		if gin.Data[i] != want {
			t.Fatalf("grad %d: got %g want %g", i, gin.Data[i], want)
		}
	}
}

func TestEmbeddingForwardBackward(t *testing.T) {
	emb := NewEmbedding(4, 2)
	if outCols, err := emb.Init(3, rand.New(rand.NewSource(17))); err != nil {
		t.Fatal(err)
	} else if outCols != 6 {
		t.Fatalf("expected out cols 6, got %d", outCols)
	}
	weights, err := tensai.NewMatrixFromSlice(4, 2, []tensai.Float{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := emb.SetParams(weights, nil); err != nil {
		t.Fatal(err)
	}
	in, err := tensai.NewMatrixFromSlice(2, 3, []tensai.Float{
		0, 1, 0,
		2, 1, 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := emb.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	wantOut := []tensai.Float{
		1, 2, 3, 4, 1, 2,
		5, 6, 3, 4, 7, 8,
	}
	for i, want := range wantOut {
		if out.Data[i] != want {
			t.Fatalf("forward %d: got %g want %g", i, out.Data[i], want)
		}
	}
	gradOut, err := tensai.NewMatrixFromSlice(2, 6, []tensai.Float{
		1, 10, 2, 20, 3, 30,
		4, 40, 5, 50, 6, 60,
	})
	if err != nil {
		t.Fatal(err)
	}
	gradIn, err := emb.Backward(gradOut)
	if err != nil {
		t.Fatal(err)
	}
	for i, got := range gradIn.Data {
		if got != 0 {
			t.Fatalf("input grad %d: got %g want 0", i, got)
		}
	}
	gradW, gradB := emb.Grads()
	if gradB != nil {
		t.Fatal("embedding should not expose bias gradients")
	}
	wantGradW := []tensai.Float{
		4, 40,
		7, 70,
		4, 40,
		6, 60,
	}
	for i, want := range wantGradW {
		if gradW.Data[i] != want {
			t.Fatalf("gradW %d: got %g want %g", i, gradW.Data[i], want)
		}
	}
}

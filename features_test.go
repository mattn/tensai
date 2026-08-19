package tensai

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// checkLayerGrad verifies a layer's Backward against central finite
// differences of its Forward, using L = sum(coeff * out) as the scalar loss.
// The float32 forward limits achievable precision, so the difference step
// and tolerances are coarser than they would be in float64, and the
// reductions accumulate in float64 to keep the comparison itself clean.
func checkLayerGrad(t *testing.T, layer Layer, input *Matrix, tol float64) {
	t.Helper()
	out, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	coeff := NewMatrix(out.Rows, out.Cols)
	for i := range coeff.Data {
		coeff.Data[i] = Float(math.Sin(float64(i)*0.7) + 0.3)
	}
	gradIn, err := layer.Backward(coeff)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	weighted := func(m *Matrix) float64 {
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

func randomInput(rows, cols int, seed int64) *Matrix {
	rng := rand.New(rand.NewSource(seed))
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = Float(rng.NormFloat64())
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
		var sum Float
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
	bn.setTraining(true)
	// Non-trivial gamma/beta so the scale path is exercised too.
	for i := range bn.gamma.Data {
		bn.gamma.Data[i] = 1.5
		bn.beta[i] = -0.25
	}
	checkLayerGrad(t, bn, randomInput(6, 4, 4), 1e-2)
}

func TestBatchNormEvalUsesRunningStats(t *testing.T) {
	bn := NewBatchNorm()
	if _, err := bn.Init(2, nil); err != nil {
		t.Fatal(err)
	}
	in := randomInput(8, 2, 5)
	bn.setTraining(true)
	if _, err := bn.Forward(in); err != nil {
		t.Fatal(err)
	}
	bn.setTraining(false)
	out1, err := bn.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	// Eval output must be deterministic and independent of batch composition.
	single := &Matrix{Rows: 1, Cols: 2, Data: in.Data[:2]}
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
	coeff := NewMatrix(out.Rows, out.Cols)
	for i := range coeff.Data {
		coeff.Data[i] = Float(math.Cos(float64(i) * 0.3))
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
	in := NewMatrix(2, 4*4*2)
	perm := rand.New(rand.NewSource(3)).Perm(len(in.Data))
	for i, p := range perm {
		in.Data[i] = Float(p)
	}
	checkLayerGrad(t, pool, in, 1e-2)
}

func TestDropout(t *testing.T) {
	d := NewDropout(0.5)
	if _, err := d.Init(100, rand.New(rand.NewSource(11))); err != nil {
		t.Fatal(err)
	}
	in := NewMatrix(10, 100)
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
	d.setTraining(true)
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
	frac := Float(zeros) / Float(len(out.Data))
	if math.Abs(float64(frac-0.5)) > 0.05 {
		t.Errorf("drop fraction %g too far from 0.5", frac)
	}

	// Backward applies the same mask.
	grad := NewMatrix(10, 100)
	for i := range grad.Data {
		grad.Data[i] = 3
	}
	gin, err := d.Backward(grad)
	if err != nil {
		t.Fatal(err)
	}
	for i := range gin.Data {
		var want Float
		if out.Data[i] != 0 {
			want = 6
		}
		if gin.Data[i] != want {
			t.Fatalf("grad %d: got %g want %g", i, gin.Data[i], want)
		}
	}
}

func TestBinaryCrossEntropyGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	pred := NewMatrix(4, 3)
	target := NewMatrix(4, 3)
	for i := range pred.Data {
		pred.Data[i] = Float(0.1 + 0.8*rng.Float64())
		target.Data[i] = Float(rng.Intn(2))
	}
	loss := BinaryCrossEntropy{}
	_, grad, err := loss.Loss(pred, target)
	if err != nil {
		t.Fatal(err)
	}
	const h = 1e-3
	for i := range pred.Data {
		orig := pred.Data[i]
		pred.Data[i] = orig + h
		lp, _, _ := loss.Loss(pred, target)
		pred.Data[i] = orig - h
		lm, _, _ := loss.Loss(pred, target)
		pred.Data[i] = orig
		num := float64(lp-lm) / (2 * h)
		if math.Abs(num-float64(grad.Data[i])) > 5e-3*(1+math.Abs(num)) {
			t.Errorf("grad %d: numeric=%.8f analytic=%.8f", i, num, grad.Data[i])
		}
	}
}

func TestAdamWDecaysWeights(t *testing.T) {
	adam := NewAdamW(0.1, 0.5)
	adam.NewLayer()
	weights := NewMatrix(1, 2)
	weights.Data[0] = 1
	weights.Data[1] = -2
	bias := []Float{3}
	// Zero gradients: plain Adam would leave parameters unchanged; AdamW
	// must still shrink weights (but never biases).
	adam.Step(0, weights, NewMatrix(1, 2), bias, []Float{0})
	if weights.Data[0] >= 1 || weights.Data[1] <= -2 {
		t.Errorf("weights not decayed: %v", weights.Data)
	}
	if bias[0] != 3 {
		t.Errorf("bias must not be decayed: %g", bias[0])
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	build := func() *Sequential {
		m := NewSequential()
		m.Add(NewDense(8))
		m.Add(NewBatchNorm())
		m.Add(NewLeakyReLU(0.1))
		m.Add(NewDropout(0.25))
		m.Add(NewDense(2))
		if err := m.Compile(3, SoftmaxCrossEntropy{}, NewAdamW(0.01, 0.01)); err != nil {
			t.Fatal(err)
		}
		return m
	}

	m1 := build()
	in := randomInput(16, 3, 21)
	tgt := NewMatrix(16, 1)
	for i := range tgt.Data {
		tgt.Data[i] = Float(i % 2)
	}
	for i := 0; i < 20; i++ {
		if _, err := m1.FitStep(in, tgt); err != nil {
			t.Fatal(err)
		}
	}

	var buf bytes.Buffer
	if err := m1.Save(&buf); err != nil {
		t.Fatal(err)
	}
	m2 := build()
	if err := m2.Load(&buf); err != nil {
		t.Fatal(err)
	}

	p1, err := m1.Predict(in)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := m2.Predict(in)
	if err != nil {
		t.Fatal(err)
	}
	for i := range p1.Data {
		if math.Abs(float64(p1.Data[i]-p2.Data[i])) > 1e-12 {
			t.Fatalf("prediction %d differs after load: %g vs %g", i, p1.Data[i], p2.Data[i])
		}
	}

	// Mismatched architecture must be rejected.
	if err := m1.Save(&buf); err != nil {
		t.Fatal(err)
	}
	m3 := NewSequential()
	m3.Add(NewDense(8))
	m3.Add(&ReLU{})
	m3.Add(NewDense(2))
	if err := m3.Compile(3, SoftmaxCrossEntropy{}, NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	if err := m3.Load(&buf); err == nil {
		t.Error("loading into a different architecture should fail")
	}
}

func TestConvNetLearnsLineOrientation(t *testing.T) {
	// 6x6 single-channel images: class 0 has a horizontal line, class 1 a
	// vertical line. A tiny conv net should separate them easily.
	const size = 6
	var inputData []Float
	var targetData []Float
	for pos := 1; pos < size-1; pos++ {
		h := make([]Float, size*size)
		v := make([]Float, size*size)
		for i := 0; i < size; i++ {
			h[pos*size+i] = 1
			v[i*size+pos] = 1
		}
		inputData = append(inputData, h...)
		targetData = append(targetData, 0)
		inputData = append(inputData, v...)
		targetData = append(targetData, 1)
	}
	rows := len(targetData)
	inputs, err := NewMatrixFromSlice(rows, size*size, inputData)
	if err != nil {
		t.Fatal(err)
	}
	targets, err := NewMatrixFromSlice(rows, 1, targetData)
	if err != nil {
		t.Fatal(err)
	}

	model := NewSequential()
	model.Add(NewConv2D(size, size, 1, 4, 3, 1, 1))
	model.Add(&ReLU{})
	model.Add(NewMaxPool2D(size, size, 4, 2))
	model.Add(NewDense(2))
	if err := model.Compile(size*size, SoftmaxCrossEntropy{}, NewAdam(0.05)); err != nil {
		t.Fatal(err)
	}
	var lossVal Float
	for i := 0; i < 200; i++ {
		if lossVal, err = model.FitStep(inputs, targets); err != nil {
			t.Fatal(err)
		}
	}
	if lossVal > 0.1 {
		t.Fatalf("conv net failed to converge: loss=%g", lossVal)
	}
	pred, err := model.Predict(inputs)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		best := 0
		if pred.At(r, 1) > pred.At(r, 0) {
			best = 1
		}
		if best != int(targets.At(r, 0)) {
			t.Errorf("sample %d misclassified", r)
		}
	}
}

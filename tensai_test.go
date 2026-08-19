package tensai

import (
	"math"
	"testing"
)

func TestDotShape(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(3, 2)
	for i := range a.Data {
		a.Data[i] = Float(i + 1)
	}
	for i := range b.Data {
		b.Data[i] = Float(i + 1)
	}
	out, err := Dot(a, b)
	if err != nil {
		t.Fatalf("Dot error: %v", err)
	}
	if out.Rows != 2 || out.Cols != 2 {
		t.Fatalf("expected 2x2, got %dx%d", out.Rows, out.Cols)
	}
	// a = [[1,2,3],[4,5,6]], b = [[1,2],[3,4],[5,6]]
	// a*b = [[22,28],[49,64]]
	checks := []struct {
		r, c int
		v    Float
	}{
		{0, 0, 22}, {0, 1, 28}, {1, 0, 49}, {1, 1, 64},
	}
	for _, ch := range checks {
		if got := out.At(ch.r, ch.c); got != ch.v {
			t.Errorf("At(%d,%d) = %g, want %g", ch.r, ch.c, got, ch.v)
		}
	}
}

func TestDenseForwardBackward(t *testing.T) {
	d := NewDense(2)
	// Manual weights: 2x2 identity-like
	d.weights = NewMatrix(2, 2)
	d.weights.Set(0, 0, 1)
	d.weights.Set(1, 1, 1)
	d.bias = []Float{0, 0}
	d.gradW = NewMatrix(2, 2)
	d.gradB = make([]Float, 2)

	input := NewMatrix(1, 2)
	input.Set(0, 0, 3)
	input.Set(0, 1, 5)

	out, err := d.Forward(input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out.At(0, 0) != 3 || out.At(0, 1) != 5 {
		t.Fatalf("forward = [%g %g], want [3 5]", out.At(0, 0), out.At(0, 1))
	}

	grad := NewMatrix(1, 2)
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

func TestMSEGradient(t *testing.T) {
	pred := NewMatrix(1, 3)
	target := NewMatrix(1, 3)
	pred.Set(0, 0, 1)
	pred.Set(0, 1, 2)
	pred.Set(0, 2, 3)
	target.Set(0, 0, 0)
	target.Set(0, 1, 0)
	target.Set(0, 2, 0)

	loss, grad, err := MeanSquaredError{}.Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// MSE = (1+4+9)/3 = 14/3 ≈ 4.667
	if math.Abs(float64(loss)-14.0/3.0) > 1e-6 {
		t.Errorf("loss = %g, want %g", loss, 14.0/3.0)
	}
	// grad = 2/n * (pred-target) = [2/3, 4/3, 2]
	expected := []Float{2.0 / 3, 4.0 / 3, 2.0}
	for i, e := range expected {
		if math.Abs(float64(grad.Data[i]-e)) > 1e-6 {
			t.Errorf("grad[%d] = %g, want %g", i, grad.Data[i], e)
		}
	}
}

func TestXORLearns(t *testing.T) {
	// A small network should be able to reduce XOR loss substantially.
	model := NewSequential()
	model.Add(NewDense(8))
	model.Add(&Tanh{})
	model.Add(NewDense(1))
	model.Add(&Sigmoid{})

	if err := model.Compile(2, MeanSquaredError{}, NewAdam(0.05)); err != nil {
		t.Fatalf("Compile: %v", err)
	}

	inputs := NewMatrix(4, 2)
	inputs.Set(0, 0, 0)
	inputs.Set(0, 1, 0)
	inputs.Set(1, 0, 0)
	inputs.Set(1, 1, 1)
	inputs.Set(2, 0, 1)
	inputs.Set(2, 1, 0)
	inputs.Set(3, 0, 1)
	inputs.Set(3, 1, 1)

	targets := NewMatrix(4, 1)
	targets.Set(0, 0, 0)
	targets.Set(1, 0, 1)
	targets.Set(2, 0, 1)
	targets.Set(3, 0, 0)

	for i := 0; i < 3000; i++ {
		if _, err := model.FitStep(inputs, targets); err != nil {
			t.Fatalf("FitStep: %v", err)
		}
	}

	pred, err := model.Predict(inputs)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for r := 0; r < 4; r++ {
		got := pred.At(r, 0)
		want := targets.At(r, 0)
		if math.Abs(float64(got-want)) > 0.2 {
			t.Errorf("row %d: pred=%.3f, target=%.0f (not converged)", r, got, want)
		}
	}
}

func TestSoftmaxCEGradient(t *testing.T) {
	// With a perfect prediction, loss should be near zero and gradient near zero.
	pred := NewMatrix(1, 3)
	pred.Set(0, 0, 10)
	pred.Set(0, 1, -10)
	pred.Set(0, 2, -10)
	target := NewMatrix(1, 1)
	target.Set(0, 0, 0) // class 0

	loss, grad, err := SoftmaxCrossEntropy{}.Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if loss > 0.001 {
		t.Errorf("expected near-zero loss, got %g", loss)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(float64(grad.Data[i])) > 0.01 {
			t.Errorf("grad[%d] = %g, expected near zero", i, grad.Data[i])
		}
	}
}

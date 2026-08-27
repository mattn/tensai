package loss

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

func TestMSEGradient(t *testing.T) {
	pred := tensai.NewMatrix(1, 3)
	target := tensai.NewMatrix(1, 3)
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
	expected := []tensai.Float{2.0 / 3, 4.0 / 3, 2.0}
	for i, e := range expected {
		if math.Abs(float64(grad.Data[i]-e)) > 1e-6 {
			t.Errorf("grad[%d] = %g, want %g", i, grad.Data[i], e)
		}
	}
}

func TestSoftmaxCEGradient(t *testing.T) {
	// With a perfect prediction, loss should be near zero and gradient near zero.
	pred := tensai.NewMatrix(1, 3)
	pred.Set(0, 0, 10)
	pred.Set(0, 1, -10)
	pred.Set(0, 2, -10)
	target := tensai.NewMatrix(1, 1)
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

func TestBinaryCrossEntropyGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	pred := tensai.NewMatrix(4, 3)
	target := tensai.NewMatrix(4, 3)
	for i := range pred.Data {
		pred.Data[i] = tensai.Float(0.1 + 0.8*rng.Float64())
		target.Data[i] = tensai.Float(rng.Intn(2))
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

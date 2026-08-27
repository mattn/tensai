package optim

import (
	"testing"

	"github.com/mattn/tensai"
)

func TestAdamWDecaysWeights(t *testing.T) {
	adam := NewAdamW(0.1, 0.5)
	adam.NewLayer()
	weights := tensai.NewMatrix(1, 2)
	weights.Data[0] = 1
	weights.Data[1] = -2
	bias := []tensai.Float{3}
	// Zero gradients: plain Adam would leave parameters unchanged; AdamW
	// must still shrink weights (but never biases).
	adam.Step(0, weights, tensai.NewMatrix(1, 2), bias, []tensai.Float{0})
	if weights.Data[0] >= 1 || weights.Data[1] <= -2 {
		t.Errorf("weights not decayed: %v", weights.Data)
	}
	if bias[0] != 3 {
		t.Errorf("bias must not be decayed: %g", bias[0])
	}
}

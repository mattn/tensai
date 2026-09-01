//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/optim"
)

// TestGPUAcceleratesAutograd runs one forward and backward pass of a small
// network twice -- once on the CPU, once with the device installed as the
// accelerator -- and checks that every parameter gradient agrees. The
// backward pass uses both transposed products, so this covers the whole
// routing path end to end.
func TestGPUAcceleratesAutograd(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	grads := func(seed int64) []*tensai.Tensor {
		rng := rand.New(rand.NewSource(seed))
		x := randTensor(rng, 32, 48)
		y := randTensor(rng, 32, 16)
		w1 := autograd.Param(randTensor(rng, 48, 64))
		w2 := autograd.Param(randTensor(rng, 64, 16))
		loss := autograd.Input(x).MatMul(w1).Tanh().MatMul(w2).MSELoss(y)
		loss.Backward()
		return []*tensai.Tensor{w1.Grad, w2.Grad}
	}

	want := grads(19)
	// A threshold of zero sends even these small products to the device.
	prev := tensai.UseAcceleratorThreshold(g, 0)
	got := grads(19)
	tensai.UseAccelerator(prev)

	for i := range want {
		if len(got[i].Data) != len(want[i].Data) {
			t.Fatalf("gradient %d: got %v want %v", i, got[i].Shape, want[i].Shape)
		}
		for j := range want[i].Data {
			diff := math.Abs(float64(got[i].Data[j] - want[i].Data[j]))
			if diff > 1e-4*(1+math.Abs(float64(want[i].Data[j]))) {
				t.Fatalf("gradient %d element %d: gpu=%v cpu=%v", i, j, got[i].Data[j], want[i].Data[j])
			}
		}
	}
}

// TestGPUAcceleratedTrainingConverges checks that a network trained with
// every product on the device still learns.
func TestGPUAcceleratedTrainingConverges(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(23))
	x := randTensor(rng, 64, 32)
	target := randTensor(rng, 64, 8)
	w1 := autograd.Param(randTensor(rng, 32, 64))
	w2 := autograd.Param(randTensor(rng, 64, 8))
	trainer := autograd.NewTrainer(optim.NewAdam(0.01), w1, w2)
	tape := autograd.NewTape()
	tape.Bind(w1, w2)

	prev := tensai.UseAcceleratorThreshold(g, 0)
	defer tensai.UseAccelerator(prev)

	var first, last tensai.Float
	for step := 0; step < 60; step++ {
		loss := autograd.Input(x).MatMul(w1).Tanh().MatMul(w2).MSELoss(target)
		last = trainer.Step(loss)
		tape.Reset()
		if step == 0 {
			first = last
		}
	}
	if !(last < first*0.5) {
		t.Fatalf("loss did not fall on the device: %g -> %g", first, last)
	}
}

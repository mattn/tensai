//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu_test

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
		return []*tensai.Tensor{w1.Grad(), w2.Grad()}
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

// TestGPUResidentTraining trains the same MLP twice from the same
// initialization -- once on the CPU, once with the graph resident on the
// device -- and checks the two stay together. With the tape on a device
// the values, the gradients and the Adam update all stay there; only the
// loss comes home each step.
func TestGPUResidentTraining(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	build := func() (x, y *tensai.Tensor, params []*autograd.Node) {
		rng := rand.New(rand.NewSource(101))
		x = randTensor(rng, 32, 24)
		y = randTensor(rng, 32, 8)
		w1 := autograd.Param(randTensor(rng, 24, 32))
		b1 := autograd.Param(tensai.NewTensor(1, 32))
		w2 := autograd.Param(randTensor(rng, 32, 8))
		return x, y, []*autograd.Node{w1, b1, w2}
	}
	forward := func(x *tensai.Tensor, p []*autograd.Node) *autograd.Node {
		return autograd.Input(x).MatMul(p[0]).Add(p[1]).Tanh().MatMul(p[2])
	}

	const steps = 120
	x, y, cpuP := build()
	cpuTrainer := autograd.NewTrainer(optim.NewAdam(0.01), cpuP...)
	var firstLoss, cpuLoss tensai.Float
	for i := 0; i < steps; i++ {
		cpuLoss = cpuTrainer.Step(forward(x, cpuP).MSELoss(y))
		if i == 0 {
			firstLoss = cpuLoss
		}
	}

	_, _, devP := build()
	devTrainer := autograd.NewTrainer(optim.NewAdam(0.01), devP...)
	tape := autograd.NewTape()
	tape.UseDevice(g)
	tape.Bind(devP...)
	var devLoss tensai.Float
	for i := 0; i < steps; i++ {
		devLoss = devTrainer.Step(forward(x, devP).MSELoss(y))
		tape.Reset()
	}

	if diff := math.Abs(float64(cpuLoss - devLoss)); diff > 1e-2*(1+math.Abs(float64(cpuLoss))) {
		t.Fatalf("loss differs after %d steps: cpu=%g device=%g", steps, cpuLoss, devLoss)
	}
	for i := range cpuP {
		want, got := cpuP[i].Value(), devP[i].Value()
		for j := range want.Data {
			if diff := math.Abs(float64(want.Data[j] - got.Data[j])); diff > 1e-2*(1+math.Abs(float64(want.Data[j]))) {
				t.Fatalf("param %d element %d: cpu=%v device=%v", i, j, want.Data[j], got.Data[j])
			}
		}
	}
	if !(cpuLoss < firstLoss*0.5) {
		t.Fatalf("training did not progress: %g -> %g", firstLoss, cpuLoss)
	}
}

// TestGPUResidentStaysOnDevice checks that a resident graph really keeps
// its intermediates on the GPU: nothing but the loss should have a host
// copy after a step.
func TestGPUResidentStaysOnDevice(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(103))
	x := randTensor(rng, 16, 16)
	w := autograd.Param(randTensor(rng, 16, 16))
	tape := autograd.NewTape()
	tape.UseDevice(g)
	tape.Bind(w)

	// The input carries no tape of its own, so this also checks that a
	// per-step constant joins the graph's rather than dropping it to the
	// CPU for the rest of the step.
	prod := autograd.Input(x).MatMul(w)
	if !prod.Resident() {
		t.Fatal("the product did not stay on the device")
	}
	h := prod.Tanh()
	if !h.Resident() {
		t.Fatal("the activation did not stay on the device")
	}
	// Reading it brings it home, and the value must match the CPU.
	got := h.Value()
	want, err := tensai.MatMul(x, w.Value())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Data {
		w := math.Tanh(float64(want.Data[i]))
		if diff := math.Abs(float64(got.Data[i]) - w); diff > 1e-4 {
			t.Fatalf("element %d: device=%v cpu=%v", i, got.Data[i], w)
		}
	}
	tape.Reset()
}

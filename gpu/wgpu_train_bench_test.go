//go:build wgpu || wgpu24

package gpu_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/optim"
)

// BenchmarkGemmTrain times the three products a training step runs -- the
// forward a*b, the input gradient a*b^T, and the weight gradient a^T*b --
// on the GPU with both operands already resident, against the CPU kernels.
// Square shapes stand in for a transformer's projections, where m is
// batch*sequence and k and n are the model width.
func BenchmarkGemmTrain(b *testing.B) {
	g, err := gpu.Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(11))

	modes := []struct {
		name string
		gpu  func(a, b *gpu.Tensor) (*gpu.Tensor, error)
		cpu  func(a, b *tensai.Tensor) (*tensai.Tensor, error)
	}{
		{"nn", (*gpu.Tensor).MatMul, tensai.MatMul},
		{"nt", (*gpu.Tensor).MatMulT, tensai.MatMulNT},
		{"tn", (*gpu.Tensor).MatMulTN, tensai.MatMulTN},
	}
	for _, size := range []int{256, 512, 1024} {
		x, w := randTensor(rng, size, size), randTensor(rng, size, size)
		gx, err := g.Upload(x)
		if err != nil {
			b.Fatal(err)
		}
		gw, err := g.Upload(w)
		if err != nil {
			b.Fatal(err)
		}
		for _, m := range modes {
			b.Run(fmt.Sprintf("gpu/%s/%d", m.name, size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					out, err := m.gpu(gx, gw)
					if err != nil {
						b.Fatal(err)
					}
					out.Free()
				}
				// Force the queue to drain before the timer stops.
				if out, err := m.gpu(gx, gw); err == nil {
					out.Download()
					out.Free()
				}
			})
			b.Run(fmt.Sprintf("cpu/%s/%d", m.name, size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if _, err := m.cpu(x, w); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
		gx.Free()
		gw.Free()
	}
}

// BenchmarkGemmRoundTrip times the same product with the operands living on
// the CPU: upload, compute, download. It is what a transparent offload
// would pay, against the CPU kernel it would replace.
func BenchmarkGemmRoundTrip(b *testing.B) {
	g, err := gpu.Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(13))
	for _, size := range []int{512, 1024, 2048} {
		x, w := randTensor(rng, size, size), randTensor(rng, size, size)
		b.Run(fmt.Sprintf("gpu/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := g.MatMul(x, w); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("cpu/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := tensai.MatMul(x, w); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkTrainStepAccel times one autograd training step of a
// transformer-width block with and without the device installed as
// tensai's accelerator. Every product in the step -- the two forward
// projections and the four gradients behind them -- is large enough to
// cross the threshold, so this is what a real model gains.
func BenchmarkTrainStepAccel(b *testing.B) {
	g, err := gpu.Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()

	run := func(rows, model int) func(*testing.B) {
		return func(b *testing.B) {
			rng := rand.New(rand.NewSource(3))
			x := randTensor(rng, rows, model)
			y := randTensor(rng, rows, model)
			w1 := autograd.Param(randTensor(rng, model, model))
			w2 := autograd.Param(randTensor(rng, model, model))
			trainer := autograd.NewTrainer(optim.NewAdam(0.001), w1, w2)
			tape := autograd.NewTape()
			tape.Bind(w1, w2)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				trainer.Step(autograd.Input(x).MatMul(w1).GELU().MatMul(w2).MSELoss(y))
				tape.Reset()
			}
		}
	}
	for _, size := range []int{512, 1024, 2048} {
		b.Run(fmt.Sprintf("cpu/%d", size), run(size, size))
		b.Run(fmt.Sprintf("accel/%d", size), func(b *testing.B) {
			prev := tensai.UseAccelerator(g)
			defer tensai.UseAccelerator(prev)
			run(size, size)(b)
		})
		b.Run(fmt.Sprintf("resident/%d", size), runResident(g, size, size))
	}
}

// runResident is the same step with the whole graph on the device: the
// tape holds the buffers, so only the loss crosses the bus each step.
func runResident(g *gpu.Device, rows, model int) func(*testing.B) {
	return func(b *testing.B) {
		rng := rand.New(rand.NewSource(3))
		x := randTensor(rng, rows, model)
		y := randTensor(rng, rows, model)
		w1 := autograd.Param(randTensor(rng, model, model))
		w2 := autograd.Param(randTensor(rng, model, model))
		trainer := autograd.NewTrainer(optim.NewAdam(0.001), w1, w2)
		tape := autograd.NewTape()
		tape.UseDevice(g)
		tape.Bind(w1, w2)
		trainer.Step(autograd.Input(x).MatMul(w1).GELU().MatMul(w2).MSELoss(y))
		tape.Reset()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			trainer.Step(autograd.Input(x).MatMul(w1).GELU().MatMul(w2).MSELoss(y))
			tape.Reset()
		}
	}
}

// BenchmarkElementwise compares one activation pass over a big tensor on
// the device against the CPU kernel, both with the data already where they
// want it. It is the other half of a resident training step: if the device
// is not ahead here, keeping activations resident cannot pay.
func BenchmarkElementwise(b *testing.B) {
	g, err := gpu.Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(17))
	for _, n := range []int{1 << 20, 1 << 22} {
		x := randTensor(rng, n)
		gx, err := g.Upload(x)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("gpu/%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				out, err := gx.Activate(gpu.ActGELU)
				if err != nil {
					b.Fatal(err)
				}
				out.Free()
			}
			if out, err := gx.Activate(gpu.ActGELU); err == nil {
				out.Download()
				out.Free()
			}
		})
		b.Run(fmt.Sprintf("cpu/%d", n), func(b *testing.B) {
			dst := make([]tensai.Float, n)
			for i := 0; i < b.N; i++ {
				kernels.GeluFwd(dst, x.Data)
			}
		})
		gx.Free()
	}
}

// BenchmarkTransformerStep times one training step of a transformer block
// -- pre-norm attention with a causal mask and a GELU feed-forward -- on
// the CPU and with the graph resident on the device.
func BenchmarkTransformerStep(b *testing.B) {
	g, err := gpu.Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()

	const (
		batch, seq, vocab = 8, 128, 512
		heads             = 8
	)
	step := func(model int, resident bool) func(*testing.B) {
		return func(b *testing.B) {
			headDim := model / heads
			rng := rand.New(rand.NewSource(7))
			tokens := make([]int, batch*seq)
			labels := make([]int, batch*seq)
			for i := range tokens {
				tokens[i] = rng.Intn(vocab)
				labels[i] = rng.Intn(vocab)
			}
			mask := tensai.NewTensor(1, 1, seq, seq)
			for i := 0; i < seq; i++ {
				for j := i + 1; j < seq; j++ {
					mask.Data[i*seq+j] = tensai.Float(math.Inf(-1))
				}
			}
			ones := tensai.NewTensor(model)
			for i := range ones.Data {
				ones.Data[i] = 1
			}
			embed := autograd.Param(randTensor(rng, vocab, model))
			gain := autograd.Param(ones)
			bias := autograd.Param(tensai.NewTensor(model))
			wq := autograd.Param(randTensor(rng, model, model))
			wk := autograd.Param(randTensor(rng, model, model))
			wv := autograd.Param(randTensor(rng, model, model))
			wo := autograd.Param(randTensor(rng, model, model))
			head := autograd.Param(randTensor(rng, model, vocab))
			params := []*autograd.Node{embed, gain, bias, wq, wk, wv, wo, head}
			trainer := autograd.NewTrainer(optim.NewAdam(0.001), params...)
			tape := autograd.NewTape()
			if resident {
				tape.UseDevice(g)
			}
			tape.Bind(params...)

			forward := func() *autograd.Node {
				split := func(t *autograd.Node) *autograd.Node {
					return t.Reshape(batch, seq, heads, headDim).Transpose(0, 2, 1, 3)
				}
				x := embed.Embed(tokens, batch, seq)
				h := x.LayerNorm(gain, bias, 1e-5)
				q, k, v := split(h.MatMul(wq)), split(h.MatMul(wk)), split(h.MatMul(wv))
				att := q.MatMul(k.T()).Scale(1 / float32(math.Sqrt(float64(headDim)))).Add(autograd.Input(mask)).Softmax()
				y := att.MatMul(v).Transpose(0, 2, 1, 3).Reshape(batch, seq, model).MatMul(wo)
				return x.Add(y).GELU().MatMul(head)
			}
			trainer.Step(forward().CrossEntropy(labels))
			tape.Reset()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				trainer.Step(forward().CrossEntropy(labels))
				tape.Reset()
			}
		}
	}
	for _, model := range []int{256, 512} {
		b.Run(fmt.Sprintf("cpu/%d", model), step(model, false))
		b.Run(fmt.Sprintf("resident/%d", model), step(model, true))
	}
}

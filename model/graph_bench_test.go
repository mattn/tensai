package model

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/optim"
)

// BenchmarkStep times one training step of the same model both ways: the
// layers' own Forward and Backward, and the autograd graph. The layer path
// reuses a scratch buffer per layer, so what this measures is what the
// graph's generality costs on a CPU.
func BenchmarkStep(b *testing.B) {
	build := func(width int) (*Sequential, *tensai.Matrix, *tensai.Matrix) {
		rng := rand.New(rand.NewSource(3))
		x := randMatrix(rng, 64, width)
		y := randMatrix(rng, 64, 16)
		net := NewSequential()
		net.Add(layer.NewDense(width))
		net.Add(&layer.Tanh{})
		net.Add(layer.NewDense(16))
		if err := net.Compile(width, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
			b.Fatal(err)
		}
		return net, x, y
	}
	for _, width := range []int{64, 256, 1024} {
		b.Run(fmt.Sprintf("layers/%d", width), func(b *testing.B) {
			net, x, y := build(width)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := net.FitStep(x, y); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("graph/%d", width), func(b *testing.B) {
			net, x, y := build(width)
			g, err := net.Graph()
			if err != nil {
				b.Fatal(err)
			}
			trainer := autograd.NewTrainer(optim.NewAdam(0.01), g.Params()...)
			tape := autograd.NewTape()
			tape.Bind(g.Params()...)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				l, err := g.Loss(g.Forward(autograd.Input(x)), y)
				if err != nil {
					b.Fatal(err)
				}
				trainer.Step(l)
				tape.Reset()
			}
		})
	}
}

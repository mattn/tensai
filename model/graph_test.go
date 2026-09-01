package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/optim"
)

func randMatrix(rng *rand.Rand, rows, cols int) *tensai.Matrix {
	m := tensai.NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return m
}

// TestGraphMatchesPredict checks that the graph form of a model computes
// what the layer stack computes, for a dense net and for a convolutional
// one, since the convolution has to fold the spatial axes in and out.
func TestGraphMatchesPredict(t *testing.T) {
	cases := []struct {
		name  string
		build func() *Sequential
		input *tensai.Matrix
	}{
		{"dense", func() *Sequential {
			net := NewSequential()
			net.Add(layer.NewDense(8))
			net.Add(&layer.Tanh{})
			net.Add(layer.NewLayerNorm())
			net.Add(layer.NewDense(3))
			if err := net.Compile(5, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
				t.Fatal(err)
			}
			return net
		}, randMatrix(rand.New(rand.NewSource(1)), 4, 5)},
		{"conv", func() *Sequential {
			net := NewSequential()
			net.Add(layer.NewConv2D(4, 3, 1, 1))
			net.Add(&layer.ReLU{})
			net.Add(layer.NewMaxPool2D(2))
			net.Add(layer.NewDense(3))
			if err := net.CompileImage(layer.Image{H: 8, W: 8, C: 2}, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
				t.Fatal(err)
			}
			return net
		}, randMatrix(rand.New(rand.NewSource(2)), 3, 2*8*8)},
	}
	for _, tc := range cases {
		net := tc.build()
		want, err := net.Predict(tc.input)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		g, err := net.Graph()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		got := g.Forward(autograd.Input(tc.input)).Value()
		if len(got.Data) != len(want.Data) {
			t.Fatalf("%s: got %d elements, want %d", tc.name, len(got.Data), len(want.Data))
		}
		for i := range want.Data {
			if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
				t.Fatalf("%s element %d: graph=%v layers=%v", tc.name, i, got.Data[i], want.Data[i])
			}
		}
	}
}

// TestGraphTrainsTheModel trains through the graph and checks that the
// model itself learned: the parameters are the layers' own buffers, so
// Predict sees the result without any copying.
func TestGraphTrainsTheModel(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	inputs, err := tensai.NewMatrixFromSlice(4, 2, []tensai.Float{0, 0, 0, 1, 1, 0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	targets, err := tensai.NewMatrixFromSlice(4, 1, []tensai.Float{0, 1, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	_ = rng

	net := NewSequential()
	net.Add(layer.NewDense(8))
	net.Add(&layer.Tanh{})
	net.Add(layer.NewDense(1))
	net.Add(&layer.Sigmoid{})
	if err := net.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05)); err != nil {
		t.Fatal(err)
	}

	g, err := net.Graph()
	if err != nil {
		t.Fatal(err)
	}
	trainer := autograd.NewTrainer(optim.NewAdam(0.05), g.Params()...)
	tape := autograd.NewTape()
	tape.Bind(g.Params()...)
	var lossVal tensai.Float
	for step := 0; step < 1500; step++ {
		out := g.Forward(autograd.Input(inputs))
		l, err := g.Loss(out, targets)
		if err != nil {
			t.Fatal(err)
		}
		lossVal = trainer.Step(l)
		tape.Reset()
	}
	if lossVal > 0.02 {
		t.Fatalf("graph training did not converge: loss=%g", lossVal)
	}

	// The layers hold the trained weights, so the model predicts with them.
	g.Sync()
	pred, err := net.Predict(inputs)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 4; r++ {
		if diff := math.Abs(float64(pred.At(r, 0) - targets.At(r, 0))); diff > 0.15 {
			t.Errorf("sample %d: model predicts %.3f, want %g", r, pred.At(r, 0), targets.At(r, 0))
		}
	}
}

// TestGraphRefusesUnsupportedLayers checks that a layer with no graph form
// is reported rather than quietly skipped.
func TestGraphRefusesUnsupportedLayers(t *testing.T) {
	net := NewSequential()
	net.Add(layer.NewDense(4))
	net.Add(layer.NewDropout(0.5))
	if err := net.Compile(3, loss.MeanSquaredError{}, optim.NewSGD(0.1, 0.9)); err != nil {
		t.Fatal(err)
	}
	if _, err := net.Graph(); err == nil {
		t.Fatal("expected an error for a model holding a Dropout")
	}
}

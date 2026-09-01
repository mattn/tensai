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
	net.Add(layer.NewEmbedding(10, 4))
	net.Add(layer.NewDense(4))
	if err := net.Compile(3, loss.MeanSquaredError{}, optim.NewSGD(0.1, 0.9)); err != nil {
		t.Fatal(err)
	}
	if _, err := net.Graph(); err == nil {
		t.Fatal("expected an error for a model holding an Embedding")
	}
}

// TestGraphDropout checks both halves of the layer's behaviour: a
// pass-through at inference, and about the right fraction of survivors
// scaled to keep the expected value during training.
func TestGraphDropout(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	x := randMatrix(rng, 64, 32)

	net := NewSequential()
	net.Add(layer.NewDropout(0.25))
	if err := net.Compile(32, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	g, err := net.Graph()
	if err != nil {
		t.Fatal(err)
	}

	g.SetTraining(false)
	got := g.Forward(autograd.Input(x)).Value()
	for i := range x.Data {
		if got.Data[i] != x.Data[i] {
			t.Fatalf("inference dropped element %d", i)
		}
	}

	g.SetTraining(true)
	out := g.Forward(autograd.Input(x)).Value()
	var kept int
	for i := range x.Data {
		switch v := out.Data[i]; {
		case v == 0:
		case math.Abs(float64(v-x.Data[i]/0.75)) < 1e-4:
			kept++
		default:
			t.Fatalf("element %d is neither dropped nor scaled: %v from %v", i, v, x.Data[i])
		}
	}
	if frac := float64(kept) / float64(len(x.Data)); frac < 0.7 || frac > 0.8 {
		t.Errorf("kept %.2f of the elements, want about 0.75", frac)
	}
}

// TestGraphBatchNorm checks the graph form against the layer it stands in
// for: the same output, the same gradients, and the same running
// estimates after a step.
func TestGraphBatchNorm(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	x := randMatrix(rng, 8, 5)
	upstream := randMatrix(rng, 8, 5)

	// The layer on its own.
	bn := layer.NewBatchNorm()
	if _, err := bn.Init(5, rng); err != nil {
		t.Fatal(err)
	}
	bn.SetTraining(true)
	want, err := bn.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	wantOut := append([]tensai.Float(nil), want.Data...)
	wantDx, err := bn.Backward(upstream)
	if err != nil {
		t.Fatal(err)
	}
	wantDxData := append([]tensai.Float(nil), wantDx.Data...)
	wantGamma, wantBeta := bn.Grads()
	wantMean, wantVar := bn.RunningStats()
	wantMeanData := append([]tensai.Float(nil), wantMean...)
	wantVarData := append([]tensai.Float(nil), wantVar...)

	// The same layer inside a graph.
	net := NewSequential()
	net.Add(layer.NewBatchNorm())
	if err := net.Compile(5, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	g, err := net.Graph()
	if err != nil {
		t.Fatal(err)
	}
	in := autograd.Param(x)
	out := g.Forward(in)
	for i := range wantOut {
		if diff := math.Abs(float64(out.Value().Data[i] - wantOut[i])); diff > 1e-4 {
			t.Fatalf("output %d: graph=%v layer=%v", i, out.Value().Data[i], wantOut[i])
		}
	}
	// Backward from the same upstream gradient.
	out.Mul(autograd.Input(upstream)).Sum().Backward()
	for i := range wantDxData {
		if diff := math.Abs(float64(in.Grad().Data[i] - wantDxData[i])); diff > 1e-4 {
			t.Fatalf("input gradient %d: graph=%v layer=%v", i, in.Grad().Data[i], wantDxData[i])
		}
	}
	gammaNode, betaNode := g.Params()[0], g.Params()[1]
	for i := range wantGamma.Data {
		if diff := math.Abs(float64(gammaNode.Grad().Data[i] - wantGamma.Data[i])); diff > 1e-4 {
			t.Fatalf("gamma gradient %d: graph=%v layer=%v", i, gammaNode.Grad().Data[i], wantGamma.Data[i])
		}
		if diff := math.Abs(float64(betaNode.Grad().Data[i] - wantBeta[i])); diff > 1e-4 {
			t.Fatalf("beta gradient %d: graph=%v layer=%v", i, betaNode.Grad().Data[i], wantBeta[i])
		}
	}
	// And the running estimates moved the same way.
	gotBN := net.Layers()[0].(*layer.BatchNorm)
	gotMean, gotVar := gotBN.RunningStats()
	for i := range wantMeanData {
		if diff := math.Abs(float64(gotMean[i] - wantMeanData[i])); diff > 1e-5 {
			t.Fatalf("running mean %d: graph=%v layer=%v", i, gotMean[i], wantMeanData[i])
		}
		if diff := math.Abs(float64(gotVar[i] - wantVarData[i])); diff > 1e-5 {
			t.Fatalf("running variance %d: graph=%v layer=%v", i, gotVar[i], wantVarData[i])
		}
	}

	// Inference uses those estimates, and matches the layer again.
	bn.SetTraining(false)
	wantInfer, err := bn.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	g.SetTraining(false)
	gotInfer := g.Forward(autograd.Input(x)).Value()
	for i := range wantInfer.Data {
		if diff := math.Abs(float64(gotInfer.Data[i] - wantInfer.Data[i])); diff > 1e-4 {
			t.Fatalf("inference %d: graph=%v layer=%v", i, gotInfer.Data[i], wantInfer.Data[i])
		}
	}
}

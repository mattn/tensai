package model

import (
	"bytes"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/optim"
)

func randomInput(rows, cols int, seed int64) *tensai.Matrix {
	rng := rand.New(rand.NewSource(seed))
	m := tensai.NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return m
}

func TestXORLearns(t *testing.T) {
	// A small network should be able to reduce XOR loss substantially.
	model := NewSequential()
	model.Add(layer.NewDense(8))
	model.Add(&layer.Tanh{})
	model.Add(layer.NewDense(1))
	model.Add(&layer.Sigmoid{})

	if err := model.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05)); err != nil {
		t.Fatalf("Compile: %v", err)
	}

	inputs := tensai.NewMatrix(4, 2)
	inputs.Set(0, 0, 0)
	inputs.Set(0, 1, 0)
	inputs.Set(1, 0, 0)
	inputs.Set(1, 1, 1)
	inputs.Set(2, 0, 1)
	inputs.Set(2, 1, 0)
	inputs.Set(3, 0, 1)
	inputs.Set(3, 1, 1)

	targets := tensai.NewMatrix(4, 1)
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

func TestSaveLoadRoundtrip(t *testing.T) {
	build := func() *Sequential {
		m := NewSequential()
		m.Add(layer.NewDense(8))
		m.Add(layer.NewBatchNorm())
		m.Add(layer.NewLeakyReLU(0.1))
		m.Add(layer.NewDropout(0.25))
		m.Add(layer.NewDense(2))
		if err := m.Compile(3, loss.SoftmaxCrossEntropy{}, optim.NewAdamW(0.01, 0.01)); err != nil {
			t.Fatal(err)
		}
		return m
	}

	m1 := build()
	in := randomInput(16, 3, 21)
	tgt := tensai.NewMatrix(16, 1)
	for i := range tgt.Data {
		tgt.Data[i] = tensai.Float(i % 2)
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
	m3.Add(layer.NewDense(8))
	m3.Add(&layer.ReLU{})
	m3.Add(layer.NewDense(2))
	if err := m3.Compile(3, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	if err := m3.Load(&buf); err == nil {
		t.Error("loading into a different architecture should fail")
	}
}

func TestEmbeddingLayerNormGELUSaveLoadRoundtrip(t *testing.T) {
	build := func() *Sequential {
		m := NewSequential()
		m.Add(layer.NewEmbedding(4, 3))
		m.Add(layer.NewLayerNorm())
		m.Add(&layer.GELU{})
		m.Add(layer.NewDense(2))
		if err := m.Compile(2, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.03)); err != nil {
			t.Fatal(err)
		}
		return m
	}

	var inputData []tensai.Float
	var targetData []tensai.Float
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			inputData = append(inputData, tensai.Float(a), tensai.Float(b))
			if a == b {
				targetData = append(targetData, 1)
			} else {
				targetData = append(targetData, 0)
			}
		}
	}
	in, err := tensai.NewMatrixFromSlice(16, 2, inputData)
	if err != nil {
		t.Fatal(err)
	}
	tgt, err := tensai.NewMatrixFromSlice(16, 1, targetData)
	if err != nil {
		t.Fatal(err)
	}

	m1 := build()
	for i := 0; i < 30; i++ {
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
}

func TestConvNetLearnsLineOrientation(t *testing.T) {
	// 6x6 single-channel images: class 0 has a horizontal line, class 1 a
	// vertical line. A tiny conv net should separate them easily.
	const size = 6
	var inputData []tensai.Float
	var targetData []tensai.Float
	for pos := 1; pos < size-1; pos++ {
		h := make([]tensai.Float, size*size)
		v := make([]tensai.Float, size*size)
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
	inputs, err := tensai.NewMatrixFromSlice(rows, size*size, inputData)
	if err != nil {
		t.Fatal(err)
	}
	targets, err := tensai.NewMatrixFromSlice(rows, 1, targetData)
	if err != nil {
		t.Fatal(err)
	}

	model := NewSequential()
	model.Add(layer.NewConv2D(4, 3, 1, 1))
	model.Add(&layer.ReLU{})
	model.Add(layer.NewMaxPool2D(2))
	model.Add(layer.NewDense(2))
	if err := model.CompileImage(layer.Image{H: size, W: size, C: 1}, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.05)); err != nil {
		t.Fatal(err)
	}
	var lossVal tensai.Float
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

func BenchmarkFitStepMLP(b *testing.B) {
	model := NewSequential()
	model.Add(layer.NewDense(256))
	model.Add(&layer.ReLU{})
	model.Add(layer.NewDense(128))
	model.Add(&layer.ReLU{})
	model.Add(layer.NewDense(4))
	if err := model.Compile(10, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.005)); err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	in := tensai.NewMatrix(64, 10)
	tgt := tensai.NewMatrix(64, 1)
	for i := range in.Data {
		in.Data[i] = tensai.Float(rng.Intn(2))
	}
	for i := range tgt.Data {
		tgt.Data[i] = tensai.Float(rng.Intn(4))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.FitStep(in, tgt); err != nil {
			b.Fatal(err)
		}
	}
}

func TestLoadLegacyTypeNames(t *testing.T) {
	m := NewSequential()
	m.Add(layer.NewDense(4))
	m.Add(&layer.ReLU{})
	m.Add(layer.NewDense(2))
	if err := m.Compile(3, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
		t.Fatalf("Compile: %v", err)
	}
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	// Checkpoints written before the package split stored the layer type
	// package-qualified ("*tensai.Dense"); loading must still accept them.
	legacy := strings.ReplaceAll(buf.String(), `"type":"Dense"`, `"type":"*tensai.Dense"`)
	legacy = strings.ReplaceAll(legacy, `"type":"ReLU"`, `"type":"*tensai.ReLU"`)
	if legacy == buf.String() {
		t.Fatalf("expected replaceable type names in %q", buf.String())
	}
	if err := m.Load(strings.NewReader(legacy)); err != nil {
		t.Fatalf("Load legacy: %v", err)
	}
}

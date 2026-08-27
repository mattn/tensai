package onnx

import (
	"bytes"
	"math/rand"
	"testing"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

func trainStep(t *testing.T, m *model.Sequential, in, out int, classes bool, rng *rand.Rand) {
	t.Helper()
	x := tensai.NewMatrix(8, in)
	for i := range x.Data {
		x.Data[i] = float32(rng.NormFloat64())
	}
	var y *tensai.Matrix
	if classes {
		y = tensai.NewMatrix(8, 1)
		for i := range y.Data {
			y.Data[i] = float32(rng.Intn(out))
		}
	} else {
		y = tensai.NewMatrix(8, out)
		for i := range y.Data {
			y.Data[i] = float32(rng.NormFloat64())
		}
	}
	if _, err := m.FitStep(x, y); err != nil {
		t.Fatal(err)
	}
}

func TestMarshalMLP(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	m := model.NewSequential()
	m.Add(layer.NewDense(16))
	m.Add(&layer.Tanh{})
	m.Add(layer.NewBatchNorm())
	m.Add(layer.NewDropout(0.5))
	m.Add(layer.NewDense(8))
	m.Add(&layer.LeakyReLU{Alpha: 0.1})
	m.Add(layer.NewDense(3))
	m.Add(&layer.Softmax{})
	if err := m.Compile(10, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	trainStep(t, m, 10, 3, false, rng)

	data, err := Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if len(data) == 0 || !bytes.Contains(data, []byte("tensai")) {
		t.Fatal("marshaled model looks wrong")
	}
	for _, op := range []string{"Gemm", "Tanh", "Mul", "Add", "LeakyRelu", "Softmax"} {
		if !bytes.Contains(data, []byte(op)) {
			t.Fatalf("missing op %s", op)
		}
	}
	if bytes.Contains(data, []byte("Dropout")) {
		t.Fatal("dropout must be dropped")
	}
	// Marshal must be deterministic.
	again, err := Marshal(m)
	if err != nil || !bytes.Equal(data, again) {
		t.Fatal("marshal is not deterministic")
	}
}

func TestMarshalCNN(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	m := model.NewSequential()
	m.Add(layer.NewConv2D(8, 8, 2, 6, 3, 1, 1))
	m.Add(&layer.ReLU{})
	m.Add(layer.NewMaxPool2D(8, 8, 6, 2))
	m.Add(layer.NewConv2D(4, 4, 6, 4, 3, 1, 0))
	m.Add(&layer.Sigmoid{})
	m.Add(layer.NewDense(5))
	if err := m.Compile(8*8*2, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	trainStep(t, m, 8*8*2, 5, true, rng)

	data, err := Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, op := range []string{"Conv", "MaxPool", "Flatten", "Gemm"} {
		if !bytes.Contains(data, []byte(op)) {
			t.Fatalf("missing op %s", op)
		}
	}
}

func TestMarshalErrors(t *testing.T) {
	if _, err := Marshal(model.NewSequential()); err == nil {
		t.Fatal("expected error for empty model")
	}

	m := model.NewSequential()
	m.Add(&layer.ReLU{})
	if err := m.Compile(4, loss.MeanSquaredError{}, optim.NewSGD(0.1, 0)); err != nil {
		t.Fatal(err)
	}
	if _, err := Marshal(m); err == nil {
		t.Fatal("expected error for activation-first model")
	}

	rng := rand.New(rand.NewSource(3))
	sm := model.NewSequential()
	sm.Add(layer.NewConv2D(4, 4, 1, 2, 3, 1, 1))
	sm.Add(&layer.Softmax{})
	if err := sm.Compile(16, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	trainStep(t, sm, 16, 32, false, rng)
	if _, err := Marshal(sm); err == nil {
		t.Fatal("expected error for softmax on spatial features")
	}
}

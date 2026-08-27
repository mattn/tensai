package tflite

import (
	"testing"

	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

func buildMLP(t *testing.T) *model.Sequential {
	t.Helper()
	m := model.NewSequential()
	m.Add(layer.NewDense(8))
	m.Add(&layer.ReLU{})
	m.Add(layer.NewDense(3))
	m.Add(&layer.Softmax{})
	if err := m.Compile(4, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	return m
}

func buildCNN(t *testing.T) *model.Sequential {
	t.Helper()
	m := model.NewSequential()
	m.Add(layer.NewConv2D(4, 3, 1, 1))
	m.Add(&layer.ReLU{})
	m.Add(layer.NewMaxPool2D(2))
	m.Add(layer.NewDense(8))
	m.Add(layer.NewBatchNorm())
	m.Add(layer.NewLeakyReLU(0.1))
	m.Add(layer.NewDropout(0.2))
	m.Add(layer.NewDense(2))
	if err := m.CompileImage(layer.Image{H: 8, W: 8, C: 1}, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestMarshalStructure(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *model.Sequential
	}{
		{"mlp", buildMLP(t)},
		{"cnn", buildCNN(t)},
	} {
		data, err := Marshal(tc.model)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if len(data) < 8 {
			t.Fatalf("%s: too short: %d bytes", tc.name, len(data))
		}
		if got := string(data[4:8]); got != "TFL3" {
			t.Fatalf("%s: file identifier = %q", tc.name, got)
		}
	}
}

func TestMarshalRejectsUnsupported(t *testing.T) {
	m := model.NewSequential()
	m.Add(layer.NewDense(4))
	m.Add(&layer.GELU{})
	if err := m.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	if _, err := Marshal(m); err == nil {
		t.Error("GELU should be reported as unsupported")
	}

	m = model.NewSequential()
	m.Add(layer.NewConv2D(4, 5, 1, 1)) // k=5, pad=1: neither VALID nor SAME
	if err := m.CompileImage(layer.Image{H: 8, W: 8, C: 1}, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	if _, err := Marshal(m); err == nil {
		t.Error("non-VALID/SAME padding should be rejected")
	}
}

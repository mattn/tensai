package tflite

import (
	"testing"

	tensai "github.com/mattn/tensai"
)

func buildMLP(t *testing.T) *tensai.Sequential {
	t.Helper()
	m := tensai.NewSequential()
	m.Add(tensai.NewDense(8))
	m.Add(&tensai.ReLU{})
	m.Add(tensai.NewDense(3))
	m.Add(&tensai.Softmax{})
	if err := m.Compile(4, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	return m
}

func buildCNN(t *testing.T) *tensai.Sequential {
	t.Helper()
	m := tensai.NewSequential()
	m.Add(tensai.NewConv2D(8, 8, 1, 4, 3, 1, 1))
	m.Add(&tensai.ReLU{})
	m.Add(tensai.NewMaxPool2D(8, 8, 4, 2))
	m.Add(tensai.NewDense(8))
	m.Add(tensai.NewBatchNorm())
	m.Add(tensai.NewLeakyReLU(0.1))
	m.Add(tensai.NewDropout(0.2))
	m.Add(tensai.NewDense(2))
	if err := m.Compile(8*8, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestMarshalStructure(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *tensai.Sequential
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
	m := tensai.NewSequential()
	m.Add(tensai.NewDense(4))
	m.Add(&tensai.GELU{})
	if err := m.Compile(2, tensai.MeanSquaredError{}, tensai.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	if _, err := Marshal(m); err == nil {
		t.Error("GELU should be reported as unsupported")
	}

	m = tensai.NewSequential()
	m.Add(tensai.NewConv2D(8, 8, 1, 4, 5, 1, 1)) // k=5, pad=1: neither VALID nor SAME
	if err := m.Compile(8*8, tensai.MeanSquaredError{}, tensai.NewAdam(0.01)); err != nil {
		t.Fatal(err)
	}
	if _, err := Marshal(m); err == nil {
		t.Error("non-VALID/SAME padding should be rejected")
	}
}

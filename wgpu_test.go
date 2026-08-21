//go:build wgpu && (linux || darwin || windows)

package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func openTestGPU(t *testing.T) *GPU {
	t.Helper()
	g, err := OpenGPU()
	if err != nil {
		t.Skipf("wgpu unavailable: %v", err)
	}
	return g
}

func TestGPUMatMul(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(7))
	cases := [][2][]int{
		{{3, 4}, {4, 5}},
		{{2, 3, 4}, {2, 4, 5}},
		{{3, 4}, {6, 4, 5}},
		{{6, 3, 4}, {4, 5}},
		{{2, 1, 3, 4}, {1, 5, 4, 2}},
		{{1, 17, 9}, {4, 9, 33}}, // not multiples of the 8x8 workgroup
	}
	for _, c := range cases {
		a, b := randTensor(rng, c[0]...), randTensor(rng, c[1]...)
		got, err := g.MatMul(a, b)
		if err != nil {
			t.Fatalf("gpu matmul %v*%v: %v", c[0], c[1], err)
		}
		want, err := MatMul(a, b)
		if err != nil {
			t.Fatalf("cpu matmul: %v", err)
		}
		if !sameDims(got.Shape, want.Shape) {
			t.Fatalf("shape: got %v want %v", got.Shape, want.Shape)
		}
		for i := range want.Data {
			if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
				t.Fatalf("%v*%v element %d: gpu=%v cpu=%v", c[0], c[1], i, got.Data[i], want.Data[i])
			}
		}
	}

	if _, err := g.MatMul(randTensor(rng, 2, 3), randTensor(rng, 4, 5)); err == nil {
		t.Fatal("expected shape mismatch error")
	}
	if _, err := g.MatMul(randTensor(rng, 4), randTensor(rng, 4, 5)); err == nil {
		t.Fatal("expected error for 1-D operand")
	}
}

func TestGPUAdapterSelection(t *testing.T) {
	// Power preference is a hint, so with any adapter set this must still
	// succeed and report a name.
	g, err := OpenGPU(GPULowPower)
	if err != nil {
		t.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	if g.Name() == "" {
		t.Fatal("adapter name is empty")
	}
	t.Logf("adapter: %s", g.Name())

	x, err := NewTensorFromSlice([]Float{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.MatMul(x, x)
	if err != nil {
		t.Fatalf("gpu matmul: %v", err)
	}
	want := []Float{7, 10, 15, 22}
	for i, v := range want {
		if got.Data[i] != v {
			t.Fatalf("element %d: got %v want %v", i, got.Data[i], v)
		}
	}
}

func TestGPUMatMulLarge(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(8))
	a, b := randTensor(rng, 8, 64, 96), randTensor(rng, 8, 96, 80)
	got, err := g.MatMul(a, b)
	if err != nil {
		t.Fatalf("gpu matmul: %v", err)
	}
	want, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("cpu matmul: %v", err)
	}
	for i := range want.Data {
		// f32 accumulation order differs between CPU and GPU.
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-2 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}
}

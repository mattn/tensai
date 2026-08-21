//go:build (wgpu || wgpu24) && (linux || darwin || windows)

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

func TestGPUTensorResident(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(11))

	// A weight uploaded once serves several MatMuls without re-upload.
	w := randTensor(rng, 64, 32)
	gw, err := g.Upload(w)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	defer gw.Free()
	if !sameDims(gw.Shape(), []int{64, 32}) || gw.Size() != 64*32 {
		t.Fatalf("shape/size: %v %d", gw.Shape(), gw.Size())
	}
	for i := 0; i < 3; i++ {
		x := randTensor(rng, 4, 5, 64)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatalf("upload x: %v", err)
		}
		gy, err := gx.MatMul(gw)
		if err != nil {
			t.Fatalf("resident matmul: %v", err)
		}
		got, err := gy.Download()
		if err != nil {
			t.Fatalf("download: %v", err)
		}
		want, err := MatMul(x, w)
		if err != nil {
			t.Fatalf("cpu matmul: %v", err)
		}
		if !sameDims(got.Shape, want.Shape) {
			t.Fatalf("shape: got %v want %v", got.Shape, want.Shape)
		}
		for j := range want.Data {
			if diff := math.Abs(float64(got.Data[j] - want.Data[j])); diff > 1e-4 {
				t.Fatalf("round %d element %d: gpu=%v cpu=%v", i, j, got.Data[j], want.Data[j])
			}
		}
		gy.Free()
		gx.Free()
	}
}

func TestGPUTensorChain(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(12))

	// (a @ b) @ c entirely on the GPU: the intermediate never touches the
	// host.
	a, b, c := randTensor(rng, 2, 4, 8), randTensor(rng, 2, 8, 3), randTensor(rng, 2, 3, 6)
	ga, err := g.Upload(a)
	if err != nil {
		t.Fatal(err)
	}
	defer ga.Free()
	gb, err := g.Upload(b)
	if err != nil {
		t.Fatal(err)
	}
	defer gb.Free()
	gc, err := g.Upload(c)
	if err != nil {
		t.Fatal(err)
	}
	defer gc.Free()

	ab, err := ga.MatMul(gb)
	if err != nil {
		t.Fatalf("a@b: %v", err)
	}
	defer ab.Free()
	abc, err := ab.MatMul(gc)
	if err != nil {
		t.Fatalf("(a@b)@c: %v", err)
	}
	got, err := abc.Download()
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	wantAB, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want, err := MatMul(wantAB, c)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}

	if _, err := ab.MatMul(ab); err == nil {
		t.Fatal("expected shape mismatch error for (2,4,3)@(2,4,3)")
	}
	abc.Free()
	if _, err := abc.Download(); err == nil {
		t.Fatal("expected error downloading a freed tensor")
	}
	if _, err := abc.MatMul(ga); err == nil {
		t.Fatal("expected error using a freed tensor")
	}
	abc.Free() // second Free is a no-op
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

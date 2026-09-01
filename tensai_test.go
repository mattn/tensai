package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func TestDotShape(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(3, 2)
	for i := range a.Data {
		a.Data[i] = Float(i + 1)
	}
	for i := range b.Data {
		b.Data[i] = Float(i + 1)
	}
	out, err := Dot(a, b)
	if err != nil {
		t.Fatalf("Dot error: %v", err)
	}
	if out.Rows != 2 || out.Cols != 2 {
		t.Fatalf("expected 2x2, got %dx%d", out.Rows, out.Cols)
	}
	// a = [[1,2,3],[4,5,6]], b = [[1,2],[3,4],[5,6]]
	// a*b = [[22,28],[49,64]]
	checks := []struct {
		r, c int
		v    Float
	}{
		{0, 0, 22}, {0, 1, 28}, {1, 0, 49}, {1, 1, 64},
	}
	for _, ch := range checks {
		if got := out.At(ch.r, ch.c); got != ch.v {
			t.Errorf("At(%d,%d) = %g, want %g", ch.r, ch.c, got, ch.v)
		}
	}
}

func TestDotTAInto(t *testing.T) {
	rng := rand.New(rand.NewSource(89))
	// The last three are tall and narrow, which is the shape a
	// convolution's weight gradient has and the register-accumulating
	// kernel takes: a k that is not a multiple of four, a b that is not a
	// multiple of eight columns, and one wide enough to need two passes.
	for _, dims := range [][3]int{{5, 3, 4}, {8, 8, 8}, {17, 9, 13}, {64, 33, 5},
		{1024, 9, 8}, {600, 6, 5}, {1000, 7, 20}} {
		r, i, j := dims[0], dims[1], dims[2]
		a := RandomMatrix(r, i, rng)
		b := RandomMatrix(r, j, rng)
		// Sprinkle zeros to exercise the skip path.
		for k := 0; k < len(a.Data); k += 3 {
			a.Data[k] = 0
		}
		want, err := Dot(a.T(), b)
		if err != nil {
			t.Fatal(err)
		}
		got := NewMatrix(i, j)
		if err := DotTAInto(got, a, b); err != nil {
			t.Fatal(err)
		}
		for k := range want.Data {
			if diff := got.Data[k] - want.Data[k]; diff > 1e-5 || diff < -1e-5 {
				t.Fatalf("dims %v: element %d differs: %g vs %g", dims, k, got.Data[k], want.Data[k])
			}
		}
	}
	if err := DotTAInto(NewMatrix(2, 2), NewMatrix(3, 2), NewMatrix(4, 2)); err == nil {
		t.Error("row mismatch should be rejected")
	}
	if err := DotTAInto(NewMatrix(2, 2), NewMatrix(3, 2), NewMatrix(3, 5)); err == nil {
		t.Error("output shape mismatch should be rejected")
	}
}

func TestDotVecAxpy(t *testing.T) {
	rng := rand.New(rand.NewSource(81))
	for _, n := range []int{0, 1, 7, 8, 15, 16, 64, 127, 1000} {
		a := make([]Float, n)
		b := make([]Float, n)
		for i := range a {
			a[i] = Float(rng.NormFloat64())
			b[i] = Float(rng.NormFloat64())
		}
		var want float64
		for i := range a {
			want += float64(a[i]) * float64(b[i])
		}
		got := float64(DotVec(a, b))
		if diff := math.Abs(got - want); diff > 1e-3*(1+math.Abs(want)) {
			t.Fatalf("DotVec n=%d: got %v want %v", n, got, want)
		}

		y := make([]Float, n)
		copy(y, b)
		Axpy(0.5, a, y)
		for i := range y {
			want := b[i] + 0.5*a[i]
			if diff := math.Abs(float64(y[i] - want)); diff > 1e-5 {
				t.Fatalf("Axpy n=%d elem %d: got %v want %v", n, i, y[i], want)
			}
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	DotVec(make([]Float, 3), make([]Float, 4))
}

func TestNewMatrixFromInts(t *testing.T) {
	m, err := NewMatrixFromInts(2, 2, []int{0, 1, 65535, 3})
	if err != nil {
		t.Fatal(err)
	}
	if m.At(1, 0) != 65535 {
		t.Fatalf("got %g", m.At(1, 0))
	}
	// 1<<25 + 1 is the first odd integer float32 cannot represent.
	if _, err := NewMatrixFromInts(1, 1, []int{1<<25 + 1}); err == nil {
		t.Fatal("expected exactness error")
	}
	if _, err := NewMatrixFromInts(2, 2, []int{1}); err == nil {
		t.Fatal("expected length error")
	}
}

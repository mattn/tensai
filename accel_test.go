package tensai

import (
	"errors"
	"math/rand"
	"testing"
)

// fakeAccel records what it was asked for and answers with a plain
// reference product, so the routing can be tested without a GPU. It must
// not call back into MatMul, which would consult the hook again.
type fakeAccel struct {
	nn, tn, nt int
	fail       bool
}

func (f *fakeAccel) MatMul(a, b *Tensor) (*Tensor, error) {
	f.nn++
	return f.run(a, b, gemmNN)
}

func (f *fakeAccel) MatMulTN(a, b *Tensor) (*Tensor, error) {
	f.tn++
	return f.run(a, b, gemmTN)
}

func (f *fakeAccel) MatMulNT(a, b *Tensor) (*Tensor, error) {
	f.nt++
	return f.run(a, b, gemmNT)
}

func (f *fakeAccel) run(a, b *Tensor, mode gemmMode) (*Tensor, error) {
	if f.fail {
		return nil, errors.New("backend refused the product")
	}
	return refProduct(a, b, mode), nil
}

// refProduct multiplies two plain matrices with the definition, taking one
// operand transposed for the modes that ask for it.
func refProduct(a, b *Tensor, mode gemmMode) *Tensor {
	at := func(t *Tensor, r, c int) Float { return t.Data[r*t.Shape[1]+c] }
	var m, k, n int
	switch mode {
	case gemmTN:
		k, m, n = a.Shape[0], a.Shape[1], b.Shape[1]
	case gemmNT:
		m, k, n = a.Shape[0], a.Shape[1], b.Shape[0]
	default:
		m, k, n = a.Shape[0], a.Shape[1], b.Shape[1]
	}
	out := NewTensor(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum Float
			for x := 0; x < k; x++ {
				av, bv := at(a, i, x), at(b, x, j)
				if mode == gemmTN {
					av = at(a, x, i)
				}
				if mode == gemmNT {
					bv = at(b, j, x)
				}
				sum += av * bv
			}
			out.Data[i*n+j] = sum
		}
	}
	return out
}

// closeEnough reports whether two tensors agree to float32 rounding.
func closeEnough(t *testing.T, got, want *Tensor) {
	t.Helper()
	for i := range want.Data {
		d := float64(got.Data[i] - want.Data[i])
		if d > 1e-4 || d < -1e-4 {
			t.Fatalf("element %d: got %g want %g", i, got.Data[i], want.Data[i])
		}
	}
}

func TestAcceleratorRouting(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	a := randTestTensor(rng, 8, 6)
	b := randTestTensor(rng, 8, 5)

	f := &fakeAccel{}
	UseAcceleratorThreshold(f, 0)
	got, err := MatMulTN(a, b)
	UseAccelerator(nil)
	if err != nil {
		t.Fatal(err)
	}
	if f.tn != 1 || f.nn != 0 || f.nt != 0 {
		t.Fatalf("routing: nn=%d tn=%d nt=%d, want tn only", f.nn, f.tn, f.nt)
	}
	closeEnough(t, got, refProduct(a, b, gemmTN))
}

func TestAcceleratorThreshold(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	small, w := randTestTensor(rng, 4, 4), randTestTensor(rng, 4, 4)
	f := &fakeAccel{}
	// 4*4*4 = 64 MACs, well under the threshold.
	UseAcceleratorThreshold(f, 1000)
	if _, err := MatMul(small, w); err != nil {
		t.Fatal(err)
	}
	UseAccelerator(nil)
	if f.nn != 0 {
		t.Errorf("small product went to the accelerator %d times", f.nn)
	}

	big, bw := randTestTensor(rng, 16, 16), randTestTensor(rng, 16, 16)
	f = &fakeAccel{}
	UseAcceleratorThreshold(f, 1000) // 16^3 = 4096 MACs
	if _, err := MatMul(big, bw); err != nil {
		t.Fatal(err)
	}
	UseAccelerator(nil)
	if f.nn != 1 {
		t.Errorf("big product went to the accelerator %d times, want 1", f.nn)
	}
}

// TestAcceleratorFallback checks that a failing backend costs correctness
// nothing: the product simply runs on the CPU.
func TestAcceleratorFallback(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	a, b := randTestTensor(rng, 6, 4), randTestTensor(rng, 4, 5)
	want := refProduct(a, b, gemmNN)

	f := &fakeAccel{fail: true}
	UseAcceleratorThreshold(f, 0)
	got, err := MatMul(a, b)
	// Into form as well: it is the one autograd uses.
	into := NewTensor(6, 5)
	err2 := MatMulInto(into, a, b)
	UseAccelerator(nil)
	if err != nil || err2 != nil {
		t.Fatalf("fallback returned errors: %v, %v", err, err2)
	}
	closeEnough(t, got, want)
	closeEnough(t, into, want)
}

// TestAcceleratorInto checks the Into forms route as well, since those are
// what the autograd engine calls.
func TestAcceleratorInto(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	a, b := randTestTensor(rng, 8, 6), randTestTensor(rng, 8, 5)
	f := &fakeAccel{}
	out := NewTensor(6, 5)
	UseAcceleratorThreshold(f, 0)
	err := MatMulTNInto(out, a, b)
	UseAccelerator(nil)
	if err != nil {
		t.Fatal(err)
	}
	if f.tn != 1 {
		t.Fatalf("Into form did not use the accelerator: %+v", f)
	}
	closeEnough(t, out, refProduct(a, b, gemmTN))
}

func randTestTensor(rng *rand.Rand, shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = Float(rng.NormFloat64())
	}
	return t
}

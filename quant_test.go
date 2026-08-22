package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantizeMatVec(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for _, c := range []struct{ rows, cols int }{
		{768, 2304}, // big enough for the parallel path
		{16, 16},
		{7, 13}, // scalar tails on every chunk
		{1, 9},
	} {
		m := RandomMatrix(c.rows, c.cols, rng)
		q := QuantizeMatrix(m)
		x := make([]Float, c.rows)
		for i := range x {
			x[i] = Float(rng.NormFloat64())
		}
		out := make([]Float, c.cols)
		if err := q.MatVec(x, out); err != nil {
			t.Fatalf("%dx%d: %v", c.rows, c.cols, err)
		}

		// Exact reference over the same quantized weights: the kernel must
		// match it to float rounding.
		want := make([]float64, c.cols)
		for i := 0; i < c.rows; i++ {
			for j := 0; j < c.cols; j++ {
				want[j] += float64(x[i]) * float64(q.Q[i*c.cols+j])
			}
		}
		var worst float64
		for j := range want {
			want[j] *= float64(q.Scale[j])
			diff := math.Abs(float64(out[j]) - want[j])
			if diff > 1e-3*(1+math.Abs(want[j])) {
				t.Fatalf("%dx%d col %d: got %v want %v", c.rows, c.cols, j, out[j], want[j])
			}
			// And it must stay close to the full-precision product.
			var full float64
			for i := 0; i < c.rows; i++ {
				full += float64(x[i]) * float64(m.Data[i*c.cols+j])
			}
			if d := math.Abs(want[j] - full); d > worst {
				worst = d
			}
		}
		if worst > 0.5 { // int8 error over a few hundred accumulations stays small
			t.Fatalf("%dx%d: quantization error %v too large", c.rows, c.cols, worst)
		}
	}

	q := QuantizeMatrix(NewMatrix(4, 4)) // all zeros: scales are zero
	out := make([]Float, 4)
	if err := q.MatVec(make([]Float, 4), out); err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		if v != 0 {
			t.Fatalf("zero matrix produced %v", out)
		}
	}
	if err := q.MatVec(make([]Float, 3), out); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

// The Big pair exceeds the L3 cache, which is the regime inference decode
// lives in: there the f32 matvec is memory-bandwidth bound and the int8
// weights pull four times less.
func BenchmarkMatVecF32Big(b *testing.B) {
	rng := rand.New(rand.NewSource(33))
	w := RandomMatrix(4096, 16384, rng)
	x := NewMatrix(1, 4096)
	for i := range x.Data {
		x.Data[i] = Float(rng.NormFloat64())
	}
	out := NewMatrix(1, 16384)
	b.SetBytes(4096 * 16384 * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := DotInto(out, x, w); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecQ8Big(b *testing.B) {
	rng := rand.New(rand.NewSource(33))
	q := QuantizeMatrix(RandomMatrix(4096, 16384, rng))
	x := make([]Float, 4096)
	for i := range x {
		x[i] = Float(rng.NormFloat64())
	}
	out := make([]Float, 16384)
	b.SetBytes(4096 * 16384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecF32(b *testing.B) {
	rng := rand.New(rand.NewSource(32))
	w := RandomMatrix(768, 2304, rng)
	x := NewMatrix(1, 768)
	for i := range x.Data {
		x.Data[i] = Float(rng.NormFloat64())
	}
	out := NewMatrix(1, 2304)
	b.SetBytes(768 * 2304 * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := DotInto(out, x, w); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecQ8(b *testing.B) {
	rng := rand.New(rand.NewSource(32))
	q := QuantizeMatrix(RandomMatrix(768, 2304, rng))
	x := make([]Float, 768)
	for i := range x {
		x[i] = Float(rng.NormFloat64())
	}
	out := make([]Float, 2304)
	b.SetBytes(768 * 2304)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

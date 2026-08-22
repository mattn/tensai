package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantize4MatVec(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	for _, c := range []struct{ rows, cols int }{
		{768, 2304}, // parallel path
		{64, 32},
		{33, 10}, // partial final group, scalar tails
		{1, 2},
	} {
		m := RandomMatrix(c.rows, c.cols, rng)
		q, err := QuantizeMatrix4(m)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]Float, c.rows)
		for i := range x {
			x[i] = Float(rng.NormFloat64())
		}
		out := make([]Float, c.cols)
		if err := q.MatVec(x, out); err != nil {
			t.Fatalf("%dx%d: %v", c.rows, c.cols, err)
		}

		// Exact reference over the same quantized nibbles.
		half := c.cols / 2
		groups := (c.rows + q4Group - 1) / q4Group
		want := make([]float64, c.cols)
		for g := 0; g < groups; g++ {
			rlo, rhi := g*q4Group, min((g+1)*q4Group, c.rows)
			for j := 0; j < c.cols; j++ {
				var acc float64
				for i := rlo; i < rhi; i++ {
					b := q.Q[i*half+j%half]
					n := int(b&0x0F) - 8
					if j >= half {
						n = int(b>>4) - 8
					}
					acc += float64(x[i]) * float64(n)
				}
				want[j] += acc * float64(q.Scale[g*c.cols+j])
			}
		}
		var worst float64
		for j := range want {
			if diff := math.Abs(float64(out[j]) - want[j]); diff > 1e-3*(1+math.Abs(want[j])) {
				t.Fatalf("%dx%d col %d: got %v want %v", c.rows, c.cols, j, out[j], want[j])
			}
			var full float64
			for i := 0; i < c.rows; i++ {
				full += float64(x[i]) * float64(m.Data[i*c.cols+j])
			}
			if d := math.Abs(want[j] - full); d > worst {
				worst = d
			}
		}
		if worst > 1.0 { // group-wise int4 stays close over these sizes
			t.Fatalf("%dx%d: quantization error %v too large", c.rows, c.cols, worst)
		}
	}

	if _, err := QuantizeMatrix4(NewMatrix(4, 3)); err == nil {
		t.Fatal("expected error for odd column count")
	}
	q, err := QuantizeMatrix4(NewMatrix(4, 4))
	if err != nil {
		t.Fatal(err)
	}
	if err := q.MatVec(make([]Float, 3), make([]Float, 4)); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func BenchmarkMatVecQ4Big(b *testing.B) {
	rng := rand.New(rand.NewSource(33))
	q, err := QuantizeMatrix4(RandomMatrix(4096, 16384, rng))
	if err != nil {
		b.Fatal(err)
	}
	x := make([]Float, 4096)
	for i := range x {
		x[i] = Float(rng.NormFloat64())
	}
	out := make([]Float, 16384)
	b.SetBytes(4096 * 16384 / 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

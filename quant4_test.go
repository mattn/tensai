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
		{33, 10}, // odd rows: pad pair, partial final group
		{5, 7},   // odd columns, scalar tails everywhere
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

		// Exact reference over the same nibbles and 7-bit activations.
		xu, sx := quantizeActs(x)
		groups := (c.rows + q4Group - 1) / q4Group
		want := make([]float64, c.cols)
		for j := 0; j < c.cols; j++ {
			for g := 0; g < groups; g++ {
				rlo, rhi := g*q4Group, min((g+1)*q4Group, c.rows)
				var acc, gs int64
				for i := rlo; i < rhi; i++ {
					nib := int64(q.Q[(i/4)*2*c.cols+2*j+(i%4)/2]>>(4*(i%2))) & 0x0F
					xs := int64(xu[i]) - 64
					acc += nib * xs
					gs += xs
				}
				want[j] += float64(acc-8*gs) * float64(q.Scale[g*c.cols+j])
			}
			want[j] *= float64(sx)
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
		if worst > 2 { // int4 weights + 7-bit activations stay close
			t.Fatalf("%dx%d: quantization error %v too large", c.rows, c.cols, worst)
		}
	}

	q, err := QuantizeMatrix4(NewMatrix(4, 4)) // all zeros: scales are zero
	if err != nil {
		t.Fatal(err)
	}
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

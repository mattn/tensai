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

		// Exact reference over the same quantized weights and the same
		// 7-bit activations: the kernel must match it to float rounding.
		xu, sx := quantizeActs(x)
		want := make([]float64, c.cols)
		for i := 0; i < c.rows; i++ {
			for j := 0; j < c.cols; j++ {
				w := q.Q[q.Index(i, j)]
				want[j] += float64(int(xu[i])-64) * float64(w)
			}
		}
		var worst float64
		for j := range want {
			want[j] *= float64(q.Scale[j]) * float64(sx)
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
		if worst > 2 { // int8 weights + 7-bit activations stay close
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

func TestQMatMulBatch(t *testing.T) {
	rng := rand.New(rand.NewSource(51))
	for _, c := range []struct{ batch, rows, cols int }{
		{11, 768, 2304}, // 8-row blocks plus a remainder
		{8, 64, 33},     // scalar column tails
		{2, 5, 8},       // pure remainder path
	} {
		w := RandomMatrix(c.rows, c.cols, rng)
		q := QuantizeMatrix(w)
		x := RandomMatrix(c.batch, c.rows, rng)
		out := NewMatrix(c.batch, c.cols)
		if err := q.MatMul(x, out); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		// The batch must equal per-row MatVec bit for bit: identical
		// activation quantization, identical integer accumulation.
		row := make([]Float, c.cols)
		for r := 0; r < c.batch; r++ {
			if err := q.MatVec(x.Data[r*c.rows:(r+1)*c.rows], row); err != nil {
				t.Fatal(err)
			}
			for j := range row {
				if out.Data[r*c.cols+j] != row[j] {
					t.Fatalf("%v row %d col %d: batch %v matvec %v", c, r, j, out.Data[r*c.cols+j], row[j])
				}
			}
		}
	}

	q := QuantizeMatrix(NewMatrix(4, 4))
	if err := q.MatMul(NewMatrix(2, 3), NewMatrix(2, 4)); err == nil {
		t.Fatal("expected shape mismatch error")
	}

	for _, c := range []struct{ batch, rows, cols int }{
		{9, 768, 2304}, // 4-row blocks plus a remainder, parallel path
		{4, 130, 33},   // partial final group, scalar column tails
		{6, 33, 10},    // odd rows: pad pair inside the first group
		{2, 5, 7},      // pure remainder path
	} {
		w := RandomMatrix(c.rows, c.cols, rng)
		q4, err := QuantizeMatrix4(w)
		if err != nil {
			t.Fatal(err)
		}
		x := RandomMatrix(c.batch, c.rows, rng)
		out := NewMatrix(c.batch, c.cols)
		if err := q4.MatMul(x, out); err != nil {
			t.Fatalf("q4 %v: %v", c, err)
		}
		row := make([]Float, c.cols)
		for r := 0; r < c.batch; r++ {
			if err := q4.MatVec(x.Data[r*c.rows:(r+1)*c.rows], row); err != nil {
				t.Fatal(err)
			}
			for j := range row {
				if out.Data[r*c.cols+j] != row[j] {
					t.Fatalf("q4 %v row %d col %d: batch %v matvec %v", c, r, j, out.Data[r*c.cols+j], row[j])
				}
			}
		}
	}
}

// BenchmarkQ8Prefill measures the batched matmul against running the same
// rows one matvec at a time — the prompt-prefill comparison.
func BenchmarkQ8PrefillBatched(b *testing.B) {
	rng := rand.New(rand.NewSource(52))
	q := QuantizeMatrix(RandomMatrix(1536, 8960, rng))
	x := RandomMatrix(64, 1536, rng)
	out := NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatMul(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQ4PrefillBatched(b *testing.B) {
	rng := rand.New(rand.NewSource(52))
	q, err := QuantizeMatrix4(RandomMatrix(1536, 8960, rng))
	if err != nil {
		b.Fatal(err)
	}
	x := RandomMatrix(64, 1536, rng)
	out := NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatMul(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQ4PrefillRowwise(b *testing.B) {
	rng := rand.New(rand.NewSource(52))
	q, err := QuantizeMatrix4(RandomMatrix(1536, 8960, rng))
	if err != nil {
		b.Fatal(err)
	}
	x := RandomMatrix(64, 1536, rng)
	out := NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := 0; r < 64; r++ {
			if err := q.MatVec(x.Data[r*1536:(r+1)*1536], out.Data[r*8960:(r+1)*8960]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkQ8PrefillRowwise(b *testing.B) {
	rng := rand.New(rand.NewSource(52))
	q := QuantizeMatrix(RandomMatrix(1536, 8960, rng))
	x := RandomMatrix(64, 1536, rng)
	out := NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := 0; r < 64; r++ {
			if err := q.MatVec(x.Data[r*1536:(r+1)*1536], out.Data[r*8960:(r+1)*8960]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func TestQuantizeActsAgree(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for _, n := range []int{5, 16, 31, 1536, 4099} {
		x := make([]Float, n)
		for i := range x {
			x[i] = Float(rng.NormFloat64())
		}
		// Force representable ties and extremes into the mix.
		x[0] = 0
		if n > 8 {
			x[7] = -x[1]
			x[8] = 63.5 * x[2]
		}
		want := make([]uint8, (n+3)&^3)
		for i := n; i < len(want); i++ {
			want[i] = 64
		}
		wantSx := quantizeActsScalar(x, want)
		got, gotSx := quantizeActs(x)
		if gotSx != wantSx {
			t.Fatalf("n=%d: sx %v vs scalar %v", n, gotSx, wantSx)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d idx %d: %d vs scalar %d (x=%v)", n, i, got[i], want[i], x[min(i, n-1)])
			}
		}
	}
}

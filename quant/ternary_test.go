package quant

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

func TestTernaryIndex(t *testing.T) {
	// Every (row, column) lands on its own slot, and reads back.
	q := NewTernaryMatrix(48, 70)
	seen := map[[2]int]bool{}
	for i := 0; i < q.Rows; i++ {
		for j := 0; j < q.Cols; j++ {
			idx, shift := q.Index(i, j)
			if idx >= len(q.Q)-32 || shift > 6 {
				t.Fatalf("(%d,%d) at %d shift %d", i, j, idx, shift)
			}
			k := [2]int{idx, int(shift)}
			if seen[k] {
				t.Fatalf("(%d,%d) collides at %d shift %d", i, j, idx, shift)
			}
			seen[k] = true
			w := int8(i+j)%3 - 1
			q.Set(i, j, w)
			if got := q.At(i, j); got != w {
				t.Fatalf("(%d,%d) set %d read %d", i, j, w, got)
			}
		}
	}
	// SetGroup lands every weight where Set would.
	q = NewTernaryMatrix(256, 40)
	var w [128]int8
	for j := 0; j < q.Cols; j++ {
		for g := 0; g < 2; g++ {
			for i := range w {
				w[i] = int8((i*j+g)%3) - 1
			}
			q.SetGroup(g, j, &w)
			for i := range w {
				if got := q.At(g*128+i, j); got != w[i] {
					t.Fatalf("group %d col %d row %d: %d, want %d", g, j, i, got, w[i])
				}
			}
		}
	}
	// A fresh matrix is all zeros, pad rows included.
	q = NewTernaryMatrix(20, 9)
	for i := 0; i < 32; i++ {
		for j := 0; j < 9; j++ {
			if i < q.Rows && q.At(i, j) != 0 {
				t.Fatalf("fresh (%d,%d) = %d", i, j, q.At(i, j))
			}
		}
	}
}

func TestTernaryMatVecAndMatMul(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, c := range []struct{ rows, cols int }{
		{1024, 2304}, // parallel path, many groups
		{300, 33},    // partial final group and tile, scalar tails
		{5, 7},
	} {
		q := NewTernaryMatrix(c.rows, c.cols)
		for j := 0; j < c.cols; j++ {
			for g := 0; g*tGroup < c.rows; g++ {
				q.Scale[q.TableIndex(g, j)] = tensai.Float(0.01 + rng.Float64())
			}
			for i := 0; i < c.rows; i++ {
				q.Set(i, j, int8(rng.Intn(3))-1)
			}
		}
		x := make([]tensai.Float, c.rows)
		for i := range x {
			x[i] = tensai.Float(rng.NormFloat64())
		}
		out := make([]tensai.Float, c.cols)
		if err := q.MatVec(x, out); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		// Exact reference over the same weights and 7-bit activations.
		xu, sx := quantizeActs(x)
		for j := 0; j < c.cols; j++ {
			var want float64
			for g := 0; g*tGroup < c.rows; g++ {
				var acc int64
				for i := g * tGroup; i < min((g+1)*tGroup, c.rows); i++ {
					acc += int64(q.At(i, j)) * (int64(xu[i]) - 64)
				}
				want += float64(acc) * float64(q.Scale[q.TableIndex(g, j)])
			}
			want *= float64(sx)
			if diff := math.Abs(float64(out[j]) - want); diff > 1e-4*(1+math.Abs(want)) {
				t.Fatalf("%v col %d: got %v want %v", c, j, out[j], want)
			}
		}
		// The batch must equal per-row MatVec bit for bit, tail rows
		// included.
		const batch = 11
		xb := tensai.RandomMatrix(batch, c.rows, rng)
		ob := tensai.NewMatrix(batch, c.cols)
		if err := q.MatMul(xb, ob); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		for r := 0; r < batch; r++ {
			if err := q.MatVec(xb.Data[r*c.rows:(r+1)*c.rows], out); err != nil {
				t.Fatal(err)
			}
			for j := 0; j < c.cols; j++ {
				if ob.Data[r*c.cols+j] != out[j] {
					t.Fatalf("%v row %d col %d: matmul %v matvec %v", c, r, j, ob.Data[r*c.cols+j], out[j])
				}
			}
		}
	}
}

// The vector kernel and the portable one agree exactly, which the
// dispatch test above only checks on whichever build it runs.
func TestTernaryGenericAgrees(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	const rows, cols = 512, 96
	q := NewTernaryMatrix(rows, cols)
	for j := 0; j < cols; j++ {
		for g := 0; g*tGroup < rows; g++ {
			q.Scale[q.TableIndex(g, j)] = tensai.Float(rng.Float64())
		}
		for i := 0; i < rows; i++ {
			q.Set(i, j, int8(rng.Intn(3))-1)
		}
	}
	x := make([]tensai.Float, rows)
	for i := range x {
		x[i] = tensai.Float(rng.NormFloat64())
	}
	xs, sx, gsum := signedActs(x, rows)
	a, b := make([]tensai.Float, cols), make([]tensai.Float, cols)
	ternaryMatvecCols(a, xs, xsQuads(xs), sx, gsum, q.Q, q.Scale, rows, cols, 0, cols)
	ternaryMatvecColsGeneric(b, xs, sx, gsum, q.Q, q.Scale, rows, cols, 0, cols)
	for j := range a {
		if a[j] != b[j] {
			t.Fatalf("col %d: kernel %v generic %v", j, a[j], b[j])
		}
	}
}

func BenchmarkTernaryMatVec(b *testing.B) {
	const rows, cols = 5120, 34816
	q := NewTernaryMatrix(rows, cols)
	rng := rand.New(rand.NewSource(1))
	for i := range q.Q {
		q.Q[i] = uint8(rng.Intn(256)) & 0xAA // codes 0 or 2 mostly
	}
	for i := range q.Scale {
		q.Scale[i] = 0.01
	}
	x := make([]tensai.Float, rows)
	for i := range x {
		x[i] = tensai.Float(rng.NormFloat64())
	}
	out := make([]tensai.Float, cols)
	b.SetBytes(int64(len(q.Q)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q.MatVec(x, out)
	}
}

func BenchmarkTernaryMatVecSmall(b *testing.B) {
	const rows, cols = 1024, 2048
	q := NewTernaryMatrix(rows, cols)
	rng := rand.New(rand.NewSource(1))
	for i := range q.Q {
		q.Q[i] = uint8(rng.Intn(256)) & 0xAA
	}
	for i := range q.Scale {
		q.Scale[i] = 0.01
	}
	x := make([]tensai.Float, rows)
	for i := range x {
		x[i] = tensai.Float(rng.NormFloat64())
	}
	out := make([]tensai.Float, cols)
	xs, sx, gsum := signedActs(x, rows)
	xq := xsQuads(xs)
	b.SetBytes(int64(rows * cols / 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ternaryMatvecCols(out, xs, xq, sx, gsum, q.Q, q.Scale, rows, cols, 0, cols)
	}
}

func BenchmarkTernaryMatMul32(b *testing.B) {
	const rows, cols, batch = 1536, 8960, 64
	q := NewTernaryMatrix(rows, cols)
	rng := rand.New(rand.NewSource(1))
	for i := range q.Q {
		q.Q[i] = uint8(rng.Intn(256)) & 0xAA
	}
	for i := range q.Scale {
		q.Scale[i] = 0.01
	}
	x := tensai.RandomMatrix(batch, rows, rng)
	out := tensai.NewMatrix(batch, cols)
	b.SetBytes(int64(rows * cols * batch))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q.MatMul(x, out)
	}
}

package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func TestMXFP4MatVecAndMatMul(t *testing.T) {
	rng := rand.New(rand.NewSource(63))
	for _, c := range []struct{ rows, cols int }{
		{768, 2304}, // parallel path, many groups
		{100, 33},   // partial final group and tile, scalar tails
		{5, 7},
	} {
		q := NewMXFP4Matrix(c.rows, c.cols)
		for j := 0; j < c.cols; j++ {
			for g := 0; g*32 < c.rows; g++ {
				q.Scale[q.TableIndex(g, j)] = MXFP4Scale(uint8(120 + rng.Intn(12)))
				var sum int32
				for i := g * 32; i < min((g+1)*32, c.rows); i++ {
					code := uint8(rng.Intn(16))
					q.Q[q.Index(i, j)] |= code << (4 * (i % 2))
					sum += int32(MXFP4Value(code))
				}
				q.ColSum64[q.TableIndex(g, j)] = 64 * sum
			}
		}
		x := make([]Float, c.rows)
		for i := range x {
			x[i] = Float(rng.NormFloat64())
		}
		out := make([]Float, c.cols)
		if err := q.MatVec(x, out); err != nil {
			t.Fatalf("%v: %v", c, err)
		}

		// Exact reference over the same codes and 7-bit activations.
		xu, sx := quantizeActs(x)
		for j := 0; j < c.cols; j++ {
			var want float64
			for g := 0; g*32 < c.rows; g++ {
				var acc int64
				for i := g * 32; i < min((g+1)*32, c.rows); i++ {
					w := int64(MXFP4Value(q.Q[q.Index(i, j)] >> (4 * (i % 2))))
					acc += w * (int64(xu[i]) - 64)
				}
				want += float64(acc) * float64(q.Scale[q.TableIndex(g, j)])
			}
			want *= float64(sx)
			if diff := math.Abs(float64(out[j]) - want); diff > 1e-3*(1+math.Abs(want)) {
				t.Fatalf("%v col %d: got %v want %v", c, j, out[j], want)
			}
		}

		// The batch must equal per-row MatVec bit for bit.
		const batch = 11
		xb := RandomMatrix(batch, c.rows, rng)
		ob := NewMatrix(batch, c.cols)
		if err := q.MatMul(xb, ob); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		row := make([]Float, c.cols)
		for r := 0; r < batch; r++ {
			if err := q.MatVec(xb.Data[r*c.rows:(r+1)*c.rows], row); err != nil {
				t.Fatal(err)
			}
			for j := range row {
				if ob.Data[r*c.cols+j] != row[j] {
					t.Fatalf("%v row %d col %d: batch %v matvec %v", c, r, j, ob.Data[r*c.cols+j], row[j])
				}
			}
		}
	}
}

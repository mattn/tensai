package tensai

import (
	"math"
	"math/rand"
	"testing"
)

// buildQ8G quantizes a float matrix group-wise into the Q8G layout, the
// way a GGUF Q8_0 repack would (each 32-row group of a column shares one
// scale).
func buildQ8G(m *Matrix) *Q8GMatrix {
	q := NewQ8GMatrix(m.Rows, m.Cols)
	groups := (m.Rows + q8Group - 1) / q8Group
	for j := 0; j < m.Cols; j++ {
		for g := 0; g < groups; g++ {
			rlo, rhi := g*q8Group, min((g+1)*q8Group, m.Rows)
			var maxAbs Float
			for i := rlo; i < rhi; i++ {
				v := m.Data[i*m.Cols+j]
				if v < 0 {
					v = -v
				}
				if v > maxAbs {
					maxAbs = v
				}
			}
			s := maxAbs / 127
			q.Scale[g*m.Cols+j] = s
			var sum int32
			for i := rlo; i < rhi; i++ {
				n := 0
				if s != 0 {
					v := m.Data[i*m.Cols+j] / s
					if v >= 0 {
						v += 0.5
					} else {
						v -= 0.5
					}
					n = int(v)
				}
				q.Q[(i/4)*4*m.Cols+4*j+i%4] = int8(n)
				sum += int32(n)
			}
			q.ColSum64[g*m.Cols+j] = 64 * sum
		}
	}
	return q
}

func TestQ8GMatVecAndMatMul(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	for _, c := range []struct{ rows, cols int }{
		{768, 2304}, // parallel path, many groups
		{100, 33},   // partial final group, scalar tails
		{33, 10},    // odd rows inside one full group plus a stub
		{5, 7},
	} {
		m := RandomMatrix(c.rows, c.cols, rng)
		q := buildQ8G(m)
		x := make([]Float, c.rows)
		for i := range x {
			x[i] = Float(rng.NormFloat64())
		}
		out := make([]Float, c.cols)
		if err := q.MatVec(x, out); err != nil {
			t.Fatalf("%v: %v", c, err)
		}

		// Exact reference over the same weights and 7-bit activations.
		xu, sx := quantizeActs(x)
		groups := (c.rows + q8Group - 1) / q8Group
		for j := 0; j < c.cols; j++ {
			var want float64
			for g := 0; g < groups; g++ {
				rlo, rhi := g*q8Group, min((g+1)*q8Group, c.rows)
				var acc int64
				for i := rlo; i < rhi; i++ {
					w := int64(q.Q[(i/4)*4*c.cols+4*j+i%4])
					acc += w * (int64(xu[i]) - 64)
				}
				want += float64(acc) * float64(q.Scale[g*c.cols+j])
			}
			want *= float64(sx)
			if diff := math.Abs(float64(out[j]) - want); diff > 1e-3*(1+math.Abs(want)) {
				t.Fatalf("%v col %d: got %v want %v", c, j, out[j], want)
			}
			var full float64
			for i := 0; i < c.rows; i++ {
				full += float64(x[i]) * float64(m.Data[i*c.cols+j])
			}
			if diff := math.Abs(want - full); diff > 2 {
				t.Fatalf("%v col %d: quantization error %v too large", c, j, diff)
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

	q := NewQ8GMatrix(8, 8)
	if err := q.MatVec(make([]Float, 7), make([]Float, 8)); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

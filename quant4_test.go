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

func TestQuantize4Group32(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	for _, c := range []struct{ rows, cols int }{
		{256, 96}, // several 32-row groups
		{100, 33}, // partial final group
	} {
		m := RandomMatrix(c.rows, c.cols, rng)
		// Build a Group-32 matrix by quantizing per 32-row group, the way
		// a GGUF Q4_0 repack does.
		quads := (c.rows + 3) / 4
		groups := (c.rows + 31) / 32
		q := &Q4Matrix{
			Rows:  c.rows,
			Cols:  c.cols,
			Q:     make([]uint8, quads*2*c.cols+32),
			Scale: make([]Float, groups*c.cols),
			Group: 32,
		}
		for j := 0; j < c.cols; j++ {
			for g := 0; g < groups; g++ {
				rlo, rhi := g*32, min((g+1)*32, c.rows)
				var maxAbs Float
				for i := rlo; i < rhi; i++ {
					v := m.Data[i*c.cols+j]
					if v < 0 {
						v = -v
					}
					if v > maxAbs {
						maxAbs = v
					}
				}
				s := maxAbs / 7
				q.Scale[g*c.cols+j] = s
				for i := rlo; i < rhi; i++ {
					n := 0
					if s != 0 {
						v := m.Data[i*c.cols+j] / s
						if v >= 0 {
							v += 0.5
						} else {
							v -= 0.5
						}
						n = int(v)
						if n < -8 {
							n = -8
						} else if n > 7 {
							n = 7
						}
					}
					q.Q[(i/4)*2*c.cols+2*j+(i%4)/2] |= uint8(n+8) << (4 * (i % 2))
				}
			}
			for i := c.rows; i < 4*quads; i++ {
				q.Q[(i/4)*2*c.cols+2*j+(i%4)/2] |= 8 << (4 * (i % 2))
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
		xu, sx := quantizeActs(x)
		for j := 0; j < c.cols; j++ {
			var want float64
			for g := 0; g < groups; g++ {
				rlo, rhi := g*32, min((g+1)*32, c.rows)
				var acc, gs int64
				for i := rlo; i < rhi; i++ {
					nib := int64(q.Q[(i/4)*2*c.cols+2*j+(i%4)/2]>>(4*(i%2))) & 0x0F
					xs := int64(xu[i]) - 64
					acc += nib * xs
					gs += xs
				}
				want += float64(acc-8*gs) * float64(q.Scale[g*c.cols+j])
			}
			want *= float64(sx)
			if diff := math.Abs(float64(out[j]) - want); diff > 1e-3*(1+math.Abs(want)) {
				t.Fatalf("%v col %d: got %v want %v", c, j, out[j], want)
			}
		}

		// Batch equals per-row MatVec bit for bit at this group length too.
		xb := RandomMatrix(6, c.rows, rng)
		ob := NewMatrix(6, c.cols)
		if err := q.MatMul(xb, ob); err != nil {
			t.Fatal(err)
		}
		row := make([]Float, c.cols)
		for r := 0; r < 6; r++ {
			if err := q.MatVec(xb.Data[r*c.rows:(r+1)*c.rows], row); err != nil {
				t.Fatal(err)
			}
			for j := range row {
				if ob.Data[r*c.cols+j] != row[j] {
					t.Fatalf("%v row %d col %d differs", c, r, j)
				}
			}
		}
	}
}

func TestQuantize4MinForm(t *testing.T) {
	rng := rand.New(rand.NewSource(44))
	for _, c := range []struct{ rows, cols int }{
		{256, 96},
		{100, 33}, // partial final group
	} {
		m := RandomMatrix(c.rows, c.cols, rng)
		// Asymmetric group quantization: value = scale*q - min, q in 0..15
		// — the form GGUF's Q4_K sub-blocks carry.
		quads := (c.rows + 3) / 4
		groups := (c.rows + 31) / 32
		q := &Q4Matrix{
			Rows:     c.rows,
			Cols:     c.cols,
			Q:        make([]uint8, quads*2*c.cols+32),
			ScaleMin: make([]uint32, groups*c.cols),
			Group:    32,
		}
		for j := 0; j < c.cols; j++ {
			for g := 0; g < groups; g++ {
				rlo, rhi := g*32, min((g+1)*32, c.rows)
				lo, hi := m.Data[rlo*c.cols+j], m.Data[rlo*c.cols+j]
				for i := rlo; i < rhi; i++ {
					v := m.Data[i*c.cols+j]
					if v < lo {
						lo = v
					}
					if v > hi {
						hi = v
					}
				}
				// value = s*q + lo = s*q - min; the pack rounds the pair
				// to bfloat16, so quantize against what the kernel will
				// actually unpack.
				q.ScaleMin[g*c.cols+j] = PackScaleMin((hi-lo)/15, -lo)
				s, mn := UnpackScaleMin(q.ScaleMin[g*c.cols+j])
				lo = -mn
				for i := rlo; i < rhi; i++ {
					n := 0
					if s != 0 {
						n = int((m.Data[i*c.cols+j]-lo)/s + 0.5)
						if n > 15 {
							n = 15
						}
					}
					q.Q[(i/4)*2*c.cols+2*j+(i%4)/2] |= uint8(n) << (4 * (i % 2))
				}
			}
			// Pad rows encode q=0; their min contribution must vanish, so
			// the pad rows rely on activation pads being exactly 64 (xs=0).
		}
		x := make([]Float, c.rows)
		for i := range x {
			x[i] = Float(rng.NormFloat64())
		}
		out := make([]Float, c.cols)
		if err := q.MatVec(x, out); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		xu, sx := quantizeActs(x)
		for j := 0; j < c.cols; j++ {
			var want float64
			for g := 0; g < groups; g++ {
				rlo, rhi := g*32, min((g+1)*32, c.rows)
				var acc, gs int64
				for i := rlo; i < rhi; i++ {
					qv := int64(q.Q[(i/4)*2*c.cols+2*j+(i%4)/2]>>(4*(i%2))) & 0x0F
					xs := int64(xu[i]) - 64
					acc += qv * xs
					gs += xs
				}
				sc, mn := UnpackScaleMin(q.ScaleMin[g*c.cols+j])
				want += float64(acc)*float64(sc) - float64(gs)*float64(mn)
			}
			want *= float64(sx)
			if diff := math.Abs(float64(out[j]) - want); diff > 1e-3*(1+math.Abs(want)) {
				t.Fatalf("%v col %d: got %v want %v", c, j, out[j], want)
			}
			// And the whole thing stays close to the float product.
			var full float64
			for i := 0; i < c.rows; i++ {
				full += float64(x[i]) * float64(m.Data[i*c.cols+j])
			}
			if diff := math.Abs(want - full); diff > 2 {
				t.Fatalf("%v col %d: quantization error %v too large", c, j, diff)
			}
		}

		xb := RandomMatrix(6, c.rows, rng)
		ob := NewMatrix(6, c.cols)
		if err := q.MatMul(xb, ob); err != nil {
			t.Fatal(err)
		}
		row := make([]Float, c.cols)
		for r := 0; r < 6; r++ {
			if err := q.MatVec(xb.Data[r*c.rows:(r+1)*c.rows], row); err != nil {
				t.Fatal(err)
			}
			for j := range row {
				if ob.Data[r*c.cols+j] != row[j] {
					t.Fatalf("%v row %d col %d differs", c, r, j)
				}
			}
		}
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

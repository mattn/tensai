package quant

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai"
)

func TestQuantizeMatVec(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 0))
	for _, c := range []struct{ rows, cols int }{
		{768, 2304}, // big enough for the parallel path
		{16, 16},
		{7, 13}, // scalar tails on every chunk
		{1, 9},
	} {
		m := tensai.RandomMatrix(c.rows, c.cols, rng)
		q := Quantize(m)
		x := make([]tensai.Float, c.rows)
		for i := range x {
			x[i] = tensai.Float(rng.NormFloat64())
		}
		out := make([]tensai.Float, c.cols)
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

	q := Quantize(tensai.NewMatrix(4, 4)) // all zeros: scales are zero
	out := make([]tensai.Float, 4)
	if err := q.MatVec(make([]tensai.Float, 4), out); err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		if v != 0 {
			t.Fatalf("zero matrix produced %v", out)
		}
	}
	if err := q.MatVec(make([]tensai.Float, 3), out); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

// The Big pair exceeds the L3 cache, which is the regime inference decode
// lives in: there the f32 matvec is memory-bandwidth bound and the int8
// weights pull four times less.
func BenchmarkMatVecF32Big(b *testing.B) {
	rng := rand.New(rand.NewPCG(33, 0))
	w := tensai.RandomMatrix(4096, 16384, rng)
	x := tensai.NewMatrix(1, 4096)
	for i := range x.Data {
		x.Data[i] = tensai.Float(rng.NormFloat64())
	}
	out := tensai.NewMatrix(1, 16384)
	b.SetBytes(4096 * 16384 * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := tensai.DotInto(out, x, w); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecQ8Big(b *testing.B) {
	rng := rand.New(rand.NewPCG(33, 0))
	q := Quantize(tensai.RandomMatrix(4096, 16384, rng))
	x := make([]tensai.Float, 4096)
	for i := range x {
		x[i] = tensai.Float(rng.NormFloat64())
	}
	out := make([]tensai.Float, 16384)
	b.SetBytes(4096 * 16384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecF32(b *testing.B) {
	rng := rand.New(rand.NewPCG(32, 0))
	w := tensai.RandomMatrix(768, 2304, rng)
	x := tensai.NewMatrix(1, 768)
	for i := range x.Data {
		x.Data[i] = tensai.Float(rng.NormFloat64())
	}
	out := tensai.NewMatrix(1, 2304)
	b.SetBytes(768 * 2304 * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := tensai.DotInto(out, x, w); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecQ8(b *testing.B) {
	rng := rand.New(rand.NewPCG(32, 0))
	q := Quantize(tensai.RandomMatrix(768, 2304, rng))
	x := make([]tensai.Float, 768)
	for i := range x {
		x[i] = tensai.Float(rng.NormFloat64())
	}
	out := make([]tensai.Float, 2304)
	b.SetBytes(768 * 2304)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func TestQMatMulBatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(51, 0))
	for _, c := range []struct{ batch, rows, cols int }{
		{11, 768, 2304}, // 8-row blocks plus a remainder
		{8, 64, 33},     // scalar column tails
		{2, 5, 8},       // pure remainder path
	} {
		w := tensai.RandomMatrix(c.rows, c.cols, rng)
		q := Quantize(w)
		x := tensai.RandomMatrix(c.batch, c.rows, rng)
		out := tensai.NewMatrix(c.batch, c.cols)
		if err := q.MatMul(x, out); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		// The batch must equal per-row MatVec bit for bit: identical
		// activation quantization, identical integer accumulation.
		row := make([]tensai.Float, c.cols)
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

	q := Quantize(tensai.NewMatrix(4, 4))
	if err := q.MatMul(tensai.NewMatrix(2, 3), tensai.NewMatrix(2, 4)); err == nil {
		t.Fatal("expected shape mismatch error")
	}

	for _, c := range []struct{ batch, rows, cols int }{
		{9, 768, 2304}, // 4-row blocks plus a remainder, parallel path
		{4, 130, 33},   // partial final group, scalar column tails
		{6, 33, 10},    // odd rows: pad pair inside the first group
		{2, 5, 7},      // pure remainder path
	} {
		w := tensai.RandomMatrix(c.rows, c.cols, rng)
		q4, err := Quantize4(w)
		if err != nil {
			t.Fatal(err)
		}
		x := tensai.RandomMatrix(c.batch, c.rows, rng)
		out := tensai.NewMatrix(c.batch, c.cols)
		if err := q4.MatMul(x, out); err != nil {
			t.Fatalf("q4 %v: %v", c, err)
		}
		row := make([]tensai.Float, c.cols)
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
	rng := rand.New(rand.NewPCG(52, 0))
	q := Quantize(tensai.RandomMatrix(1536, 8960, rng))
	x := tensai.RandomMatrix(64, 1536, rng)
	out := tensai.NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatMul(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQ4PrefillBatched(b *testing.B) {
	rng := rand.New(rand.NewPCG(52, 0))
	q, err := Quantize4(tensai.RandomMatrix(1536, 8960, rng))
	if err != nil {
		b.Fatal(err)
	}
	x := tensai.RandomMatrix(64, 1536, rng)
	out := tensai.NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatMul(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQ4PrefillRowwise(b *testing.B) {
	rng := rand.New(rand.NewPCG(52, 0))
	q, err := Quantize4(tensai.RandomMatrix(1536, 8960, rng))
	if err != nil {
		b.Fatal(err)
	}
	x := tensai.RandomMatrix(64, 1536, rng)
	out := tensai.NewMatrix(64, 8960)
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
	rng := rand.New(rand.NewPCG(52, 0))
	q := Quantize(tensai.RandomMatrix(1536, 8960, rng))
	x := tensai.RandomMatrix(64, 1536, rng)
	out := tensai.NewMatrix(64, 8960)
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
	rng := rand.New(rand.NewPCG(77, 0))
	for _, n := range []int{5, 16, 31, 1536, 4099} {
		x := make([]tensai.Float, n)
		for i := range x {
			x[i] = tensai.Float(rng.NormFloat64())
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

// BenchmarkQ4PrefillMinForm is the batched matmul over the asymmetric
// Group-32 form GGUF's Q4_K repacks into — the shape a Q4_K_M prefill
// actually runs.
func BenchmarkQ4PrefillMinForm(b *testing.B) {
	rng := rand.New(rand.NewPCG(52, 0))
	q := NewQ4Matrix(1536, 8960, 32, true)
	for i := range q.Q {
		q.Q[i] = uint8(rng.IntN(256))
	}
	for i := range q.ScaleMin {
		q.ScaleMin[i] = PackScaleMin(0.01, 0.2)
	}
	x := tensai.RandomMatrix(64, 1536, rng)
	out := tensai.NewMatrix(64, 8960)
	b.SetBytes(int64(64 * 1536 * 8960))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatMul(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

// quantizeColumnsRef is the column-at-a-time quantizer the tiled one
// replaced: the reference the tiles and their vector bodies answer to.
func quantizeColumnsRef(m *tensai.Matrix, q *QMatrix, colLo, colHi int) {
	for j := colLo; j < colHi; j++ {
		var maxAbs tensai.Float
		for i := 0; i < m.Rows; i++ {
			v := m.Data[i*m.Cols+j]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		s := maxAbs / 127
		q.Scale[j] = s
		if s == 0 {
			continue
		}
		inv := 1 / s
		var sum int32
		for i := 0; i < m.Rows; i++ {
			v := m.Data[i*m.Cols+j] * inv
			if v >= 0 {
				v += 0.5
			} else {
				v -= 0.5
			}
			w := int8(v)
			q.Q[q.Index(i, j)] = w
			sum += int32(w)
		}
		q.ColSum64[j] = 64 * sum
	}
}

// TestQuantizeTiledMatchesColumns pins the tiled quantizer (and the
// vector bodies the build picks for it) to the column-at-a-time
// reference, over shapes that straddle the tile width and the row quad,
// and over columns whose scale underflows to zero.
func TestQuantizeTiledMatchesColumns(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 5))
	for _, shape := range [][2]int{{1, 1}, {3, 5}, {4, 32}, {7, 33}, {64, 31}, {65, 64}, {130, 97}} {
		rows, cols := shape[0], shape[1]
		m := tensai.NewMatrix(rows, cols)
		for i := range m.Data {
			m.Data[i] = tensai.Float(rng.NormFloat64())
		}
		// A zero column, a denormal one (maxAbs/127 underflows to zero),
		// and a column carrying the largest magnitude of the matrix.
		for i := 0; i < rows; i++ {
			m.Data[i*cols] = 0
			if cols > 1 {
				m.Data[i*cols+1] = math.Float32frombits(uint32(i%3) + 1)
			}
			if cols > 2 && i == rows/2 {
				m.Data[i*cols+2] = -1e30
			}
		}
		got := Quantize(m)

		quads := (rows + 3) / 4
		want := &QMatrix{
			Rows:     rows,
			Cols:     cols,
			Q:        make([]int8, ((cols+q4Tile-1)/q4Tile)*quads*4*q4Tile+32),
			Scale:    make([]tensai.Float, cols),
			ColSum64: make([]int32, cols+8),
		}
		quantizeColumnsRef(m, want, 0, cols)

		for j := 0; j < cols; j++ {
			if got.Scale[j] != want.Scale[j] {
				t.Fatalf("%dx%d col %d: scale %v want %v", rows, cols, j, got.Scale[j], want.Scale[j])
			}
			if got.ColSum64[j] != want.ColSum64[j] {
				t.Fatalf("%dx%d col %d: colsum %d want %d", rows, cols, j, got.ColSum64[j], want.ColSum64[j])
			}
			for i := 0; i < rows; i++ {
				if g, w := got.Q[got.Index(i, j)], want.Q[want.Index(i, j)]; g != w {
					t.Fatalf("%dx%d (%d,%d): q %d want %d", rows, cols, i, j, g, w)
				}
			}
		}
	}
}

// BenchmarkQuantize sizes the quantize-at-load pass, which a whole
// checkpoint goes through once. quantizeColumnsRef is the old
// column-at-a-time body, kept here as the thing to beat.
func BenchmarkQuantize(b *testing.B) {
	const rows, cols = 4096, 4096
	m := tensai.NewMatrix(rows, cols)
	rng := rand.New(rand.NewPCG(3, 9))
	for i := range m.Data {
		m.Data[i] = tensai.Float(rng.NormFloat64())
	}
	quads := (rows + 3) / 4
	fresh := func() *QMatrix {
		return &QMatrix{
			Rows: rows, Cols: cols,
			Q:        make([]int8, ((cols+q4Tile-1)/q4Tile)*quads*4*q4Tile+32),
			Scale:    make([]tensai.Float, cols),
			ColSum64: make([]int32, cols+8),
		}
	}
	b.Run("tiled", func(b *testing.B) {
		q := fresh()
		b.SetBytes(int64(rows) * int64(cols) * 4)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			quantizeColumns(m, q, 0, cols)
		}
	})
	b.Run("columns", func(b *testing.B) {
		q := fresh()
		b.SetBytes(int64(rows) * int64(cols) * 4)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			quantizeColumnsRef(m, q, 0, cols)
		}
	})
}

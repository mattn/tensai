package quant

import (
	"testing"

	"github.com/mattn/tensai"
)

func benchmarkQMatrix(rows, cols int) *QMatrix {
	q := &QMatrix{
		Rows:     rows,
		Cols:     cols,
		Q:        make([]int8, ((cols+q4Tile-1)/q4Tile)*((rows+3)/4)*4*q4Tile+32),
		Scale:    make([]tensai.Float, cols),
		ColSum64: make([]int32, cols+8),
	}
	for i := range q.Q {
		q.Q[i] = int8(i%127 - 63)
	}
	for i := range q.Scale {
		q.Scale[i] = 1.0 / 127
	}
	return q
}

func benchmarkQ8Shape(b *testing.B, rows, cols int) {
	q := benchmarkQMatrix(rows, cols)
	x := make([]tensai.Float, rows)
	for i := range x {
		x[i] = tensai.Float(i%31-15) / 16
	}
	out := make([]tensai.Float, cols)
	b.SetBytes(int64(rows * cols))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

// These are Qwen2.5-0.5B's decode projections. Construct packed matrices
// directly so setup does not need much larger temporary float32 matrices.
func BenchmarkQ8QKVProjection(b *testing.B)    { benchmarkQ8Shape(b, 896, 1152) }
func BenchmarkQ8GateUpProjection(b *testing.B) { benchmarkQ8Shape(b, 896, 9728) }
func BenchmarkQ8DownProjection(b *testing.B)   { benchmarkQ8Shape(b, 4864, 896) }
func BenchmarkQ8VocabProjection(b *testing.B)  { benchmarkQ8Shape(b, 896, 151936) }

// Cold cycles through one down projection per transformer layer, matching
// decode where the next matrix displaces the previous one from shared cache.
func BenchmarkQ8DownProjectionCold(b *testing.B) {
	const rows, cols, layers = 4864, 896, 24
	qs := make([]*QMatrix, layers)
	for i := range qs {
		qs[i] = benchmarkQMatrix(rows, cols)
	}
	x := make([]tensai.Float, rows)
	out := make([]tensai.Float, cols)
	b.SetBytes(rows * cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := qs[i%layers].MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}

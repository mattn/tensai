//go:build goexperiment.simd && amd64

package tensai

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// The tiled kernel accumulates in the same order as the row kernel, so on
// finite inputs the two agree to the bit, including the column and row
// tails the tiles leave to the row kernel and inputs with zeros in a.
func TestDotRowsTiledMatchesAxpy(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sh := range [][3]int{
		{4, 16, 16}, {7, 33, 17}, {12, 300, 48}, {9, 600, 70}, {32, 1, 64}, {5, 257, 31}, {64, 512, 144},
	} {
		m, k, n := sh[0], sh[1], sh[2]
		a, b := NewMatrix(m, k), NewMatrix(k, n)
		for i := range a.Data {
			if rng.IntN(4) > 0 { // leave zeros for the row kernel to skip
				a.Data[i] = Float(rng.NormFloat64())
			}
		}
		for i := range b.Data {
			b.Data[i] = Float(rng.NormFloat64())
		}
		got, want := NewMatrix(m, n), NewMatrix(m, n)
		dotRows(got, a, b, 0, m)
		dotRowsAxpy(want, a, b, 0, m, 0)
		for i := range got.Data {
			if math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i]) && got.Data[i] != want.Data[i] {
				t.Fatalf("%dx%dx%d: element %d = %v, row kernel %v", m, k, n, i, got.Data[i], want.Data[i])
			}
		}
	}
}

func BenchmarkDotTall(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sh := range [][3]int{{428, 896, 4864}, {428, 4864, 896}, {428, 896, 896}} {
		m, k, n := sh[0], sh[1], sh[2]
		x, w, out := NewMatrix(m, k), NewMatrix(k, n), NewMatrix(m, n)
		for i := range x.Data {
			x.Data[i] = Float(rng.NormFloat64())
		}
		for i := range w.Data {
			w.Data[i] = Float(rng.NormFloat64())
		}
		b.Run(fmt.Sprintf("NN/%dx%dx%d", m, k, n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := DotInto(out, x, w); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		// The input gradient of x*w: out * w^T, back to x's shape.
		g := NewMatrix(m, k)
		b.Run(fmt.Sprintf("NT/%dx%dx%d", m, n, k), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := DotTBInto(g, out, w); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

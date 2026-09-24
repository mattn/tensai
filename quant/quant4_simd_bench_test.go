//go:build goexperiment.simd && amd64

package quant

import (
	"testing"

	"github.com/mattn/tensai"
)

func BenchmarkQ4Kernel(b *testing.B) {
	const rows, cols = 896, 4096
	q := NewQ4Matrix(rows, cols, 64, false)
	x := make([]tensai.Float, rows)
	for i := range x {
		x[i] = tensai.Float(i%31-15) / 16
	}
	for i := range q.Q {
		q.Q[i] = uint8(i)
	}
	for i := range q.Scale {
		q.Scale[i] = 1.0 / 7
	}
	xu, sx := quantizeActs(x)
	xq := packQuads(xu)
	gsum := make([]int32, rows/64)
	groupSums(gsum, xu, 64)
	out := make([]tensai.Float, cols)
	for _, tc := range []struct {
		name string
		f    func([]tensai.Float, []uint8, []uint32, tensai.Float, []int32, []uint8, []tensai.Float, []uint32, int, int, int, int)
	}{
		{"wide", q4matvecColsWide}, {"narrow", q4matvecColsNarrow},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(rows * cols / 2)
			for i := 0; i < b.N; i++ {
				tc.f(out, xu, xq, sx, gsum, q.Q, q.Scale, nil, 64, cols, 0, cols)
			}
		})
	}
}

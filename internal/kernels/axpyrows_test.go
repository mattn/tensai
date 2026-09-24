package kernels

import (
	"math/rand/v2"
	"testing"
)

func TestAxpyRowsMatchesAxpy(t *testing.T) {
	rng := rand.New(rand.NewPCG(75, 9))
	for _, d := range []int{0, 7, 16, 63, 64, 65, 128, 129} {
		for _, n := range []int{0, 1, 5, 97} {
			const off = 7
			rows := make([][]float32, n)
			ws := make([]float32, n)
			out := make([]float32, d)
			for i := range out {
				out[i] = float32(rng.NormFloat64())
			}
			want := append([]float32(nil), out...)
			for i := range rows {
				rows[i] = make([]float32, off+d)
				ws[i] = float32(rng.NormFloat64())
				for j := range rows[i] {
					rows[i][j] = float32(rng.NormFloat64())
				}
				Axpy(ws[i], rows[i][off:], want)
			}
			AxpyRows(out, ws, rows, off)
			for i := range out {
				if out[i] != want[i] {
					t.Fatalf("d=%d n=%d elem=%d: got %v want %v", d, n, i, out[i], want[i])
				}
			}
		}
	}
}

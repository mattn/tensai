package tensai

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai/internal/kernels"
)

func TestDotTBBlockedMatchesRowwise(t *testing.T) {
	for _, n := range []int{16, 17, 22, 23, 64, 65, 134, 135} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(41, 0))
			a, b := RandomMatrix(257, 4097, rng), RandomMatrix(n, 4097, rng)
			got, want := NewMatrix(a.Rows, n), NewMatrix(a.Rows, n)
			for r := 0; r < a.Rows; r++ {
				kernels.DotVecs(b.Data, a.Data[r*a.Cols:(r+1)*a.Cols], want.Data[r*n:(r+1)*n])
			}
			if err := DotTBInto(got, a, b); err != nil {
				t.Fatal(err)
			}
			for i, v := range want.Data {
				if math.Float32bits(got.Data[i]) != math.Float32bits(v) {
					t.Fatalf("element %d: got %g want %g", i, got.Data[i], v)
				}
			}
		})
	}
}

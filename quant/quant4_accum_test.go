package quant

import (
	"fmt"
	"testing"

	"github.com/mattn/tensai"
)

// All-positive/all-negative products reach the narrow accumulator's
// bound: 16 quads * 2 products * 15 * 63 = 30240 per int16 lane.
// Groups larger than 64 must still use the wide accumulation path.
func TestQ4MatVecAccumulatorBounds(t *testing.T) {
	for _, group := range []int{16, 32, 64, 128} {
		for _, minForm := range []bool{false, true} {
			for _, sign := range []int{-1, 0, 1} {
				for _, nibble := range []uint8{0, 15} {
					t.Run(fmt.Sprintf("group%d/min%t/sign%d/nibble%d", group, minForm, sign, nibble), func(t *testing.T) {
						const cols = 97     // three SIMD tiles and a scalar tail
						rows := 2*group + 3 // partial final group and quad
						q := NewQ4Matrix(rows, cols, group, minForm)
						for i := range q.Q {
							q.Q[i] = nibble | nibble<<4
						}
						for i := range q.Scale {
							q.Scale[i] = 0.125
						}
						for i := range q.ScaleMin {
							q.ScaleMin[i] = PackScaleMin(0.125, 0.5)
						}
						x := make([]tensai.Float, rows)
						for i := range x {
							x[i] = tensai.Float(sign)
							if sign == 0 {
								x[i] = tensai.Float(2*(i%2) - 1)
							}
						}
						xu, sx := quantizeActs(x)
						gsum := make([]int32, (rows+group-1)/group)
						groupSums(gsum, xu, group)
						want := make([]tensai.Float, cols)
						q4matvecColsGeneric(want, xu, sx, gsum, q.Q, q.Scale, q.ScaleMin, group, cols, 0, cols)
						got := make([]tensai.Float, cols)
						if err := q.MatVec(x, got); err != nil {
							t.Fatal(err)
						}
						for i := range got {
							if got[i] != want[i] {
								t.Fatalf("column %d: got %v, want %v", i, got[i], want[i])
							}
						}
					})
				}
			}
		}
	}
}

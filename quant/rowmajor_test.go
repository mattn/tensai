package quant

import (
	"fmt"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/workpool"
)

func TestPackRowMajor(t *testing.T) {
	for _, rows := range []int{1, 2, 3, 4, 5, 17, 128} {
		for _, cols := range []int{1, 3, 4, 5, 31, 32, 33, 63, 64, 65, 129} {
			t.Run(fmt.Sprintf("%dx%d", rows, cols), func(t *testing.T) {
				q := Quantize(tensai.NewMatrix(rows, cols))
				for i := range q.Q {
					q.Q[i] = int8(i*37 + 131)
				}
				got := q.PackRowMajor()
				words := (cols + 3) / 4
				if len(got) != rows*words {
					t.Fatal("wrong length")
				}
				for r := 0; r < rows; r++ {
					for c := 0; c < words*4; c++ {
						value := int8(got[r*words+c/4] >> uint(8*(c%4)))
						var want int8
						if c < cols {
							want = q.Q[q.Index(r, c)]
						}
						if value != want {
							t.Fatalf("(%d,%d): got %d want %d", r, c, value, want)
						}
					}
				}
			})
		}
	}
}
func BenchmarkPackRowMajor(b *testing.B) {
	const rows, cols = 4096, 12288
	q := &QMatrix{Rows: rows, Cols: cols, Q: make([]int8, rows*cols+32)}
	for i := range q.Q {
		q.Q[i] = int8(i * 37)
	}
	b.Run("before", func(b *testing.B) {
		b.SetBytes(rows * cols)
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			words := (cols + 3) / 4
			packed := make([]uint32, rows*words)
			workpool.Bulk(rows, 1, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					for j := 0; j < cols; j++ {
						packed[i*words+j/4] |= uint32(uint8(q.Q[q.Index(i, j)])) << uint(8*(j%4))
					}
				}
			})
		}
	})
	b.Run("after", func(b *testing.B) {
		b.SetBytes(rows * cols)
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			_ = q.PackRowMajor()
		}
	})
}

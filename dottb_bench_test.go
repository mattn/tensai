package tensai

import (
	"fmt"
	"testing"
)

// Decoder convolutions multiply many pixel rows by the same wide filters.
func BenchmarkDotTBDecoder(b *testing.B) {
	for _, tc := range []struct{ m, k, n int }{{1024, 10368, 1152}, {4096, 2592, 288}} {
		b.Run(fmt.Sprintf("%dx%dx%d", tc.m, tc.k, tc.n), func(b *testing.B) {
			x, w := NewMatrix(tc.m, tc.k), NewMatrix(tc.n, tc.k)
			out := NewMatrix(tc.m, tc.n)
			for i := range x.Data {
				x.Data[i] = Float(i%31-15) / 32
			}
			for i := range w.Data {
				w.Data[i] = Float(i%23-11) / 32
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := DotTBInto(out, x, w); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

package qwenimage

import (
	"testing"

	"github.com/mattn/tensai"
)

// One block's attention at a 512x512 image's shape: 1024 image tokens
// after a short prompt, every head, as Forward runs it.
func BenchmarkAttention(b *testing.B) {
	const n, text = 1038, 14
	q := tensai.NewMatrix(n, ditDim)
	k := tensai.NewMatrix(n, ditDim)
	v := tensai.NewMatrix(n, ditDim)
	for i := range q.Data {
		q.Data[i] = tensai.Float(i%17-8) / 64
		k.Data[i] = tensai.Float(i%13-6) / 64
		v.Data[i] = tensai.Float(i%11-5) / 64
	}
	limit := make([]int, n)
	for r := range limit {
		if r < text {
			limit[r] = r + 1 // the prompt is causal
		} else {
			limit[r] = n // the image reads everything
		}
	}
	out := tensai.NewMatrix(n, ditDim)
	s := NewScratch(n)
	s.reset(n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := attention(out, q, k, v, limit, s); err != nil {
			b.Fatal(err)
		}
	}
}

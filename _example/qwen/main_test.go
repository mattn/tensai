package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestSampleDistribution(t *testing.T) {
	logits := []float32{0, 1, 2, -1}
	p := sampleDistribution(logits, 0.8, 1)
	var sum float64
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("distribution sums to %g", sum)
	}
	if !(p[2] > p[1] && p[1] > p[0] && p[0] > p[3]) {
		t.Fatalf("probabilities do not follow logits: %v", p)
	}

	p = sampleDistribution(logits, 1, 0.5)
	nonzero := 0
	for _, v := range p {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero != 1 || p[2] != 1 {
		t.Fatalf("top-p nucleus = %v, want only token 2", p)
	}
}

func BenchmarkPrefill(b *testing.B) {
	m, _, err := loadGGUF("../../qwen2.5-0.5b-instruct-q8_0.gguf", 8, true, true)
	if err != nil {
		b.Skip(err)
	}
	tokens := make([]int, 512)
	for i := range tokens {
		tokens[i] = (i * 97) % 32000
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.truncate(0)
		m.prefill(tokens, 0)
	}
}

func BenchmarkSample(b *testing.B) {
	logits := make([]float32, 151936)
	for i := range logits {
		logits[i] = float32((i*7919)%10000)/1000 - 10
	}
	rng := rand.New(rand.NewSource(1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sample(logits, 0.8, 0.9, rng)
	}
}

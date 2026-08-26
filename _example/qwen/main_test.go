package main

import (
	"math/rand"
	"testing"
)

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

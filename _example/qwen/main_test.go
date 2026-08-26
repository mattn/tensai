package main

import (
	"math/rand"
	"testing"
)

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

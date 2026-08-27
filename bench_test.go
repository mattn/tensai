package tensai

import (
	"math/rand"
	"testing"
)

func benchmarkDot(b *testing.B, size int) {
	rng := rand.New(rand.NewSource(1))
	x := RandomMatrix(size, size, rng)
	y := RandomMatrix(size, size, rng)
	b.SetBytes(int64(size * size * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Dot(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDot128(b *testing.B) { benchmarkDot(b, 128) }

func BenchmarkDot512(b *testing.B) { benchmarkDot(b, 512) }

func BenchmarkTranspose1024(b *testing.B) {
	src := NewMatrix(1024, 1024)
	dst := NewMatrix(1024, 1024)
	b.SetBytes(int64(len(src.Data) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TInto(dst, src); err != nil {
			b.Fatal(err)
		}
	}
}

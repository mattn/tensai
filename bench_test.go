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

func BenchmarkFitStepMLP(b *testing.B) {
	model := NewSequential()
	model.Add(NewDense(256))
	model.Add(&ReLU{})
	model.Add(NewDense(128))
	model.Add(&ReLU{})
	model.Add(NewDense(4))
	if err := model.Compile(10, SoftmaxCrossEntropy{}, NewAdam(0.005)); err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	in := NewMatrix(64, 10)
	tgt := NewMatrix(64, 1)
	for i := range in.Data {
		in.Data[i] = Float(rng.Intn(2))
	}
	for i := range tgt.Data {
		tgt.Data[i] = Float(rng.Intn(4))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.FitStep(in, tgt); err != nil {
			b.Fatal(err)
		}
	}
}

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

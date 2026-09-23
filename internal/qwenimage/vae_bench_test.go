package qwenimage

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
)

func BenchmarkDecoder512(b *testing.B) {
	home, err := os.UserHomeDir()
	if err != nil {
		b.Fatal(err)
	}
	path := filepath.Join(home, ".cache", "tensai", "Qwen-Image-2.1", "vae", "diffusion_pytorch_model.safetensors")
	if _, err := os.Stat(path); err != nil {
		b.Skipf("needs the VAE checkpoint: %v", err)
	}
	d, err := LoadDecoder(path)
	if err != nil {
		b.Fatal(err)
	}
	z := tensai.NewTensor(64, 32, 32)
	for i := range z.Data {
		z.Data[i] = tensai.Float(math.Sin(float64(i) * 0.13))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Decode(d, z); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecoderShortcut(b *testing.B) {
	x := tensai.NewMatrix(128*128, 576)
	for i := range x.Data {
		x.Data[i] = tensai.Float(i % 37)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dupUp(x, 128, 128, 288, 2, 1)
	}
}

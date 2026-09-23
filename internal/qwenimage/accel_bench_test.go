//go:build wgpu || wgpu24

package qwenimage

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
)

// BenchmarkGPUBlock512 isolates the dominant operations at 512 pixels.
// Use the existing cache so benchmarking never triggers model quantization.
func BenchmarkGPUBlock512(b *testing.B) {
	home, err := os.UserHomeDir()
	if err != nil {
		b.Fatal(err)
	}
	dir := filepath.Join(home, ".cache", "tensai", "Qwen-Image-2.1", "transformer")
	m := &Transformer{blocks: make([]*Block, ditLayers)}
	for i := range m.blocks {
		m.blocks[i] = &Block{}
	}
	release, err := readCache(dir, 8, m.walk)
	if err != nil {
		b.Skipf("needs an existing q8 cache: %v", err)
	}
	m.release = release
	m.blocks = m.blocks[:1]
	defer m.Close()
	name, _, err := UseGPU(m, 1<<30)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("device: %s", name)
	l := NewLayout(12, 32, 32)
	s := NewScratch(l.Tokens())
	for _, x := range []*tensai.Matrix{s.norm, s.q, s.k, s.v, s.attn} {
		for i := range x.Data {
			x.Data[i] = tensai.Float(math.Sin(float64(i) * 0.13))
		}
	}
	bl := m.blocks[0]
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"qkv", func() error {
			for _, p := range []struct {
				w   *linear
				out *tensai.Matrix
			}{{bl.toQ, s.q}, {bl.toK, s.k}, {bl.toV, s.v}} {
				if err := p.w.apply(p.out, s.norm); err != nil {
					return err
				}
			}
			return nil
		}},
		{"attention", func() error { return bl.attentionOnDevice(s.attn, s.q, s.k, s.v, l, s) }},
		{"output", func() error { return bl.toOut.apply(s.norm, s.attn) }},
		{"ffn", func() error { return bl.mlpOnDevice(s.attn, s.norm) }},
	} {
		b.Run(tc.name, func(b *testing.B) {
			if err := tc.run(); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := tc.run(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

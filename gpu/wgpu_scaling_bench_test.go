//go:build wgpu || wgpu24

package gpu

import (
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

// benchAttnAt times one 512-query prefill chunk attending seqKV cached
// positions — the shape of the last chunk of a long prompt, where the
// grouped kernel spends its time.
func benchAttnAt(b *testing.B, seqKV int) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(9))
	const heads, kvHeads, dh = 14, 2, 64
	const d, kvDim = heads * dh, kvHeads * dh
	const seqQ = 512
	upload := func(f *tensai.Tensor) *Tensor {
		if g.HasF16() {
			c, err := g.NewF16Tensor(seqKV, kvDim)
			if err != nil {
				b.Fatal(err)
			}
			src, err := g.Upload(f)
			if err != nil {
				b.Fatal(err)
			}
			if err := src.CopyRowsInto(c, 0); err != nil {
				b.Fatal(err)
			}
			src.Free()
			return c
		}
		gt, err := g.Upload(f)
		if err != nil {
			b.Fatal(err)
		}
		return gt
	}
	gk := upload(randTensor(rng, seqKV, kvDim))
	defer gk.Free()
	gv := upload(randTensor(rng, seqKV, kvDim))
	defer gv.Free()
	gq, err := g.Upload(randTensor(rng, seqQ, d))
	if err != nil {
		b.Fatal(err)
	}
	defer gq.Free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
}

func BenchmarkGPUAttnChunk1K(b *testing.B) { benchAttnAt(b, 1024) }
func BenchmarkGPUAttnChunk2K(b *testing.B) { benchAttnAt(b, 2048) }
func BenchmarkGPUAttnChunk4K(b *testing.B) { benchAttnAt(b, 4000) }

//go:build wgpu || wgpu24

package gpu

import (
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/quant"
)

// The fused norm prologue must match the standalone rmsnorm_row dispatch
// followed by the plain matvec bit for bit — greedy decode depends on it.
func TestGPUMatMulRMSNorm(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(65))
	const rows, cols = 896, 300
	q, err := g.UploadQ8(quant.Quantize(tensai.RandomMatrix(rows, cols, rng)))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Free()
	x, err := g.Upload(randTensor(rng, 1, rows))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	norm, err := g.Upload(randTensor(rng, rows))
	if err != nil {
		t.Fatal(err)
	}
	defer norm.Free()
	bias, err := g.Upload(randTensor(rng, cols))
	if err != nil {
		t.Fatal(err)
	}
	defer bias.Free()
	const eps = 1e-6

	fused, err := q.MatMulRMSNorm(x, norm, eps, bias, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer fused.Free()
	nx, err := x.RMSNorm(norm, eps)
	if err != nil {
		t.Fatal(err)
	}
	defer nx.Free()
	split, err := q.MatMulOpts(nx, bias, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer split.Free()

	gt, err := fused.Download()
	if err != nil {
		t.Fatal(err)
	}
	wt, err := split.Download()
	if err != nil {
		t.Fatal(err)
	}
	for i := range wt.Data {
		if gt.Data[i] != wt.Data[i] {
			t.Fatalf("logit %d: fused %v != split %v", i, gt.Data[i], wt.Data[i])
		}
	}
}

// The fused attention combine must match the standalone attn_reduce_g
// dispatch followed by the plain projection bit for bit.
func TestGPUMatMulAttnCombine(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	if !g.HasF16() {
		t.Skip("device has no shader-f16")
	}
	rng := rand.New(rand.NewSource(67))
	const heads, kvHeads, dh, seqKV = 14, 2, 64, 400
	const d, kvDim = heads * dh, kvHeads * dh
	upload16 := func(src *tensai.Tensor) *Tensor {
		t.Helper()
		f32, err := g.Upload(src)
		if err != nil {
			t.Fatal(err)
		}
		defer f32.Free()
		c, err := g.NewF16Tensor(seqKV, kvDim)
		if err != nil {
			t.Fatal(err)
		}
		if err := f32.CopyRowsInto(c, 0); err != nil {
			t.Fatal(err)
		}
		return c
	}
	gk := upload16(randTensor(rng, seqKV, kvDim))
	defer gk.Free()
	gv := upload16(randTensor(rng, seqKV, kvDim))
	defer gv.Free()
	gq, err := g.Upload(randTensor(rng, 1, d))
	if err != nil {
		t.Fatal(err)
	}
	defer gq.Free()
	w, err := g.UploadQ8(quant.Quantize(tensai.RandomMatrix(d, 300, rng)))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Free()

	scr, slabs, err := gq.GroupedCausalAttentionParts(gk, gv, heads, kvHeads, seqKV, 0)
	if err != nil {
		t.Fatal(err)
	}
	if scr == nil {
		t.Fatal("split path not taken at seqKV 400")
	}
	defer scr.Free()
	fused, err := w.MatMulAttnCombine(scr, slabs, dh, heads/kvHeads, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer fused.Free()

	attn, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer attn.Free()
	split, err := w.MatMulOpts(attn, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer split.Free()

	gt, err := fused.Download()
	if err != nil {
		t.Fatal(err)
	}
	wt, err := split.Download()
	if err != nil {
		t.Fatal(err)
	}
	for i := range wt.Data {
		if gt.Data[i] != wt.Data[i] {
			t.Fatalf("out %d: fused %v != split %v", i, gt.Data[i], wt.Data[i])
		}
	}
}

// benchChainSplit is benchChain with the submission cut in half: the
// first half of the links flushes mid-chain, so the device starts while
// the host encodes the rest. The difference against the one-submission
// chain prices a mid-token flush.
func benchChainSplit(b *testing.B, links, inter int) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(64))
	up, err := g.UploadQ8(quant.Quantize(tensai.RandomMatrix(896, inter, rng)))
	if err != nil {
		b.Fatal(err)
	}
	defer up.Free()
	down, err := g.UploadQ8(quant.Quantize(tensai.RandomMatrix(inter, 896, rng)))
	if err != nil {
		b.Fatal(err)
	}
	defer down.Free()
	x, err := g.Upload(randTensor(rng, 1, 896))
	if err != nil {
		b.Fatal(err)
	}
	defer x.Free()
	run := func() {
		if err := g.BeginBatch(); err != nil {
			b.Fatal(err)
		}
		cur := x
		for i := 0; i < links; i++ {
			if i == links/2 {
				if err := g.Flush(); err != nil {
					b.Fatal(err)
				}
				if err := g.BeginBatch(); err != nil {
					b.Fatal(err)
				}
			}
			h, err := up.MatMul(cur)
			if err != nil {
				b.Fatal(err)
			}
			if cur != x {
				cur.Free()
			}
			o, err := down.MatMul(h)
			if err != nil {
				b.Fatal(err)
			}
			h.Free()
			cur = o
		}
		out, err := cur.DownloadRange(0, 896)
		if err != nil {
			b.Fatal(err)
		}
		_ = out
		if cur != x {
			cur.Free()
		}
	}
	run()
	b.SetBytes(int64(links) * 2 * 896 * int64(inter))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run()
	}
}

func BenchmarkGPUDecodeChain48Split(b *testing.B) { benchChainSplit(b, 24, 4864) }

// benchChain times a decode-shaped dispatch chain: single-row matvecs
// bouncing between hidden and intermediate width, all recorded into one
// submission the way step() records a token, flushed by one readback.
// Varying the number of links at the same total weight traffic separates
// the per-dispatch overhead from the streaming itself.
func benchChain(b *testing.B, links, inter int) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(64))
	up, err := g.UploadQ8(quant.Quantize(tensai.RandomMatrix(896, inter, rng)))
	if err != nil {
		b.Fatal(err)
	}
	defer up.Free()
	down, err := g.UploadQ8(quant.Quantize(tensai.RandomMatrix(inter, 896, rng)))
	if err != nil {
		b.Fatal(err)
	}
	defer down.Free()
	x, err := g.Upload(randTensor(rng, 1, 896))
	if err != nil {
		b.Fatal(err)
	}
	defer x.Free()
	run := func() {
		if err := g.BeginBatch(); err != nil {
			b.Fatal(err)
		}
		cur := x
		for i := 0; i < links; i++ {
			h, err := up.MatMul(cur)
			if err != nil {
				b.Fatal(err)
			}
			if cur != x {
				cur.Free()
			}
			o, err := down.MatMul(h)
			if err != nil {
				b.Fatal(err)
			}
			h.Free()
			cur = o
		}
		out, err := cur.DownloadRange(0, 896)
		if err != nil {
			b.Fatal(err)
		}
		_ = out
		if cur != x {
			cur.Free()
		}
	}
	run()                                             // warm the pipelines outside the timing
	b.SetBytes(int64(links) * 2 * 896 * int64(inter)) // int8 weight bytes per submission
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run()
	}
}

// 24 links x 4864 wide is one token's worth of gate+down traffic in 48
// dispatches; 6 links x 4x the width carries the same bytes in an eighth
// of the dispatches.
func BenchmarkGPUDecodeChain48(b *testing.B) { benchChain(b, 24, 4864) }
func BenchmarkGPUDecodeChain12(b *testing.B) { benchChain(b, 6, 4864*4) }

// 200 near-empty dependent dispatches: the per-dispatch latency floor a
// decode step pays on every norm, rope, and cache copy.
func BenchmarkGPUDecodeChainTiny(b *testing.B) { benchChain(b, 100, 64) }

// The qkv/o shape: ~1MB weights per dispatch. If small matvecs stream
// slower per byte than Chain48's, the kernel underfills the device on
// narrow outputs.
func BenchmarkGPUDecodeChainNarrow(b *testing.B) { benchChain(b, 50, 1152) }

// One decode step's attention chained 100 times at a 400-token context:
// per-link time is the per-layer attention cost inside a token.
func BenchmarkGPUDecodeAttn(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(66))
	const heads, kvHeads, headSz, steps = 14, 2, 64, 400
	kc, err := g.Upload(randTensor(rng, 512, kvHeads*headSz))
	if err != nil {
		b.Fatal(err)
	}
	defer kc.Free()
	vc, err := g.Upload(randTensor(rng, 512, kvHeads*headSz))
	if err != nil {
		b.Fatal(err)
	}
	defer vc.Free()
	x, err := g.Upload(randTensor(rng, 1, heads*headSz))
	if err != nil {
		b.Fatal(err)
	}
	defer x.Free()
	run := func() {
		if err := g.BeginBatch(); err != nil {
			b.Fatal(err)
		}
		cur := x
		for i := 0; i < 100; i++ {
			o, err := cur.GroupedCausalAttention(kc, vc, heads, kvHeads, steps, 0)
			if err != nil {
				b.Fatal(err)
			}
			if cur != x {
				cur.Free()
			}
			cur = o
		}
		if _, err := cur.DownloadRange(0, heads*headSz); err != nil {
			b.Fatal(err)
		}
		if cur != x {
			cur.Free()
		}
	}
	run()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run()
	}
}

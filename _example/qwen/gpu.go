package main

// GPU decode: every transformer block runs on the device — int8 weights
// resident via GPUQMatrix, the KV cache resident in two preallocated
// buffers per layer — so one token costs a handful of dispatches and a
// single download of the hidden state. The final norm, the lm_head matvec
// (whose weight would exceed modest per-buffer storage limits), and
// sampling stay on the CPU. The prompt still prefills through the batched
// CPU path; syncCache then copies the caches up once before decoding.

import (
	"fmt"

	tensai "github.com/mattn/tensai"
)

type gpuLayer struct {
	ln1, ln2                          *tensai.GPUTensor
	bq, bk, bv                        *tensai.GPUTensor
	qq, qk, qv, qo, qGate, qUp, qDown *tensai.GPUQMatrix
	kc, vc                            *tensai.GPUTensor // [nCtx, kvDim]
}

type gpuQwen struct {
	m      *qwen
	g      *tensai.GPU
	layers []gpuLayer
	nCtx   int
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// newGPUQwen uploads the model's transformer blocks to the GPU: the int8
// weight twins, the norm weights and biases, and zeroed KV caches sized
// for nCtx positions.
func newGPUQwen(m *qwen, g *tensai.GPU, nCtx int) (*gpuQwen, error) {
	kvDim := m.cfg.KVHeads * m.headSz
	vec := func(v []float32) *tensai.GPUTensor {
		return must(g.Upload(&tensai.Tensor{Shape: []int{len(v)}, Data: v}))
	}
	gq := &gpuQwen{m: m, g: g, nCtx: nCtx, layers: make([]gpuLayer, len(m.blocks))}
	for i := range m.blocks {
		b := &m.blocks[i]
		if b.qq == nil || b.qq.q8 == nil {
			return nil, fmt.Errorf("gpu decode needs int8 weights (run with -q8)")
		}
		l := &gq.layers[i]
		l.ln1, l.ln2 = vec(b.ln1), vec(b.ln2)
		l.bq, l.bk, l.bv = vec(b.bq), vec(b.bk), vec(b.bv)
		l.qq = must(g.UploadQ8(b.qq.q8))
		l.qk = must(g.UploadQ8(b.qk.q8))
		l.qv = must(g.UploadQ8(b.qv.q8))
		l.qo = must(g.UploadQ8(b.qo.q8))
		l.qGate = must(g.UploadQ8(b.qGate.q8))
		l.qUp = must(g.UploadQ8(b.qUp.q8))
		l.qDown = must(g.UploadQ8(b.qDown.q8))
		l.kc = must(g.Upload(tensai.NewTensor(nCtx, kvDim)))
		l.vc = must(g.Upload(tensai.NewTensor(nCtx, kvDim)))
	}
	return gq, nil
}

// syncCache copies the CPU-side KV cache — the prompt prefill — into the
// GPU caches, one upload per layer.
func (gq *gpuQwen) syncCache() error {
	kvDim := gq.m.cfg.KVHeads * gq.m.headSz
	for i := range gq.layers {
		cb := &gq.m.blocks[i]
		if len(cb.kc) > gq.nCtx {
			return fmt.Errorf("prefill of %d tokens exceeds gpu cache of %d", len(cb.kc), gq.nCtx)
		}
		for _, pair := range [2]struct {
			rows [][]float32
			dst  *tensai.GPUTensor
		}{{cb.kc, gq.layers[i].kc}, {cb.vc, gq.layers[i].vc}} {
			if len(pair.rows) == 0 {
				continue
			}
			flat := &tensai.Tensor{Shape: []int{len(pair.rows), kvDim}, Data: make([]float32, len(pair.rows)*kvDim)}
			for t, row := range pair.rows {
				copy(flat.Data[t*kvDim:], row)
			}
			tmp, err := gq.g.Upload(flat)
			if err != nil {
				return err
			}
			err = tmp.CopyRowsInto(pair.dst, 0)
			tmp.Free()
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// step feeds one token at position pos through the GPU-resident blocks and
// returns the next-token logits. Same signature as the CPU step, so the
// decode loop swaps between them freely.
func (gq *gpuQwen) step(token, pos int) []float32 {
	m := gq.m
	cfg := m.cfg
	hs := cfg.HiddenSize
	kvDim := cfg.KVHeads * m.headSz

	x := must(gq.g.Upload(&tensai.Tensor{Shape: []int{1, hs}, Data: m.embed.Data[token*hs : (token+1)*hs]}))
	defer x.Free()
	// One submission for the whole token: every dispatch below is recorded
	// into a single command buffer, and the Download at the end flushes it.
	if err := gq.g.BeginBatch(); err != nil {
		panic(err)
	}
	for i := range gq.layers {
		l := &gq.layers[i]
		a := must(x.RMSNorm(l.ln1, cfg.RMSEps))
		q := must(l.qq.MatMul(a))
		k := must(l.qk.MatMul(a))
		v := must(l.qv.MatMul(a))
		a.Free()
		if err := q.Add(l.bq); err != nil {
			panic(err)
		}
		if err := k.Add(l.bk); err != nil {
			panic(err)
		}
		if err := v.Add(l.bv); err != nil {
			panic(err)
		}
		if err := q.RoPE(m.headSz, pos, cfg.RopeTheta); err != nil {
			panic(err)
		}
		if err := k.RoPE(m.headSz, pos, cfg.RopeTheta); err != nil {
			panic(err)
		}
		if err := k.CopyRowsInto(l.kc, pos*kvDim); err != nil {
			panic(err)
		}
		if err := v.CopyRowsInto(l.vc, pos*kvDim); err != nil {
			panic(err)
		}
		k.Free()
		v.Free()
		attn := must(q.GroupedCausalAttention(l.kc, l.vc, cfg.Heads, cfg.KVHeads, pos+1))
		q.Free()
		proj := must(l.qo.MatMul(attn))
		attn.Free()
		if err := x.Add(proj); err != nil {
			panic(err)
		}
		proj.Free()

		a = must(x.RMSNorm(l.ln2, cfg.RMSEps))
		gate := must(l.qGate.MatMul(a))
		up := must(l.qUp.MatMul(a))
		a.Free()
		if err := gate.SiluMul(up); err != nil {
			panic(err)
		}
		up.Free()
		down := must(l.qDown.MatMul(gate))
		gate.Free()
		if err := x.Add(down); err != nil {
			panic(err)
		}
		down.Free()
	}
	xt := must(x.Download())
	return mv(rmsnorm(xt.Data, m.normW, cfg.RMSEps), m.lmT, m.qLmT, nil)
}

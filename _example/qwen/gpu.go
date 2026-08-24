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

// gpuMat is the resident weight interface both quantization widths
// satisfy.
type gpuMat interface {
	MatMul(*tensai.GPUTensor) (*tensai.GPUTensor, error)
	Free()
}

type gpuLayer struct {
	ln1, ln2                          *tensai.GPUTensor
	bq, bk, bv                        *tensai.GPUTensor
	qq, qk, qv, qo, qGate, qUp, qDown gpuMat
	kc, vc                            *tensai.GPUTensor // [nCtx, kvDim]
}

type gpuQwen struct {
	m      *qwen
	g      *tensai.GPU
	layers []gpuLayer
	nCtx   int
	gpuLen int // cache positions currently valid on the GPU
}

// sliceQ4 copies a column range out of a fused 4-bit quantized matrix;
// scales are per (group, column), so the slice is exact.
func sliceQ4(q *tensai.Q4Matrix, lo, hi int) *tensai.Q4Matrix {
	pairs := (q.Rows + 1) / 2
	groups := len(q.Scale) / q.Cols
	cols := hi - lo
	out := &tensai.Q4Matrix{
		Rows:  q.Rows,
		Cols:  cols,
		Q:     make([]uint8, pairs*cols+16),
		Scale: make([]float32, groups*cols),
	}
	for i2 := 0; i2 < pairs; i2++ {
		copy(out.Q[i2*cols:i2*cols+cols], q.Q[i2*q.Cols+lo:i2*q.Cols+hi])
	}
	for g := 0; g < groups; g++ {
		copy(out.Scale[g*cols:(g+1)*cols], q.Scale[g*q.Cols+lo:g*q.Cols+hi])
	}
	return out
}

// sliceQ copies a column range out of a fused quantized matrix: the GPU
// path keeps q, k, v (and gate, up) as separate resident weights, and
// per-column quantization makes the slice exact.
func sliceQ(q *tensai.QMatrix, lo, hi int) *tensai.QMatrix {
	pairs := (q.Rows + 1) / 2
	cols := hi - lo
	out := &tensai.QMatrix{
		Rows:     q.Rows,
		Cols:     cols,
		Q:        make([]int8, pairs*2*cols+16),
		Scale:    make([]float32, cols),
		ColSum64: make([]int32, cols+8),
	}
	copy(out.Scale, q.Scale[lo:hi])
	copy(out.ColSum64, q.ColSum64[lo:hi])
	for i2 := 0; i2 < pairs; i2++ {
		copy(out.Q[i2*2*cols:i2*2*cols+2*cols], q.Q[i2*2*q.Cols+2*lo:i2*2*q.Cols+2*hi])
	}
	return out
}

// vecRange slices a fused bias, staying nil for biasless models.
func vecRange(v []float32, lo, hi int) []float32 {
	if v == nil {
		return nil
	}
	return v[lo:hi]
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
		if v == nil { // llama: no attention biases
			return nil
		}
		return must(g.Upload(&tensai.Tensor{Shape: []int{len(v)}, Data: v}))
	}
	hs := m.cfg.HiddenSize
	inter := m.cfg.Intermediate
	gq := &gpuQwen{m: m, g: g, nCtx: nCtx, layers: make([]gpuLayer, len(m.blocks))}
	// upSlice uploads a column range of a fused weight in whichever width
	// it was quantized to; up uploads a whole one.
	upSlice := func(q *qmat, lo, hi int) gpuMat {
		if q.q8 != nil {
			return must(g.UploadQ8(sliceQ(q.q8, lo, hi)))
		}
		return must(g.UploadQ4(sliceQ4(q.q4, lo, hi)))
	}
	up := func(q *qmat) gpuMat {
		if q.q8 != nil {
			return must(g.UploadQ8(q.q8))
		}
		return must(g.UploadQ4(q.q4))
	}
	for i := range m.blocks {
		b := &m.blocks[i]
		if b.qQKV == nil || (b.qQKV.q8 == nil && b.qQKV.q4 == nil) {
			return nil, fmt.Errorf("gpu decode needs quantized weights (run with -q8 or -q4)")
		}
		l := &gq.layers[i]
		l.ln1, l.ln2 = vec(b.ln1), vec(b.ln2)
		l.bq = vec(vecRange(b.bQKV, 0, hs))
		l.bk = vec(vecRange(b.bQKV, hs, hs+kvDim))
		l.bv = vec(vecRange(b.bQKV, hs+kvDim, hs+2*kvDim))
		l.qq = upSlice(b.qQKV, 0, hs)
		l.qk = upSlice(b.qQKV, hs, hs+kvDim)
		l.qv = upSlice(b.qQKV, hs+kvDim, hs+2*kvDim)
		l.qo = up(b.qo)
		l.qGate = upSlice(b.qGU, 0, inter)
		l.qUp = upSlice(b.qGU, inter, 2*inter)
		l.qDown = up(b.qDown)
		l.kc = must(g.Upload(tensai.NewTensor(nCtx, kvDim)))
		l.vc = must(g.Upload(tensai.NewTensor(nCtx, kvDim)))
	}
	return gq, nil
}

// syncUp copies CPU-side cache rows the GPU has not seen yet — a fresh
// prompt prefill, or a chat turn's new tokens — into the GPU caches.
func (gq *gpuQwen) syncUp() error {
	kvDim := gq.m.cfg.KVHeads * gq.m.headSz
	cpuLen := len(gq.m.blocks[0].kc)
	if cpuLen > gq.nCtx {
		return fmt.Errorf("prefill of %d tokens exceeds gpu cache of %d", cpuLen, gq.nCtx)
	}
	lo := gq.gpuLen
	if cpuLen <= lo {
		return nil
	}
	for i := range gq.layers {
		cb := &gq.m.blocks[i]
		for _, pair := range [2]struct {
			rows [][]float32
			dst  *tensai.GPUTensor
		}{{cb.kc, gq.layers[i].kc}, {cb.vc, gq.layers[i].vc}} {
			flat := &tensai.Tensor{Shape: []int{cpuLen - lo, kvDim}, Data: make([]float32, (cpuLen-lo)*kvDim)}
			for t, row := range pair.rows[lo:] {
				copy(flat.Data[t*kvDim:], row)
			}
			tmp, err := gq.g.Upload(flat)
			if err != nil {
				return err
			}
			err = tmp.CopyRowsInto(pair.dst, lo*kvDim)
			tmp.Free()
			if err != nil {
				return err
			}
		}
	}
	gq.gpuLen = cpuLen
	return nil
}

// syncBack appends the rows GPU decoding produced since the CPU cache
// last saw them, so the next turn's CPU prefill attends over the whole
// conversation — the piece that lets -gpu and -chat compose.
func (gq *gpuQwen) syncBack() error {
	kvDim := gq.m.cfg.KVHeads * gq.m.headSz
	cpuLen := len(gq.m.blocks[0].kc)
	if gq.gpuLen <= cpuLen {
		return nil
	}
	n := gq.gpuLen - cpuLen
	for i := range gq.layers {
		cb := &gq.m.blocks[i]
		kt, err := gq.layers[i].kc.DownloadRange(cpuLen*kvDim, n*kvDim)
		if err != nil {
			return err
		}
		vt, err := gq.layers[i].vc.DownloadRange(cpuLen*kvDim, n*kvDim)
		if err != nil {
			return err
		}
		for t := 0; t < n; t++ {
			cb.kc = append(cb.kc, kt.Data[t*kvDim:(t+1)*kvDim])
			cb.vc = append(cb.vc, vt.Data[t*kvDim:(t+1)*kvDim])
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
		if l.bq != nil {
			if err := q.Add(l.bq); err != nil {
				panic(err)
			}
			if err := k.Add(l.bk); err != nil {
				panic(err)
			}
			if err := v.Add(l.bv); err != nil {
				panic(err)
			}
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
		if pos+1 > gq.gpuLen {
			gq.gpuLen = pos + 1
		}
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

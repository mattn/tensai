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
	qNorm, kNorm                      *tensai.GPUTensor // Qwen3/Gemma QK-norm; nil otherwise
	postAttn, postFFN                 *tensai.GPUTensor // Gemma sandwich norms; nil otherwise
	noPE                              bool
	window                            int
	ropeTheta                         float64
	geglu                             bool
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
// scales are per (group, column), so the slice is exact. Tile-aligned
// ranges (every fused projection boundary in practice) copy whole
// contiguous tile blocks; anything else moves nibble by nibble.
func sliceQ4(q *tensai.Q4Matrix, lo, hi int) *tensai.Q4Matrix {
	quads := (q.Rows + 3) / 4
	gsz := q.Group
	if gsz == 0 {
		gsz = 64
	}
	groups := (q.Rows + gsz - 1) / gsz
	cols := hi - lo
	out := tensai.NewQ4Matrix(q.Rows, cols, q.Group, false)
	if lo%32 == 0 && cols%32 == 0 {
		block := quads * 64
		copy(out.Q[:(cols/32)*block], q.Q[(lo/32)*block:(hi/32)*block])
	} else {
		for j := 0; j < cols; j++ {
			for i := 0; i < 4*quads; i += 2 {
				b := q.Q[q.Index(i, lo+j)]
				out.Q[out.Index(i, j)] |= b & 0x0F << (4 * uint(i%2))
				out.Q[out.Index(i+1, j)] |= b >> 4 << (4 * uint((i+1)%2))
			}
		}
	}
	for g := 0; g < groups; g++ {
		for j := 0; j < cols; j++ {
			out.Scale[out.TableIndex(g, j)] = q.Scale[q.TableIndex(g, lo+j)]
		}
	}
	return out
}

// sliceQ copies a column range out of a fused quantized matrix: the GPU
// path keeps q, k, v (and gate, up) as separate resident weights, and
// per-column quantization makes the slice exact.
func sliceQ(q *tensai.QMatrix, lo, hi int) *tensai.QMatrix {
	quads := (q.Rows + 3) / 4
	cols := hi - lo
	out := &tensai.QMatrix{
		Rows:     q.Rows,
		Cols:     cols,
		Q:        make([]int8, ((cols+31)/32)*quads*4*32+32),
		Scale:    make([]float32, cols),
		ColSum64: make([]int32, cols+8),
	}
	copy(out.Scale, q.Scale[lo:hi])
	copy(out.ColSum64, q.ColSum64[lo:hi])
	if lo%32 == 0 && cols%32 == 0 {
		block := quads * 4 * 32
		copy(out.Q[:(cols/32)*block], q.Q[(lo/32)*block:(hi/32)*block])
	} else {
		for j := 0; j < cols; j++ {
			for i := 0; i < 4*quads; i++ {
				out.Q[out.Index(i, j)] = q.Q[q.Index(i, lo+j)]
			}
		}
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
	hs := m.cfg.Heads * m.headSz // query dimension; equals hidden except qwen3
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
		l.noPE = b.noPE
		l.window = b.window
		l.ropeTheta = b.ropeTheta
		l.geglu = b.geglu
		l.ln1, l.ln2 = vec(b.ln1), vec(b.ln2)
		l.qNorm, l.kNorm = vec(b.qNorm), vec(b.kNorm)
		l.postAttn, l.postFFN = vec(b.postAttn), vec(b.postFFN)
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
	// Warm every kernel the decode and both prefill batch sizes touch:
	// drivers like dozen compile a pipeline's GPU code at first dispatch,
	// which otherwise lands in the first timed prefill. The dummy tokens
	// rewind afterwards, and the real prefill overwrites their cache rows.
	warm := make([]int, 33)
	gq.prefill(warm, 0)
	gq.prefill(warm[:4], 33)
	gq.step(0, 37)
	gq.gpuLen = 0
	return gq, nil
}

// prefill feeds a batch of tokens through the GPU-resident blocks — the
// batched twins of every decode op (row-batched RMSNorm and RoPE, the
// multi-row quantized matmuls, causal attention with the queries aligned
// to the cache end) — extending the resident KV cache and returning the
// next-token logits after the last position. With this, -gpu never
// touches the CPU KV cache at all.
func (gq *gpuQwen) prefill(tokens []int, startPos int) []float32 {
	// The widest intermediate is a gate/up projection row, so batches
	// whose activations would exceed the device's storage-buffer limit
	// split into chunks; the KV cache carries across them.
	chunk := len(tokens)
	if lim := gq.g.StorageLimit(); lim > 0 {
		w := gq.m.cfg.Intermediate
		if gq.m.cfg.MoeFF > w {
			w = gq.m.cfg.MoeFF
		}
		if c := int(lim / uint64(4*w)); c > 0 && c < chunk {
			chunk = c
		}
	}
	for len(tokens) > chunk {
		gq.prefillChunk(tokens[:chunk], startPos)
		tokens = tokens[chunk:]
		startPos += chunk
	}
	return gq.prefillChunk(tokens, startPos)
}

func (gq *gpuQwen) prefillChunk(tokens []int, startPos int) []float32 {
	m := gq.m
	cfg := m.cfg
	hs := cfg.HiddenSize
	kvDim := cfg.KVHeads * m.headSz
	n := len(tokens)
	if startPos+n > gq.nCtx {
		panic(fmt.Sprintf("prefill of %d tokens exceeds gpu cache of %d", startPos+n, gq.nCtx))
	}

	flat := &tensai.Tensor{Shape: []int{n, hs}, Data: make([]float32, n*hs)}
	for t, tk := range tokens {
		copy(flat.Data[t*hs:], m.embed.Data[tk*hs:(tk+1)*hs])
	}
	x := must(gq.g.Upload(flat))
	defer x.Free()
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
		if l.qNorm != nil {
			nq := must(q.RMSNormEach(l.qNorm, cfg.RMSEps))
			q.Free()
			q = nq
			nk := must(k.RMSNormEach(l.kNorm, cfg.RMSEps))
			k.Free()
			k = nk
		}
		if !l.noPE {
			theta := l.ropeTheta
			if theta == 0 {
				theta = cfg.RopeTheta
			}
			if err := q.RoPE(m.headSz, startPos, theta); err != nil {
				panic(err)
			}
			if err := k.RoPE(m.headSz, startPos, theta); err != nil {
				panic(err)
			}
		}
		if err := k.CopyRowsInto(l.kc, startPos*kvDim); err != nil {
			panic(err)
		}
		if err := v.CopyRowsInto(l.vc, startPos*kvDim); err != nil {
			panic(err)
		}
		k.Free()
		v.Free()
		attn := must(q.GroupedCausalAttention(l.kc, l.vc, cfg.Heads, cfg.KVHeads, startPos+n, l.window))
		q.Free()
		proj := must(l.qo.MatMul(attn))
		attn.Free()
		if l.postAttn != nil {
			np := must(proj.RMSNorm(l.postAttn, cfg.RMSEps))
			proj.Free()
			proj = np
		}
		if err := x.Add(proj); err != nil {
			panic(err)
		}
		proj.Free()

		a = must(x.RMSNorm(l.ln2, cfg.RMSEps))
		gate := must(l.qGate.MatMul(a))
		up := must(l.qUp.MatMul(a))
		a.Free()
		if l.geglu {
			if err := gate.GeluMul(up); err != nil {
				panic(err)
			}
		} else if err := gate.SiluMul(up); err != nil {
			panic(err)
		}
		up.Free()
		down := must(l.qDown.MatMul(gate))
		gate.Free()
		if l.postFFN != nil {
			nd := must(down.RMSNorm(l.postFFN, cfg.RMSEps))
			down.Free()
			down = nd
		}
		if err := x.Add(down); err != nil {
			panic(err)
		}
		down.Free()
	}
	gq.gpuLen = startPos + n
	// Only the final position's hidden state comes back; the download
	// flushes the batch.
	last := must(x.DownloadRange((n-1)*hs, hs))
	a := make([]float32, hs)
	rmsnormInto(a, last.Data, m.normW, cfg.RMSEps)
	return mv(a, m.lmT, m.qLmT, nil)
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
		if l.qNorm != nil {
			nq := must(q.RMSNormEach(l.qNorm, cfg.RMSEps))
			q.Free()
			q = nq
			nk := must(k.RMSNormEach(l.kNorm, cfg.RMSEps))
			k.Free()
			k = nk
		}
		if !l.noPE {
			theta := l.ropeTheta
			if theta == 0 {
				theta = cfg.RopeTheta
			}
			if err := q.RoPE(m.headSz, pos, theta); err != nil {
				panic(err)
			}
			if err := k.RoPE(m.headSz, pos, theta); err != nil {
				panic(err)
			}
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
		attn := must(q.GroupedCausalAttention(l.kc, l.vc, cfg.Heads, cfg.KVHeads, pos+1, l.window))
		q.Free()
		proj := must(l.qo.MatMul(attn))
		attn.Free()
		if l.postAttn != nil {
			np := must(proj.RMSNorm(l.postAttn, cfg.RMSEps))
			proj.Free()
			proj = np
		}
		if err := x.Add(proj); err != nil {
			panic(err)
		}
		proj.Free()

		a = must(x.RMSNorm(l.ln2, cfg.RMSEps))
		gate := must(l.qGate.MatMul(a))
		up := must(l.qUp.MatMul(a))
		a.Free()
		if l.geglu {
			if err := gate.GeluMul(up); err != nil {
				panic(err)
			}
		} else if err := gate.SiluMul(up); err != nil {
			panic(err)
		}
		up.Free()
		down := must(l.qDown.MatMul(gate))
		gate.Free()
		if l.postFFN != nil {
			nd := must(down.RMSNorm(l.postFFN, cfg.RMSEps))
			down.Free()
			down = nd
		}
		if err := x.Add(down); err != nil {
			panic(err)
		}
		down.Free()
	}
	xt := must(x.Download())
	a := make([]float32, hs)
	rmsnormInto(a, xt.Data, m.normW, cfg.RMSEps)
	return mv(a, m.lmT, m.qLmT, nil)
}

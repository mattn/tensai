package llm

// GPU decode: every transformer block runs on the device — int8 weights
// resident via GPUQMatrix, the KV cache resident in two preallocated
// buffers per layer — so one token costs a handful of dispatches and a
// single download. The final norm and the lm_head join them when the
// device's storage limit has room for the vocabulary projection, which
// leaves only sampling on the CPU; where it does not fit, the hidden
// state comes back and the CPU finishes the token. The prompt still
// prefills through the batched CPU path; syncCache then copies the caches
// up once before decoding.

import (
	"fmt"
	"io"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/quant"
)

// gpuMat is the resident weight interface both quantization widths
// satisfy.
type gpuMat interface {
	MatMul(*gpu.Tensor) (*gpu.Tensor, error)
	MatMulOpts(x, bias, dst *gpu.Tensor) (*gpu.Tensor, error)
	MatMulRMSNorm(x, norm *gpu.Tensor, eps float64, bias, dst *gpu.Tensor) (*gpu.Tensor, error)
	Free()
}

type gpuLayer struct {
	ln1, ln2                   *gpu.Tensor
	qNorm, kNorm               *gpu.Tensor // Qwen3/Gemma QK-norm; nil otherwise
	postAttn, postFFN          *gpu.Tensor // Gemma sandwich norms; nil otherwise
	noPE                       bool
	window                     int
	ropeTheta                  float64
	geglu                      bool
	bq, bk, bv                 *gpu.Tensor
	bQKV                       *gpu.Tensor // fused q|k|v bias, when the split aligns
	qQKV                       gpuMat      // fused q|k|v projection; nil when split
	qq, qk, qv, qo, qGU, qDown gpuMat
	kc, vc                     *gpu.Tensor // [nCtx, kvDim]
}

type gpuQwen struct {
	m      *qwen
	g      *gpu.Device
	layers []gpuLayer
	nCtx   int
	gpuLen int // cache positions currently valid on the GPU
	// The final norm and lm_head, resident when the device's storage
	// limit has room for the vocabulary projection — in one piece when it
	// fits, else as column slices with lmOff holding each slice's first
	// column and lmLogits collecting the slices for the one readback.
	// Empty on a device that cannot hold a slice, and decode falls back
	// to reading the hidden state back and finishing on the CPU.
	gNorm    *gpu.Tensor
	lmHead   []gpuMat
	lmOff    []int
	lmLogits *gpu.Tensor
}

// sliceQ4 copies a column range out of a fused 4-bit quantized matrix;
// scales are per (group, column), so the slice is exact. Tile-aligned
// ranges (every fused projection boundary in practice) copy whole
// contiguous tile blocks; anything else moves nibble by nibble.
func sliceQ4(q *quant.Q4Matrix, lo, hi int) *quant.Q4Matrix {
	quads := (q.Rows + 3) / 4
	gsz := q.Group
	if gsz == 0 {
		gsz = 64
	}
	groups := (q.Rows + gsz - 1) / gsz
	cols := hi - lo
	out := quant.NewQ4Matrix(q.Rows, cols, q.Group, false)
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
func sliceQ(q *quant.QMatrix, lo, hi int) *quant.QMatrix {
	quads := (q.Rows + 3) / 4
	cols := hi - lo
	out := &quant.QMatrix{
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

// tryUp uploads a whole quantized weight, reporting the error instead of
// panicking: the lm_head is the one weight big enough to be turned away by
// a device's storage limit, and that has to stay recoverable.
func tryUp(g *gpu.Device, q *qmat) (gpuMat, error) {
	if q.q8 != nil {
		return g.UploadQ8(q.q8)
	}
	return g.UploadQ4(q.q4)
}

// upLm uploads the lm_head, split into column slices when the whole
// weight overflows the device's storage limit — an untied Q8 head over a
// 150K vocabulary is a few MB past dozen's 128MiB, and one buffer too
// many used to send the entire projection back to the CPU every token.
func upLm(g *gpu.Device, q *qmat) ([]gpuMat, []int, error) {
	var rows, cols int
	var colBytes uint64
	if q.q8 != nil {
		rows, cols = q.q8.Rows, q.q8.Cols
		colBytes = uint64(4 * ((rows + 3) / 4)) // one uploaded column, as UploadQ8 lays it out
	} else {
		rows, cols = q.q4.Rows, q.q4.Cols
		colBytes = uint64(2 * ((rows + 3) / 4))
	}
	chunk := cols
	if limit := g.StorageLimit(); limit != 0 && colBytes*uint64(cols) > limit {
		// A megabyte of slack covers the scale table and padding; slice
		// boundaries stay on 64 columns so every part copies whole tiles.
		chunk = int((limit-1<<20)/colBytes) &^ 63
		if chunk <= 0 {
			return nil, nil, fmt.Errorf("device stores %d bytes, one lm_head column is %d", limit, colBytes)
		}
	}
	var parts []gpuMat
	var offs []int
	for lo := 0; lo < cols; lo += chunk {
		hi := min(lo+chunk, cols)
		var part gpuMat
		var err error
		switch {
		case lo == 0 && hi == cols:
			part, err = tryUp(g, q)
		case q.q8 != nil:
			part, err = g.UploadQ8(sliceQ(q.q8, lo, hi))
		default:
			part, err = g.UploadQ4(sliceQ4(q.q4, lo, hi))
		}
		if err != nil {
			for _, p := range parts {
				p.Free()
			}
			return nil, nil, err
		}
		parts = append(parts, part)
		offs = append(offs, lo)
	}
	return parts, offs, nil
}

// newGPUQwen uploads the model's transformer blocks to the GPU: the int8
// weight twins, the norm weights and biases, and zeroed KV caches sized
// for nCtx positions.
func newGPUQwen(m *qwen, g *gpu.Device, nCtx int, logw io.Writer) (*gpuQwen, error) {
	kvDim := m.cfg.KVHeads * m.headSz
	vec := func(v []float32) *gpu.Tensor {
		if v == nil { // llama: no attention biases
			return nil
		}
		return must(g.Upload(&tensai.Tensor{Shape: []int{len(v)}, Data: v}))
	}
	hs := m.cfg.Heads * m.headSz // query dimension; equals hidden except qwen3
	gq := &gpuQwen{m: m, g: g, nCtx: nCtx, layers: make([]gpuLayer, len(m.blocks))}
	// A half-precision KV cache halves the resident cache when the device
	// has shader-f16 and the model's head grouping fits the f16 kernel.
	group := m.cfg.Heads / m.cfg.KVHeads
	useF16 := g.HasF16() && group > 1 && group <= 8 && m.headSz <= 256
	// A view onto the fused result needs a 256-byte aligned offset, so
	// both the q and the q|k boundary have to sit on 64 floats.
	fuseQKV := hs%64 == 0 && kvDim%64 == 0
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
		// q, k, and v come out of one weight the loader already fused.
		// Splitting it costs two extra dispatches over matrices narrow
		// enough to leave the GPU idle -- measured on a Radeon 780M, the
		// three-way split runs the projection at 24.3 GB/s against 37.4
		// for the single fused dispatch. Keep it whole where the views
		// that carve q, k, and v back out of the result are legal, and
		// fall back to the split when they are not.
		if fuseQKV {
			l.qQKV = up(b.qQKV)
			l.bQKV = vec(b.bQKV)
		} else {
			l.bq = vec(vecRange(b.bQKV, 0, hs))
			l.bk = vec(vecRange(b.bQKV, hs, hs+kvDim))
			l.bv = vec(vecRange(b.bQKV, hs+kvDim, hs+2*kvDim))
			l.qq = upSlice(b.qQKV, 0, hs)
			l.qk = upSlice(b.qQKV, hs, hs+kvDim)
			l.qv = upSlice(b.qQKV, hs+kvDim, hs+2*kvDim)
		}
		l.qo = up(b.qo)
		// gate and up stay one fused weight: one dispatch projects both,
		// and glu_split joins the halves.
		l.qGU = up(b.qGU)
		l.qDown = up(b.qDown)
		if useF16 {
			l.kc = must(g.NewF16Tensor(nCtx, kvDim))
			l.vc = must(g.NewF16Tensor(nCtx, kvDim))
		} else {
			l.kc = must(g.Upload(tensai.NewTensor(nCtx, kvDim)))
			l.vc = must(g.Upload(tensai.NewTensor(nCtx, kvDim)))
		}
	}
	// The lm_head is the single biggest weight in a small model — for
	// Qwen3-0.6B its 1024x151936 is a quarter of everything decode
	// streams — so leaving it on the CPU caps what moving the blocks can
	// buy. Upload it when it fits, and keep the final norm next to it so
	// the whole token, logits included, stays one submission. Devices
	// with a smaller storage limit just miss out and keep the CPU path.
	if q := m.qLmT; q != nil {
		if parts, offs, err := upLm(g, q); err != nil {
			fmt.Fprintf(logw, "lm_head stays on the cpu: %v\n", err)
		} else {
			gq.lmHead, gq.lmOff = parts, offs
			gq.gNorm = vec(m.normW)
			if len(parts) > 1 {
				gq.lmLogits = must(g.Upload(tensai.NewTensor(1, m.cfg.Vocab)))
			}
		}
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

// qkv projects one row into q, k, and v. The fused weight does it in a
// single dispatch and hands back three views onto the result, so the
// caller frees the returned owner once and the views not at all; without
// it the three split weights each produce their own tensor and owner is
// nil. Only decode can take the fused path: a multi-row result interleaves
// q, k, and v per row, and no contiguous view covers that.
func (gq *gpuQwen) qkv(l *gpuLayer, x *gpu.Tensor, norm *gpu.Tensor, eps float64) (q, k, v, owner *gpu.Tensor) {
	if l.qQKV == nil {
		// Each projection re-derives the same inverse rms in its
		// prologue; that is still three dispatches against the four the
		// standalone norm made it.
		return must(l.qq.MatMulRMSNorm(x, norm, eps, l.bq, nil)),
			must(l.qk.MatMulRMSNorm(x, norm, eps, l.bk, nil)),
			must(l.qv.MatMulRMSNorm(x, norm, eps, l.bv, nil)), nil
	}
	hs := gq.m.cfg.Heads * gq.m.headSz
	kvDim := gq.m.cfg.KVHeads * gq.m.headSz
	f := must(l.qQKV.MatMulRMSNorm(x, norm, eps, l.bQKV, nil))
	return must(f.View(0, 1, hs)),
		must(f.View(hs, 1, kvDim)),
		must(f.View(hs+kvDim, 1, kvDim)), f
}

// qkvRows is qkv for a batch of token rows. The fused result is
// [rows, hs+2*kvDim] with q, k, and v side by side inside every row, so
// no contiguous view covers one of them and each is copied out instead.
// That copy is what lets the fused weight serve prefill as well, which
// is what keeps the split weights off the device entirely.
func (gq *gpuQwen) qkvRows(l *gpuLayer, a *gpu.Tensor) (q, k, v *gpu.Tensor) {
	if l.qQKV == nil {
		return must(l.qq.MatMulOpts(a, l.bq, nil)),
			must(l.qk.MatMulOpts(a, l.bk, nil)),
			must(l.qv.MatMulOpts(a, l.bv, nil))
	}
	hs := gq.m.cfg.Heads * gq.m.headSz
	kvDim := gq.m.cfg.KVHeads * gq.m.headSz
	f := must(l.qQKV.MatMulOpts(a, l.bQKV, nil))
	defer f.Free()
	return must(f.SliceCols(0, hs)),
		must(f.SliceCols(hs, kvDim)),
		must(f.SliceCols(hs+kvDim, kvDim))
}

// prefill feeds a batch of tokens through the GPU-resident blocks — the
// batched twins of every decode op (row-batched RMSNorm and RoPE, the
// multi-row quantized matmuls, causal attention with the queries aligned
// to the cache end) — extending the resident KV cache and returning the
// next-token logits after the last position. With this, -gpu never
// touches the CPU KV cache at all.
func (gq *gpuQwen) prefill(tokens []int, startPos int) []float32 {
	// Batches split into chunks, the KV cache carrying across them: the
	// widest intermediate (a gate/up projection row) must fit the
	// device's storage-buffer limit, and a chunk caps at 512 tokens so
	// no single submission runs long enough for the OS's GPU watchdog
	// (Windows TDR behind dozen) to declare the device lost.
	chunk := 512
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
		q, k, v := gq.qkvRows(l, a)
		a.Free()
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
		// The residual add rides the projection's epilogue unless a
		// sandwich norm (Gemma) has to run in between.
		if l.postAttn == nil {
			must(l.qo.MatMulOpts(attn, nil, x))
			attn.Free()
		} else {
			proj := must(l.qo.MatMul(attn))
			attn.Free()
			np := must(proj.RMSNorm(l.postAttn, cfg.RMSEps))
			proj.Free()
			proj = np
			if err := x.Add(proj); err != nil {
				panic(err)
			}
			proj.Free()
		}

		a = must(x.RMSNorm(l.ln2, cfg.RMSEps))
		gu := must(l.qGU.MatMul(a))
		a.Free()
		gate := must(gu.GLUSplit(cfg.Intermediate, l.geglu))
		gu.Free()
		if l.postFFN == nil {
			must(l.qDown.MatMulOpts(gate, nil, x))
			gate.Free()
		} else {
			down := must(l.qDown.MatMul(gate))
			gate.Free()
			nd := must(down.RMSNorm(l.postFFN, cfg.RMSEps))
			down.Free()
			down = nd
			if err := x.Add(down); err != nil {
				panic(err)
			}
			down.Free()
		}
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
		// A whole token is ~3ms of host-side encoding, serial with the
		// device under one submission. Cutting every eight layers lets
		// the device chew the early layers while the host encodes the
		// late ones; a submission boundary costs ~0.24ms against it.
		if i > 0 && i%6 == 0 {
			if err := gq.g.Flush(); err != nil {
				panic(err)
			}
			if err := gq.g.BeginBatch(); err != nil {
				panic(err)
			}
		}
		l := &gq.layers[i]
		q, k, v, qkvOwner := gq.qkv(l, x, l.ln1, cfg.RMSEps)
		if l.qNorm != nil {
			nq := must(q.RMSNormEach(l.qNorm, cfg.RMSEps))
			q.Free()
			q = nq
			nk := must(k.RMSNormEach(l.kNorm, cfg.RMSEps))
			k.Free()
			k = nk
		}
		theta := l.ropeTheta
		if theta == 0 {
			theta = cfg.RopeTheta
		}
		if !l.noPE && qkvOwner != nil && l.qNorm == nil && l.kc.IsF16() {
			// One dispatch rotates q and k on the fused row and appends
			// the rotated k and the v straight into the caches. Every
			// small dispatch in this chain costs ~40us of dependent-
			// dispatch latency on dozen, so the three-in-one matters
			// more than any of the work inside.
			if err := qkvOwner.RopeCacheF16(l.kc, l.vc, m.headSz, cfg.Heads*m.headSz, kvDim, pos, theta, pos*kvDim); err != nil {
				panic(err)
			}
			k.Free()
			v.Free()
		} else {
			if !l.noPE {
				if qkvOwner != nil && l.qNorm == nil {
					// q and k sit side by side in the fused row and the
					// heads are uniform, so one dispatch rotates both.
					qk := must(qkvOwner.View(0, 1, cfg.Heads*m.headSz+kvDim))
					if err := qk.RoPE(m.headSz, pos, theta); err != nil {
						panic(err)
					}
				} else {
					if err := q.RoPE(m.headSz, pos, theta); err != nil {
						panic(err)
					}
					if err := k.RoPE(m.headSz, pos, theta); err != nil {
						panic(err)
					}
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
		}
		if pos+1 > gq.gpuLen {
			gq.gpuLen = pos + 1
		}
		attn := must(q.GroupedCausalAttention(l.kc, l.vc, cfg.Heads, cfg.KVHeads, pos+1, l.window))
		q.Free()
		if qkvOwner != nil { // the buffer q, k, and v were views onto
			qkvOwner.Free()
		}
		// The residual add rides the projection's epilogue unless a
		// sandwich norm (Gemma) has to run in between.
		if l.postAttn == nil {
			must(l.qo.MatMulOpts(attn, nil, x))
			attn.Free()
		} else {
			proj := must(l.qo.MatMul(attn))
			attn.Free()
			np := must(proj.RMSNorm(l.postAttn, cfg.RMSEps))
			proj.Free()
			proj = np
			if err := x.Add(proj); err != nil {
				panic(err)
			}
			proj.Free()
		}

		gu := must(l.qGU.MatMulRMSNorm(x, l.ln2, cfg.RMSEps, nil, nil))
		gate := must(gu.GLUSplit(cfg.Intermediate, l.geglu))
		gu.Free()
		if l.postFFN == nil {
			must(l.qDown.MatMulOpts(gate, nil, x))
			gate.Free()
		} else {
			down := must(l.qDown.MatMul(gate))
			gate.Free()
			nd := must(down.RMSNorm(l.postFFN, cfg.RMSEps))
			down.Free()
			down = nd
			if err := x.Add(down); err != nil {
				panic(err)
			}
			down.Free()
		}
	}
	// With the lm_head resident the norm and the vocabulary projection
	// ride the same submission, and the one readback is the logits
	// themselves — the hidden state never crosses back. A sliced head
	// gathers its parts into one tensor first: a Download flushes and
	// waits, so two of them would cost a second round trip.
	if len(gq.lmHead) == 1 {
		logits := must(gq.lmHead[0].MatMulRMSNorm(x, gq.gNorm, cfg.RMSEps, nil, nil))
		defer logits.Free()
		return must(logits.Download()).Data
	}
	if len(gq.lmHead) > 1 {
		for i, part := range gq.lmHead {
			p := must(part.MatMulRMSNorm(x, gq.gNorm, cfg.RMSEps, nil, nil))
			if err := p.CopyRowsInto(gq.lmLogits, gq.lmOff[i]); err != nil {
				panic(err)
			}
			p.Free()
		}
		return must(gq.lmLogits.Download()).Data
	}
	xt := must(x.Download())
	a := make([]float32, hs)
	rmsnormInto(a, xt.Data, m.normW, cfg.RMSEps)
	return mv(a, m.lmT, m.qLmT, nil)
}

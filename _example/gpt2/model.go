package main

// GPT-2 small (124M) inference: 12 pre-norm transformer blocks with a KV
// cache, decoded one token at a time. The heavy lifting — every matvec
// against the checkpoint weights — runs on tensai's Dot kernel, so the
// same AVX2 path that trains the MNIST examples now drives a real
// published model.

import (
	"fmt"
	"math"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/quant"
)

const (
	nLayer = 12
	nHead  = 12
	nEmbd  = 768
	nCtx   = 1024
	headSz = nEmbd / nHead
	lnEps  = 1e-5
)

type block struct {
	ln1w, ln1b, ln2w, ln2b []float32
	attnW, projW           *tensai.Matrix // [768,2304], [768,768]
	fcW, fc2W              *tensai.Matrix // [768,3072], [3072,768]
	attnB, projB           []float32
	fcB, fc2B              []float32
	kc, vc                 [][]float32 // KV cache, one [768] per position

	// int8 twins of the four weight matrices, used by decode when -q8.
	qAttnW, qProjW, qFcW, qFc2W *quant.QMatrix
}

type gpt2 struct {
	wte, wpe   *tensai.Tensor // [50257,768], [1024,768]
	wteT       *tensai.Matrix // [768,50257], for the tied lm head
	qWteT      *quant.QMatrix
	lnfW, lnfB []float32
	blocks     [nLayer]block
	vocab      int
}

// quantize builds int8 twins of every decode-path weight. Prefill keeps
// the float32 originals: it is one batched pass where the matmuls are
// compute bound, while decode streams the whole checkpoint per token and
// is bandwidth bound — exactly where int8 pays.
func (m *gpt2) quantize() {
	m.qWteT = quant.Quantize(m.wteT)
	for i := range m.blocks {
		b := &m.blocks[i]
		b.qAttnW = quant.Quantize(b.attnW)
		b.qProjW = quant.Quantize(b.projW)
		b.qFcW = quant.Quantize(b.fcW)
		b.qFc2W = quant.Quantize(b.fc2W)
	}
}

// mv routes a decode matvec through the int8 weights when they exist.
func mv(x []float32, w *tensai.Matrix, q *quant.QMatrix, bias []float32) []float32 {
	if q == nil {
		return matvec(x, w, bias)
	}
	out := make([]float32, q.Cols)
	if err := q.MatVec(x, out); err != nil {
		panic(err)
	}
	if bias != nil {
		for i := range out {
			out[i] += bias[i]
		}
	}
	return out
}

func loadModel(path string) (*gpt2, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	vec := func(name string) []float32 {
		t, err := f.Tensor(name)
		if err != nil {
			panic(err)
		}
		return t.Data
	}
	mat := func(name string) *tensai.Matrix {
		t, err := f.Tensor(name)
		if err != nil {
			panic(err)
		}
		m, err := t.Matrix()
		if err != nil {
			panic(err)
		}
		return m
	}

	m := &gpt2{}
	m.wte, err = f.Tensor("wte.weight")
	if err != nil {
		return nil, err
	}
	m.wpe, err = f.Tensor("wpe.weight")
	if err != nil {
		return nil, err
	}
	m.vocab = m.wte.Shape[0]
	wteM, err := m.wte.Matrix()
	if err != nil {
		return nil, err
	}
	m.wteT = wteM.T() // one-time transpose for x @ wte^T
	m.lnfW, m.lnfB = vec("ln_f.weight"), vec("ln_f.bias")
	for i := range m.blocks {
		b := &m.blocks[i]
		p := fmt.Sprintf("h.%d.", i)
		b.ln1w, b.ln1b = vec(p+"ln_1.weight"), vec(p+"ln_1.bias")
		b.ln2w, b.ln2b = vec(p+"ln_2.weight"), vec(p+"ln_2.bias")
		b.attnW, b.attnB = mat(p+"attn.c_attn.weight"), vec(p+"attn.c_attn.bias")
		b.projW, b.projB = mat(p+"attn.c_proj.weight"), vec(p+"attn.c_proj.bias")
		b.fcW, b.fcB = mat(p+"mlp.c_fc.weight"), vec(p+"mlp.c_fc.bias")
		b.fc2W, b.fc2B = mat(p+"mlp.c_proj.weight"), vec(p+"mlp.c_proj.bias")
	}
	return m, nil
}

func layernorm(x, w, b []float32) []float32 {
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(len(x))
	inv := 1 / math.Sqrt(variance+lnEps)
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32((float64(v)-mean)*inv)*w[i] + b[i]
	}
	return out
}

// matvec computes x @ W (+ b when given) on tensai's Dot kernel.
func matvec(x []float32, w *tensai.Matrix, b []float32) []float32 {
	xm := &tensai.Matrix{Rows: 1, Cols: len(x), Data: x}
	out, err := tensai.Dot(xm, w)
	if err != nil {
		panic(err)
	}
	if b != nil {
		for i := range out.Data {
			out.Data[i] += b[i]
		}
	}
	return out.Data
}

// geluNew is the tanh approximation GPT-2 was trained with.
func geluNew(x []float32) {
	for i, v := range x {
		f := float64(v)
		x[i] = float32(0.5 * f * (1 + math.Tanh(0.7978845608028654*(f+0.044715*f*f*f))))
	}
}

// step feeds one token at position pos through the model and returns the
// logits for the next token.
func (m *gpt2) step(token, pos int) []float32 {
	x := make([]float32, nEmbd)
	for i := range x {
		x[i] = m.wte.Data[token*nEmbd+i] + m.wpe.Data[pos*nEmbd+i]
	}

	for li := range m.blocks {
		b := &m.blocks[li]

		// Attention with the KV cache: the single query row attends over
		// every cached position, so causality holds by construction.
		qkv := mv(layernorm(x, b.ln1w, b.ln1b), b.attnW, b.qAttnW, b.attnB)
		k := make([]float32, nEmbd)
		v := make([]float32, nEmbd)
		copy(k, qkv[nEmbd:2*nEmbd])
		copy(v, qkv[2*nEmbd:])
		b.kc = append(b.kc, k)
		b.vc = append(b.vc, v)
		q := qkv[:nEmbd]

		attn := make([]float32, nEmbd)
		steps := len(b.kc)
		scores := make([]float64, steps)
		for h := 0; h < nHead; h++ {
			off := h * headSz
			var maxs float64 = math.Inf(-1)
			for t := 0; t < steps; t++ {
				var s float64
				kt := b.kc[t]
				for i := 0; i < headSz; i++ {
					s += float64(q[off+i]) * float64(kt[off+i])
				}
				s /= 8 // sqrt(headSz)
				scores[t] = s
				if s > maxs {
					maxs = s
				}
			}
			var sum float64
			for t := 0; t < steps; t++ {
				scores[t] = math.Exp(scores[t] - maxs)
				sum += scores[t]
			}
			for t := 0; t < steps; t++ {
				p := float32(scores[t] / sum)
				vt := b.vc[t]
				for i := 0; i < headSz; i++ {
					attn[off+i] += p * vt[off+i]
				}
			}
		}
		proj := mv(attn, b.projW, b.qProjW, b.projB)
		for i := range x {
			x[i] += proj[i]
		}

		// MLP.
		h := mv(layernorm(x, b.ln2w, b.ln2b), b.fcW, b.qFcW, b.fcB)
		geluNew(h)
		out := mv(h, b.fc2W, b.qFc2W, b.fc2B)
		for i := range x {
			x[i] += out[i]
		}
	}

	return mv(layernorm(x, m.lnfW, m.lnfB), m.wteT, m.qWteT, nil)
}

// reset clears the KV cache for a fresh sequence.
func (m *gpt2) reset() {
	for i := range m.blocks {
		m.blocks[i].kc = nil
		m.blocks[i].vc = nil
	}
}

func layernormRows(x *tensai.Matrix, w, b []float32) *tensai.Matrix {
	out := tensai.NewMatrix(x.Rows, x.Cols)
	for r := 0; r < x.Rows; r++ {
		copy(out.Data[r*x.Cols:(r+1)*x.Cols], layernorm(x.Data[r*x.Cols:(r+1)*x.Cols], w, b))
	}
	return out
}

func addBiasRows(x *tensai.Matrix, b []float32) {
	for r := 0; r < x.Rows; r++ {
		row := x.Data[r*x.Cols : (r+1)*x.Cols]
		for i := range row {
			row[i] += b[i]
		}
	}
}

func addInPlace(x, y *tensai.Matrix) {
	for i := range x.Data {
		x.Data[i] += y.Data[i]
	}
}

func matmul(a, w *tensai.Matrix, bias []float32) *tensai.Matrix {
	out, err := tensai.Dot(a, w)
	if err != nil {
		panic(err)
	}
	if bias != nil {
		addBiasRows(out, bias)
	}
	return out
}

// cpuCausalMHA is multi-head causal attention over full (T, 768) q/k/v
// matrices, the CPU half of the prefill.
func cpuCausalMHA(q, k, v *tensai.Matrix) *tensai.Matrix {
	T := q.Rows
	out := tensai.NewMatrix(T, nEmbd)
	scores := make([]float64, T)
	for h := 0; h < nHead; h++ {
		off := h * headSz
		for i := 0; i < T; i++ {
			maxs := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var s float64
				for c := 0; c < headSz; c++ {
					s += float64(q.Data[i*nEmbd+off+c]) * float64(k.Data[j*nEmbd+off+c])
				}
				s /= 8
				scores[j] = s
				if s > maxs {
					maxs = s
				}
			}
			var sum float64
			for j := 0; j <= i; j++ {
				scores[j] = math.Exp(scores[j] - maxs)
				sum += scores[j]
			}
			for j := 0; j <= i; j++ {
				p := float32(scores[j] / sum)
				for c := 0; c < headSz; c++ {
					out.Data[i*nEmbd+off+c] += p * v.Data[j*nEmbd+off+c]
				}
			}
		}
	}
	return out
}

// gpuCausalMHA runs the same attention as one masked multi-head dispatch
// on resident tensors.
func gpuCausalMHA(g *gpu.Device, q, k, v *tensai.Matrix) *tensai.Matrix {
	upload := func(m *tensai.Matrix) *gpu.Tensor {
		t, err := g.Upload(m.Tensor())
		if err != nil {
			panic(err)
		}
		return t
	}
	gq, gk, gv := upload(q), upload(k), upload(v)
	defer gq.Free()
	defer gk.Free()
	defer gv.Free()
	got, err := gq.CausalMultiHeadAttention(gk, gv, nHead)
	if err != nil {
		panic(err)
	}
	defer got.Free()
	t, err := got.Download()
	if err != nil {
		panic(err)
	}
	m, err := t.Matrix()
	if err != nil {
		panic(err)
	}
	return m
}

// prefill runs the whole prompt through the model in one batched pass,
// filling the KV cache and returning the logits after the last token. The
// matmuls run on tensai's Dot kernel; with a non-nil GPU the causal
// attention of every block runs as one masked multi-head dispatch on the
// GPU instead of the CPU loops.
func (m *gpt2) prefill(tokens []int, g *gpu.Device) []float32 {
	T := len(tokens)
	x := tensai.NewMatrix(T, nEmbd)
	for t, tok := range tokens {
		row := x.Data[t*nEmbd : (t+1)*nEmbd]
		for i := range row {
			row[i] = m.wte.Data[tok*nEmbd+i] + m.wpe.Data[t*nEmbd+i]
		}
	}

	for li := range m.blocks {
		b := &m.blocks[li]

		qkv := matmul(layernormRows(x, b.ln1w, b.ln1b), b.attnW, b.attnB)
		q := tensai.NewMatrix(T, nEmbd)
		k := tensai.NewMatrix(T, nEmbd)
		v := tensai.NewMatrix(T, nEmbd)
		for t := 0; t < T; t++ {
			copy(q.Data[t*nEmbd:(t+1)*nEmbd], qkv.Data[t*3*nEmbd:t*3*nEmbd+nEmbd])
			copy(k.Data[t*nEmbd:(t+1)*nEmbd], qkv.Data[t*3*nEmbd+nEmbd:t*3*nEmbd+2*nEmbd])
			copy(v.Data[t*nEmbd:(t+1)*nEmbd], qkv.Data[t*3*nEmbd+2*nEmbd:(t+1)*3*nEmbd])
			b.kc = append(b.kc, k.Data[t*nEmbd:(t+1)*nEmbd])
			b.vc = append(b.vc, v.Data[t*nEmbd:(t+1)*nEmbd])
		}

		var attn *tensai.Matrix
		if g != nil {
			attn = gpuCausalMHA(g, q, k, v)
		} else {
			attn = cpuCausalMHA(q, k, v)
		}
		addInPlace(x, matmul(attn, b.projW, b.projB))

		h := matmul(layernormRows(x, b.ln2w, b.ln2b), b.fcW, b.fcB)
		geluNew(h.Data)
		addInPlace(x, matmul(h, b.fc2W, b.fc2B))
	}

	last := x.Data[(T-1)*nEmbd:]
	return matvec(layernorm(last, m.lnfW, m.lnfB), m.wteT, nil)
}

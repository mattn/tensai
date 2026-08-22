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
}

type gpt2 struct {
	wte, wpe   *tensai.Tensor // [50257,768], [1024,768]
	wteT       *tensai.Matrix // [768,50257], for the tied lm head
	lnfW, lnfB []float32
	blocks     [nLayer]block
	vocab      int
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
		qkv := matvec(layernorm(x, b.ln1w, b.ln1b), b.attnW, b.attnB)
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
		proj := matvec(attn, b.projW, b.projB)
		for i := range x {
			x[i] += proj[i]
		}

		// MLP.
		h := matvec(layernorm(x, b.ln2w, b.ln2b), b.fcW, b.fcB)
		geluNew(h)
		out := matvec(h, b.fc2W, b.fc2B)
		for i := range x {
			x[i] += out[i]
		}
	}

	return matvec(layernorm(x, m.lnfW, m.lnfB), m.wteT, nil)
}

// reset clears the KV cache for a fresh sequence.
func (m *gpt2) reset() {
	for i := range m.blocks {
		m.blocks[i].kc = nil
		m.blocks[i].vc = nil
	}
}

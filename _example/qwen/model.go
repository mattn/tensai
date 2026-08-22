package main

// Qwen2-family inference: pre-norm transformer blocks with RMSNorm, rotary
// position embeddings, grouped-query attention, and a SwiGLU MLP, decoded
// one token at a time with a KV cache. Dimensions come from config.json,
// so any Qwen2 checkpoint that fits in memory works; every matvec runs on
// tensai's Dot kernel or, with -q8, the int8 kernel.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
)

type config struct {
	HiddenSize   int     `json:"hidden_size"`
	Intermediate int     `json:"intermediate_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	KVHeads      int     `json:"num_key_value_heads"`
	RMSEps       float64 `json:"rms_norm_eps"`
	RopeTheta    float64 `json:"rope_theta"`
	Vocab        int     `json:"vocab_size"`
	TieEmbedding bool    `json:"tie_word_embeddings"`
	EOS          int     `json:"eos_token_id"`
	ModelType    string  `json:"model_type"`
}

type qblock struct {
	ln1, ln2          []float32
	wq, wk, wv, wo    *tensai.Matrix // [in, out] after transposing HF's [out, in]
	bq, bk, bv        []float32
	wGate, wUp, wDown *tensai.Matrix
	qq, qk, qv, qo    *tensai.QMatrix
	qGate, qUp, qDown *tensai.QMatrix
	kc, vc            [][]float32 // KV cache, kvHeads*headDim per position
}

type qwen struct {
	cfg    config
	headSz int
	embed  *tensai.Tensor // [vocab, hidden]
	lmT    *tensai.Matrix // [hidden, vocab]
	qLmT   *tensai.QMatrix
	normW  []float32
	blocks []qblock
}

func loadConfig(path string) (config, error) {
	var c config
	raw, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, err
	}
	if c.ModelType != "qwen2" {
		return c, fmt.Errorf("unsupported model_type %q (this example speaks qwen2)", c.ModelType)
	}
	return c, nil
}

func loadQwen(cfgPath, weightsPath string) (*qwen, error) {
	cfg, err := loadConfig(cfgPath)
	if err != nil {
		return nil, err
	}
	f, err := safetensors.Open(weightsPath)
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
	// HF Linear weights are [out, in]; transpose once so matvec sees
	// [in, out].
	lin := func(name string) *tensai.Matrix {
		t, err := f.Tensor(name)
		if err != nil {
			panic(err)
		}
		m, err := t.Matrix()
		if err != nil {
			panic(err)
		}
		return m.T()
	}

	m := &qwen{cfg: cfg, headSz: cfg.HiddenSize / cfg.Heads}
	m.embed, err = f.Tensor("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	em, err := m.embed.Matrix()
	if err != nil {
		return nil, err
	}
	if cfg.TieEmbedding {
		m.lmT = em.T()
	} else {
		m.lmT = lin("lm_head.weight")
	}
	m.normW = vec("model.norm.weight")
	m.blocks = make([]qblock, cfg.Layers)
	for i := range m.blocks {
		b := &m.blocks[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		b.ln1 = vec(p + "input_layernorm.weight")
		b.ln2 = vec(p + "post_attention_layernorm.weight")
		b.wq, b.bq = lin(p+"self_attn.q_proj.weight"), vec(p+"self_attn.q_proj.bias")
		b.wk, b.bk = lin(p+"self_attn.k_proj.weight"), vec(p+"self_attn.k_proj.bias")
		b.wv, b.bv = lin(p+"self_attn.v_proj.weight"), vec(p+"self_attn.v_proj.bias")
		b.wo = lin(p + "self_attn.o_proj.weight")
		b.wGate = lin(p + "mlp.gate_proj.weight")
		b.wUp = lin(p + "mlp.up_proj.weight")
		b.wDown = lin(p + "mlp.down_proj.weight")
	}
	return m, nil
}

// quantize builds int8 twins of every matvec weight.
func (m *qwen) quantize() {
	m.qLmT = tensai.QuantizeMatrix(m.lmT)
	for i := range m.blocks {
		b := &m.blocks[i]
		b.qq = tensai.QuantizeMatrix(b.wq)
		b.qk = tensai.QuantizeMatrix(b.wk)
		b.qv = tensai.QuantizeMatrix(b.wv)
		b.qo = tensai.QuantizeMatrix(b.wo)
		b.qGate = tensai.QuantizeMatrix(b.wGate)
		b.qUp = tensai.QuantizeMatrix(b.wUp)
		b.qDown = tensai.QuantizeMatrix(b.wDown)
	}
}

func rmsnorm(x, w []float32, eps float64) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v)*inv) * w[i]
	}
	return out
}

// mv computes x @ W (+ bias), on the int8 twin when it exists.
func mv(x []float32, w *tensai.Matrix, q *tensai.QMatrix, bias []float32) []float32 {
	var out []float32
	if q != nil {
		out = make([]float32, q.Cols)
		if err := q.MatVec(x, out); err != nil {
			panic(err)
		}
	} else {
		xm := &tensai.Matrix{Rows: 1, Cols: len(x), Data: x}
		o, err := tensai.Dot(xm, w)
		if err != nil {
			panic(err)
		}
		out = o.Data
	}
	if bias != nil {
		for i := range out {
			out[i] += bias[i]
		}
	}
	return out
}

// rope rotates one head in place, half-split style: pair (i, i+dh/2).
func (m *qwen) rope(h []float32, pos int) {
	half := m.headSz / 2
	for i := 0; i < half; i++ {
		freq := math.Pow(m.cfg.RopeTheta, -2*float64(i)/float64(m.headSz))
		s, c := math.Sincos(float64(pos) * freq)
		a, b := float64(h[i]), float64(h[i+half])
		h[i] = float32(a*c - b*s)
		h[i+half] = float32(b*c + a*s)
	}
}

// step feeds one token at position pos and returns the next-token logits.
func (m *qwen) step(token, pos int) []float32 {
	cfg := m.cfg
	hs := cfg.HiddenSize
	group := cfg.Heads / cfg.KVHeads

	x := make([]float32, hs)
	copy(x, m.embed.Data[token*hs:(token+1)*hs])

	for li := range m.blocks {
		b := &m.blocks[li]
		a := rmsnorm(x, b.ln1, cfg.RMSEps)
		q := mv(a, b.wq, b.qq, b.bq)
		k := mv(a, b.wk, b.qk, b.bk)
		v := mv(a, b.wv, b.qv, b.bv)
		for h := 0; h < cfg.Heads; h++ {
			m.rope(q[h*m.headSz:(h+1)*m.headSz], pos)
		}
		for h := 0; h < cfg.KVHeads; h++ {
			m.rope(k[h*m.headSz:(h+1)*m.headSz], pos)
		}
		b.kc = append(b.kc, k)
		b.vc = append(b.vc, v)

		attn := make([]float32, hs)
		steps := len(b.kc)
		scores := make([]float64, steps)
		scale := 1 / math.Sqrt(float64(m.headSz))
		for h := 0; h < cfg.Heads; h++ {
			qOff := h * m.headSz
			kvOff := (h / group) * m.headSz
			maxs := math.Inf(-1)
			for t := 0; t < steps; t++ {
				var s float64
				kt := b.kc[t]
				for i := 0; i < m.headSz; i++ {
					s += float64(q[qOff+i]) * float64(kt[kvOff+i])
				}
				s *= scale
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
				for i := 0; i < m.headSz; i++ {
					attn[qOff+i] += p * vt[kvOff+i]
				}
			}
		}
		proj := mv(attn, b.wo, b.qo, nil)
		for i := range x {
			x[i] += proj[i]
		}

		a = rmsnorm(x, b.ln2, cfg.RMSEps)
		gate := mv(a, b.wGate, b.qGate, nil)
		up := mv(a, b.wUp, b.qUp, nil)
		for i := range gate {
			g := float64(gate[i])
			gate[i] = float32(g/(1+math.Exp(-g))) * up[i] // silu(gate) * up
		}
		down := mv(gate, b.wDown, b.qDown, nil)
		for i := range x {
			x[i] += down[i]
		}
	}

	return mv(rmsnorm(x, m.normW, cfg.RMSEps), m.lmT, m.qLmT, nil)
}

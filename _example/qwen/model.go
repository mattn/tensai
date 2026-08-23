package main

// Qwen2- and Llama-family inference: pre-norm transformer blocks with
// RMSNorm, rotary position embeddings, grouped-query attention, and a
// SwiGLU MLP, decoded one token at a time with a KV cache. The two
// architectures share this exact block — Llama simply has no attention
// biases — so dimensions come from config.json and any qwen2 or llama
// checkpoint whose tokenizer.json is byte-level BPE works (SmolLM2, for
// example); every matvec runs on tensai's Dot kernel or, with -q8, the
// int8 kernel.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"

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
	MaxPos       int     `json:"max_position_embeddings"`
	Vocab        int     `json:"vocab_size"`
	TieEmbedding bool    `json:"tie_word_embeddings"`
	EOS          int     `json:"eos_token_id"`
	ModelType    string  `json:"model_type"`
}

// qmat abstracts the int8 and int4 twins behind one matvec call, plus the
// batched form prefill uses.
type qmat struct {
	cols int
	f    func(x, out []float32) error
	mm   func(x, out *tensai.Matrix) error
	q8   *tensai.QMatrix // retained for GPU upload; nil for int4
}

func quantizeMat(m *tensai.Matrix, bits int) *qmat {
	switch bits {
	case 8:
		q := tensai.QuantizeMatrix(m)
		return &qmat{cols: q.Cols, f: q.MatVec, mm: q.MatMul, q8: q}
	case 4:
		q, err := tensai.QuantizeMatrix4(m)
		if err != nil {
			panic(err)
		}
		return &qmat{cols: q.Cols, f: q.MatVec, mm: q.MatMul}
	}
	panic("unsupported quantization width")
}

type qblock struct {
	ln1, ln2          []float32
	wq, wk, wv, wo    *tensai.Matrix // [in, out] after transposing HF's [out, in]
	bq, bk, bv        []float32
	wGate, wUp, wDown *tensai.Matrix
	qq, qk, qv, qo    *qmat
	qGate, qUp, qDown *qmat
	kc, vc            [][]float32 // KV cache, kvHeads*headDim per position
}

type qwen struct {
	cfg    config
	headSz int
	embed  *tensai.Tensor // [vocab, hidden]
	lmT    *tensai.Matrix // [hidden, vocab]
	qLmT   *qmat
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
	if c.ModelType != "qwen2" && c.ModelType != "llama" {
		return c, fmt.Errorf("unsupported model_type %q (this example speaks qwen2 and llama)", c.ModelType)
	}
	return c, nil
}

// weightsFile is the part of safetensors.File and safetensors.Shards the
// loader needs.
type weightsFile interface {
	Tensor(string) (*tensai.Tensor, error)
	Close() error
}

// loadQwen reads a checkpoint — a single model.safetensors or a sharded
// one via its index.json — quantizing each weight to `bits` (0 keeps
// float32) as it loads, so the full float32 model never has to fit in
// memory at once.
func loadQwen(cfgPath, weightsPath string, bits int) (*qwen, error) {
	cfg, err := loadConfig(cfgPath)
	if err != nil {
		return nil, err
	}
	var f weightsFile
	if strings.HasSuffix(weightsPath, ".index.json") {
		f, err = safetensors.OpenSharded(weightsPath)
	} else {
		f, err = safetensors.Open(weightsPath)
	}
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
	// vecOpt returns nil for weights the architecture does not have —
	// Llama-family checkpoints carry no attention biases.
	vecOpt := func(name string) []float32 {
		t, err := f.Tensor(name)
		if err != nil {
			return nil
		}
		return t.Data
	}
	// HF Linear weights are [out, in]; transpose once so matvec sees
	// [in, out], then quantize immediately so the float32 copy dies here.
	linq := func(name string) (*tensai.Matrix, *qmat) {
		t, err := f.Tensor(name)
		if err != nil {
			panic(err)
		}
		m, err := t.Matrix()
		if err != nil {
			panic(err)
		}
		w := m.T()
		if bits == 0 {
			return w, nil
		}
		return nil, quantizeMat(w, bits)
	}

	m := &qwen{cfg: cfg, headSz: cfg.HiddenSize / cfg.Heads}
	m.embed, err = f.Tensor("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if cfg.TieEmbedding {
		em, err := m.embed.Matrix()
		if err != nil {
			return nil, err
		}
		lmT := em.T()
		if bits == 0 {
			m.lmT = lmT
		} else {
			m.qLmT = quantizeMat(lmT, bits)
		}
	} else {
		m.lmT, m.qLmT = linq("lm_head.weight")
	}
	m.normW = vec("model.norm.weight")
	m.blocks = make([]qblock, cfg.Layers)
	for i := range m.blocks {
		b := &m.blocks[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		b.ln1 = vec(p + "input_layernorm.weight")
		b.ln2 = vec(p + "post_attention_layernorm.weight")
		b.wq, b.qq = linq(p + "self_attn.q_proj.weight")
		b.bq = vecOpt(p + "self_attn.q_proj.bias")
		b.wk, b.qk = linq(p + "self_attn.k_proj.weight")
		b.bk = vecOpt(p + "self_attn.k_proj.bias")
		b.wv, b.qv = linq(p + "self_attn.v_proj.weight")
		b.bv = vecOpt(p + "self_attn.v_proj.bias")
		b.wo, b.qo = linq(p + "self_attn.o_proj.weight")
		b.wGate, b.qGate = linq(p + "mlp.gate_proj.weight")
		b.wUp, b.qUp = linq(p + "mlp.up_proj.weight")
		b.wDown, b.qDown = linq(p + "mlp.down_proj.weight")
	}
	return m, nil
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

// mv computes x @ W (+ bias), on the quantized twin when it exists.
func mv(x []float32, w *tensai.Matrix, q *qmat, bias []float32) []float32 {
	var out []float32
	if q != nil {
		out = make([]float32, q.cols)
		if err := q.f(x, out); err != nil {
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

// mmb is mv for a batch of rows: x @ W (+ bias per row), on the quantized
// twin when it exists.
func mmb(x, w *tensai.Matrix, q *qmat, bias []float32) *tensai.Matrix {
	var out *tensai.Matrix
	if q != nil {
		out = tensai.NewMatrix(x.Rows, q.cols)
		if err := q.mm(x, out); err != nil {
			panic(err)
		}
	} else {
		o, err := tensai.Dot(x, w)
		if err != nil {
			panic(err)
		}
		out = o
	}
	if bias != nil {
		for r := 0; r < out.Rows; r++ {
			row := out.Data[r*out.Cols : (r+1)*out.Cols]
			for i := range bias {
				row[i] += bias[i]
			}
		}
	}
	return out
}

// attendHead runs one head of cached-KV attention: SIMD dot-product
// scores over the cache, a float64 softmax, and SIMD weighted value
// accumulation into attn's slot for the head. Heads touch disjoint
// output ranges, so callers fan heads out across goroutines freely.
func (m *qwen) attendHead(b *qblock, q, attn []float32, h, group, steps int, scores []float64) {
	qOff := h * m.headSz
	kvOff := (h / group) * m.headSz
	scale := 1 / math.Sqrt(float64(m.headSz))
	qh := q[qOff : qOff+m.headSz]
	maxs := math.Inf(-1)
	for t := 0; t < steps; t++ {
		s := float64(tensai.DotVec(qh, b.kc[t][kvOff:kvOff+m.headSz])) * scale
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
	out := attn[qOff : qOff+m.headSz]
	for t := 0; t < steps; t++ {
		tensai.Axpy(float32(scores[t]/sum), b.vc[t][kvOff:kvOff+m.headSz], out)
	}
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

// prefill feeds a batch of tokens starting at startPos, extending the KV
// cache, and returns the next-token logits after the last one. It computes
// exactly what feeding the tokens through step one by one would — the
// batched quantized matmul quantizes each activation row identically and
// the attention loops match — but streams each weight matrix once per
// batch instead of once per token.
func (m *qwen) prefill(tokens []int, startPos int) []float32 {
	cfg := m.cfg
	hs := cfg.HiddenSize
	group := cfg.Heads / cfg.KVHeads
	kvDim := cfg.KVHeads * m.headSz
	n := len(tokens)

	x := tensai.NewMatrix(n, hs)
	for t, tk := range tokens {
		copy(x.Data[t*hs:(t+1)*hs], m.embed.Data[tk*hs:(tk+1)*hs])
	}
	a := tensai.NewMatrix(n, hs)
	norm := func(w []float32) {
		for t := 0; t < n; t++ {
			copy(a.Data[t*hs:(t+1)*hs], rmsnorm(x.Data[t*hs:(t+1)*hs], w, cfg.RMSEps))
		}
	}
	for li := range m.blocks {
		b := &m.blocks[li]
		norm(b.ln1)
		q := mmb(a, b.wq, b.qq, b.bq)
		k := mmb(a, b.wk, b.qk, b.bk)
		v := mmb(a, b.wv, b.qv, b.bv)
		for t := 0; t < n; t++ {
			pos := startPos + t
			qr := q.Data[t*hs:]
			for h := 0; h < cfg.Heads; h++ {
				m.rope(qr[h*m.headSz:(h+1)*m.headSz], pos)
			}
			kr := k.Data[t*kvDim : (t+1)*kvDim]
			for h := 0; h < cfg.KVHeads; h++ {
				m.rope(kr[h*m.headSz:(h+1)*m.headSz], pos)
			}
			b.kc = append(b.kc, kr)
			b.vc = append(b.vc, v.Data[t*kvDim:(t+1)*kvDim])
		}

		// Causal attention: row t sees cache positions [0, startPos+t].
		// Rows are independent, so they fan out across CPUs.
		attn := tensai.NewMatrix(n, hs)
		var wg sync.WaitGroup
		rowCh := make(chan int, n)
		for t := 0; t < n; t++ {
			rowCh <- t
		}
		close(rowCh)
		for w := min(runtime.NumCPU(), n); w > 0; w-- {
			wg.Add(1)
			go func() {
				defer wg.Done()
				scores := make([]float64, startPos+n)
				for t := range rowCh {
					steps := startPos + t + 1
					qr := q.Data[t*hs : (t+1)*hs]
					ar := attn.Data[t*hs : (t+1)*hs]
					for h := 0; h < cfg.Heads; h++ {
						m.attendHead(b, qr, ar, h, group, steps, scores[:steps])
					}
				}
			}()
		}
		wg.Wait()

		proj := mmb(attn, b.wo, b.qo, nil)
		for i := range x.Data {
			x.Data[i] += proj.Data[i]
		}

		norm(b.ln2)
		gate := mmb(a, b.wGate, b.qGate, nil)
		up := mmb(a, b.wUp, b.qUp, nil)
		for i, g0 := range gate.Data {
			g := float64(g0)
			gate.Data[i] = float32(g/(1+math.Exp(-g))) * up.Data[i]
		}
		down := mmb(gate, b.wDown, b.qDown, nil)
		for i := range x.Data {
			x.Data[i] += down.Data[i]
		}
	}
	last := x.Data[(n-1)*hs : n*hs]
	return mv(rmsnorm(last, m.normW, cfg.RMSEps), m.lmT, m.qLmT, nil)
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
		// Short contexts run the heads serially; past that the goroutine
		// cost disappears into the O(steps*headSz) work per head.
		if workers := min(runtime.NumCPU(), cfg.Heads); workers > 1 && steps >= 64 {
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()
					scores := make([]float64, steps)
					for h := w; h < cfg.Heads; h += workers {
						m.attendHead(b, q, attn, h, group, steps, scores)
					}
				}(w)
			}
			wg.Wait()
		} else {
			scores := make([]float64, steps)
			for h := 0; h < cfg.Heads; h++ {
				m.attendHead(b, q, attn, h, group, steps, scores)
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

package main

// GGUF loading: one downloaded .gguf file carries the config (typed
// metadata), the tokenizer (embedded vocab, merges, and pre-tokenizer
// tag), and the weights, so -gguf runs a llama.cpp checkpoint with no
// other files. Weights dequantize to float32 on the way out of the
// container and then requantize through the same -q8/-q4 path as the
// safetensors loader. llama.cpp's converter permutes the attention q/k
// projections into interleaved rope order for the llama architecture
// only (its ROPE_NORM style; qwen2 stays half-split NEOX), so for llama
// checkpoints the permutation is undone here and the half-split RoPE
// code sees HF row order.

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/gguf"
	"github.com/mattn/tensai/tokenizer"
)

// ggufTokenizer rebuilds a tokenizer.json from the metadata llama.cpp
// embeds and hands it to the tokenizer package's parser.
func ggufTokenizer(g *gguf.File) (*tokenizer.Tokenizer, error) {
	if model, _ := g.String("tokenizer.ggml.model"); model != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer model %q (byte-level BPE only)", model)
	}
	toksAny, ok := g.KV("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("gguf has no embedded tokenizer")
	}
	mergesAny, _ := g.KV("tokenizer.ggml.merges")
	typesAny, _ := g.KV("tokenizer.ggml.token_type")

	pre, _ := g.String("tokenizer.ggml.pre")
	var preJSON string
	switch pre {
	case "smollm":
		preJSON = `{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","use_regex":true}]}`
	case "qwen2", "llama-bpe", "llama3", "smaug-bpe":
		preJSON = `{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}`
	case "gpt-2", "olmo", "":
		preJSON = `{"type":"ByteLevel","use_regex":true}`
	default:
		return nil, fmt.Errorf("unsupported pre-tokenizer tag %q", pre)
	}

	tokens := toksAny.([]any)
	vocab := make(map[string]int, len(tokens))
	for id, t := range tokens {
		s, ok := t.(string)
		if !ok {
			return nil, fmt.Errorf("token %d is not a string", id)
		}
		vocab[s] = id
	}
	var merges []string
	if arr, ok := mergesAny.([]any); ok {
		merges = make([]string, len(arr))
		for i, m := range arr {
			merges[i], _ = m.(string)
		}
	}
	type added struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	}
	var specials []added
	if arr, ok := typesAny.([]any); ok {
		for id, tp := range arr {
			// Type 3 marks control tokens (<|im_start|> and friends), type
			// 4 user-defined added tokens (Qwen3's <think> tags).
			if n, ok := tp.(int32); ok && (n == 3 || n == 4) && id < len(tokens) {
				specials = append(specials, added{ID: id, Content: tokens[id].(string)})
			}
		}
	}

	spec := map[string]any{
		"pre_tokenizer": json.RawMessage(preJSON),
		"added_tokens":  specials,
		"model": map[string]any{
			"type":   "BPE",
			"vocab":  vocab,
			"merges": merges,
		},
	}
	raw, err := json.Marshal(spec)
	if err != nil {
		return nil, err
	}
	return tokenizer.Parse(raw)
}

// unpermuteRows reverses llama.cpp's rope interleave on a projection's
// output rows: gguf row (head, i, s) moves back to HF row (head, s, i).
func unpermuteRows(m *tensai.Matrix, heads int) {
	dim := m.Rows / heads
	half := dim / 2
	out := make([]float32, len(m.Data))
	for h := 0; h < heads; h++ {
		for s := 0; s < 2; s++ {
			for i := 0; i < half; i++ {
				src := (h*dim + i*2 + s) * m.Cols
				dst := (h*dim + s*half + i) * m.Cols
				copy(out[dst:dst+m.Cols], m.Data[src:src+m.Cols])
			}
		}
	}
	m.Data = out
}

func unpermuteVec(v []float32, heads int) []float32 {
	if v == nil || heads == 0 {
		return v
	}
	m := &tensai.Matrix{Rows: len(v), Cols: 1, Data: v}
	unpermuteRows(m, heads)
	return m.Data
}

// loadGGUF builds the model and its tokenizer from a single .gguf file,
// quantizing each weight to `bits` (0 keeps float32) as it loads.
func loadGGUF(path string, bits int) (*qwen, *tokenizer.Tokenizer, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer g.Close()

	arch, _ := g.String("general.architecture")
	switch arch {
	case "llama", "qwen2", "qwen3", "smollm3":
	default:
		return nil, nil, fmt.Errorf("unsupported architecture %q (this example speaks qwen2, qwen3, llama, and smollm3)", arch)
	}
	meta := func(key string) int64 {
		n, _ := g.Int(arch + "." + key)
		return n
	}
	var cfg config
	cfg.ModelType = arch
	cfg.HiddenSize = int(meta("embedding_length"))
	cfg.Intermediate = int(meta("feed_forward_length"))
	cfg.Layers = int(meta("block_count"))
	cfg.Heads = int(meta("attention.head_count"))
	cfg.KVHeads = int(meta("attention.head_count_kv"))
	cfg.MaxPos = int(meta("context_length"))
	cfg.Vocab = int(meta("vocab_size"))
	cfg.HeadDim = int(meta("attention.key_length"))
	cfg.RMSEps, _ = g.Float(arch + ".attention.layer_norm_rms_epsilon")
	cfg.RopeTheta, _ = g.Float(arch + ".rope.freq_base")
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.HiddenSize == 0 || cfg.Layers == 0 || cfg.Heads == 0 || cfg.KVHeads == 0 {
		return nil, nil, fmt.Errorf("gguf is missing %s.* dimensions", arch)
	}

	tok, err := ggufTokenizer(g)
	if err != nil {
		return nil, nil, err
	}

	tensor := func(name string) *tensai.Tensor {
		t, err := g.Tensor(name)
		if err != nil {
			panic(err)
		}
		return t
	}
	vecOpt := func(name string) []float32 {
		t, err := g.Tensor(name)
		if err != nil {
			return nil
		}
		return t.Data
	}
	// Projection weights arrive [out, in] like HF's; transpose for the
	// matvec and quantize immediately so the float32 copy dies here.
	trans := func(name string, unpermute int) *tensai.Matrix {
		m, err := tensor(name).Matrix()
		if err != nil {
			panic(err)
		}
		if unpermute > 0 {
			unpermuteRows(m, unpermute)
		}
		return m.T()
	}
	quant := func(w *tensai.Matrix) (*tensai.Matrix, *qmat) {
		if bits == 0 {
			return w, nil
		}
		return nil, quantizeMat(w, bits)
	}
	lin := func(name string, unpermute int) (*tensai.Matrix, *qmat) {
		return quant(trans(name, unpermute))
	}

	// Only llama-converter architectures store q/k rows permuted
	// (llama.cpp's SmolLM3 converter subclasses the Llama one).
	qPerm, kPerm := 0, 0
	if arch == "llama" || arch == "smollm3" {
		qPerm, kPerm = cfg.Heads, cfg.KVHeads
	}
	headSz := cfg.HiddenSize / cfg.Heads
	if cfg.HeadDim != 0 {
		headSz = cfg.HeadDim
	}
	m := &qwen{cfg: cfg, headSz: headSz}
	m.embed = tensor("token_embd.weight")
	m.normW = tensor("output_norm.weight").Data
	m.blocks = make([]qblock, cfg.Layers)
	// Layers load concurrently, same as the safetensors path: dequantize,
	// transpose, and requantize are CPU-bound and independent per layer.
	var wg sync.WaitGroup
	sem := make(chan struct{}, min(runtime.NumCPU(), 8))
	stage := layerStage(cfg, headSz)
	// The lm head loads alongside the layers.
	wg.Add(1)
	go func() {
		defer wg.Done()
		lmStage := 3 * 4 * int64(cfg.Vocab) * int64(cfg.HiddenSize)
		got := loadGate.acquire(lmStage)
		defer loadGate.release(got)
		if _, _, ok := g.Info("output.weight"); ok {
			m.lmT, m.qLmT = lin("output.weight", 0)
			return
		}
		// Tied embedding.
		em, err := m.embed.Matrix()
		if err != nil {
			panic(err)
		}
		lmT := em.T()
		if bits == 0 {
			m.lmT = lmT
		} else {
			m.qLmT = quantizeMat(lmT, bits)
		}
	}()
	for i := range m.blocks {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			got := loadGate.acquire(stage)
			defer loadGate.release(got)
			b := &m.blocks[i]
			// SmolLM3 skips RoPE on every fourth layer; the GGUF carries no
			// flag for it, matching llama.cpp's hardcoded rule.
			b.noPE = arch == "smollm3" && i%4 == 3
			p := fmt.Sprintf("blk.%d.", i)
			b.ln1 = tensor(p + "attn_norm.weight").Data
			b.ln2 = tensor(p + "ffn_norm.weight").Data
			b.qNorm = vecOpt(p + "attn_q_norm.weight")
			b.kNorm = vecOpt(p + "attn_k_norm.weight")
			b.wQKV, b.qQKV = quant(hcat([]*tensai.Matrix{
				trans(p+"attn_q.weight", qPerm),
				trans(p+"attn_k.weight", kPerm),
				trans(p+"attn_v.weight", 0),
			}))
			b.bQKV = catVec(
				unpermuteVec(vecOpt(p+"attn_q.bias"), qPerm),
				unpermuteVec(vecOpt(p+"attn_k.bias"), kPerm),
				vecOpt(p+"attn_v.bias"))
			b.wo, b.qo = lin(p+"attn_output.weight", 0)
			b.wGU, b.qGU = quant(hcat([]*tensai.Matrix{
				trans(p+"ffn_gate.weight", 0),
				trans(p+"ffn_up.weight", 0),
			}))
			b.wDown, b.qDown = lin(p+"ffn_down.weight", 0)
		}(i)
	}
	wg.Wait()
	return m, tok, nil
}

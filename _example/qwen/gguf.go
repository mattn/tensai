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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"sync"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/gguf"
	"github.com/mattn/tensai/tokenizer"
)

// ggufTokenizer rebuilds a tokenizer.json from the metadata llama.cpp
// embeds and hands it to the tokenizer package's parser.
func ggufTokenizer(g *gguf.File) (*tokenizer.Tokenizer, error) {
	switch model, _ := g.String("tokenizer.ggml.model"); model {
	case "gpt2":
	case "llama": // SentencePiece: Gemma and the Llama-2 family
		return ggufSPMTokenizer(g)
	default:
		return nil, fmt.Errorf("unsupported tokenizer model %q", model)
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

// repackQ8 copies a Q8_0 tensor's blocks — laid out [out, in] with one
// f16 scale per 32 input values — into columns [colOff, colOff+out) of a
// transposed Q8GMatrix, integer work only: the weights never pass through
// float32. colMap permutes output rows on the way in (the llama-family
// q/k rope unpermutation).
func repackQ8(dst *tensai.Q8GMatrix, raw []byte, out, in, colOff int, colMap func(int) int) {
	nb := in / 32
	for r := 0; r < out; r++ {
		j := colOff + r
		if colMap != nil {
			j = colOff + colMap(r)
		}
		for b := 0; b < nb; b++ {
			blk := raw[(r*nb+b)*34:]
			dst.Scale[b*dst.Cols+j] = gguf.Float16(binary.LittleEndian.Uint16(blk))
			var sum int32
			for i := 0; i < 32; i++ {
				w := int8(blk[2+i])
				gi := b*32 + i
				dst.Q[(gi/4)*4*dst.Cols+4*j+gi%4] = w
				sum += int32(w)
			}
			dst.ColSum64[b*dst.Cols+j] = 64 * sum
		}
	}
}

// repackQ4 copies a Q4_0 tensor's blocks — [out, in] with an f16 scale
// and 32 offset-binary nibbles apiece (low nibbles first, then high) —
// into columns [colOff, colOff+out) of a transposed Group-32 Q4Matrix.
// The nibble encoding matches tensai's exactly, so this is integer work
// plus one f16 widen per block.
func repackQ4(dst *tensai.Q4Matrix, raw []byte, out, in, colOff int, colMap func(int) int) {
	nb := in / 32
	for r := 0; r < out; r++ {
		j := colOff + r
		if colMap != nil {
			j = colOff + colMap(r)
		}
		for b := 0; b < nb; b++ {
			blk := raw[(r*nb+b)*18:]
			dst.Scale[b*dst.Cols+j] = gguf.Float16(binary.LittleEndian.Uint16(blk))
			for l := 0; l < 16; l++ {
				q := blk[2+l]
				iLo := b*32 + l
				iHi := b*32 + l + 16
				dst.Q[(iLo/4)*2*dst.Cols+2*j+(iLo%4)/2] |= (q & 0x0F) << (4 * (iLo % 2))
				dst.Q[(iHi/4)*2*dst.Cols+2*j+(iHi%4)/2] |= (q >> 4) << (4 * (iHi % 2))
			}
		}
	}
}

// ggufSPMTokenizer builds a SentencePiece tokenizer from the embedded
// vocabulary, scores, and token types.
func ggufSPMTokenizer(g *gguf.File) (*tokenizer.Tokenizer, error) {
	toksAny, ok := g.KV("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("gguf has no embedded tokenizer")
	}
	scoresAny, ok := g.KV("tokenizer.ggml.scores")
	if !ok {
		return nil, fmt.Errorf("gguf spm tokenizer has no scores")
	}
	typesAny, _ := g.KV("tokenizer.ggml.token_type")
	ta := toksAny.([]any)
	sa := scoresAny.([]any)
	ya := typesAny.([]any)
	tokens := make([]string, len(ta))
	scores := make([]float32, len(ta))
	types := make([]int32, len(ta))
	for i := range ta {
		tokens[i], _ = ta[i].(string)
		scores[i], _ = sa[i].(float32)
		types[i], _ = ya[i].(int32)
	}
	pre := false
	if v, ok := g.KV("tokenizer.ggml.add_space_prefix"); ok {
		pre, _ = v.(bool)
	}
	return tokenizer.NewSPM(tokens, scores, types, pre)
}

// unpermuteMap returns the llama rope unpermutation as a row index map,
// or nil when heads is zero.
func unpermuteMap(rows, heads int) func(int) int {
	if heads == 0 {
		return nil
	}
	dim := rows / heads
	half := dim / 2
	return func(r int) int {
		h := r / dim
		c := r % dim
		return h*dim + (c%2)*half + c/2
	}
}

// loadGGUF builds the model and its tokenizer from a single .gguf file,
// quantizing each weight to `bits` (0 keeps float32) as it loads. With
// bits == 8 and direct set, tensors stored as Q8_0 repack straight into
// grouped-int8 matrices — no dequantize/requantize round trip, and finer
// (32-row) scales than the float path would produce. The GPU path has no
// grouped kernel yet, so -gpu passes direct=false.
func loadGGUF(path string, bits int, direct bool) (*qwen, *tokenizer.Tokenizer, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer g.Close()

	arch, _ := g.String("general.architecture")
	switch arch {
	case "llama", "qwen2", "qwen3", "smollm3", "gemma3":
	default:
		return nil, nil, fmt.Errorf("unsupported architecture %q (this example speaks qwen2, qwen3, llama, smollm3, and gemma3)", arch)
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
	cfg.SlidingWin = int(meta("attention.sliding_window"))
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
	// allQ8 reports whether every named tensor is stored as Q8_0, the
	// precondition for the direct repack.
	allQ8 := func(names ...string) bool {
		if bits != 8 || !direct {
			return false
		}
		for _, name := range names {
			if typ, _, ok := g.Info(name); !ok || typ != "Q8_0" {
				return false
			}
		}
		return true
	}
	// allQ4 is the int4 twin of allQ8.
	allQ4 := func(names ...string) bool {
		if bits != 4 || !direct {
			return false
		}
		for _, name := range names {
			if typ, _, ok := g.Info(name); !ok || typ != "Q4_0" {
				return false
			}
		}
		return true
	}
	// linDirect4 repacks Q4_0 tensors into a fused Group-32 Q4Matrix.
	linDirect4 := func(names []string, perms []int) *qmat {
		var outs []int
		var in int
		for _, name := range names {
			_, shape, _ := g.Info(name)
			outs = append(outs, shape[0])
			in = shape[1]
		}
		total := 0
		for _, o := range outs {
			total += o
		}
		quads := (in + 3) / 4
		groups := (in + 31) / 32
		dst := &tensai.Q4Matrix{
			Rows:  in,
			Cols:  total,
			Q:     make([]uint8, quads*2*total+32),
			Scale: make([]float32, groups*total),
			Group: 32,
		}
		colOff := 0
		for i, name := range names {
			_, raw, err := g.RawTensor(name)
			if err != nil {
				panic(err)
			}
			repackQ4(dst, raw, outs[i], in, colOff, unpermuteMap(outs[i], perms[i]))
			colOff += outs[i]
		}
		return &qmat{cols: dst.Cols, f: dst.MatVec, mm: dst.MatMul}
	}
	// linDirect repacks one or more Q8_0 tensors into a fused grouped-int8
	// matrix, column ranges concatenated in order; perms maps each part's
	// output rows (nil for none).
	linDirect := func(names []string, perms []int) *qmat {
		var outs []int
		var in int
		for _, name := range names {
			_, shape, _ := g.Info(name)
			outs = append(outs, shape[0])
			in = shape[1]
		}
		total := 0
		for _, o := range outs {
			total += o
		}
		dst := tensai.NewQ8GMatrix(in, total)
		colOff := 0
		for i, name := range names {
			_, raw, err := g.RawTensor(name)
			if err != nil {
				panic(err)
			}
			repackQ8(dst, raw, outs[i], in, colOff, unpermuteMap(outs[i], perms[i]))
			colOff += outs[i]
		}
		return &qmat{cols: dst.Cols, f: dst.MatVec, mm: dst.MatMul}
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
	if arch == "gemma3" {
		// Gemma scales embeddings by sqrt(hidden) before the first block.
		s := float32(math.Sqrt(float64(cfg.HiddenSize)))
		for i := range m.embed.Data {
			m.embed.Data[i] *= s
		}
	}
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
		if _, _, ok := g.Info("output.weight"); ok {
			if allQ8("output.weight") {
				m.qLmT = linDirect([]string{"output.weight"}, []int{0})
				return
			}
			if allQ4("output.weight") {
				m.qLmT = linDirect4([]string{"output.weight"}, []int{0})
				return
			}
			lmStage := 3 * 4 * int64(cfg.Vocab) * int64(cfg.HiddenSize)
			got := loadGate.acquire(lmStage)
			defer loadGate.release(got)
			m.lmT, m.qLmT = lin("output.weight", 0)
			return
		}
		// Tied embedding: the quantized blocks repack directly too.
		if allQ8("token_embd.weight") {
			m.qLmT = linDirect([]string{"token_embd.weight"}, []int{0})
			return
		}
		if allQ4("token_embd.weight") {
			m.qLmT = linDirect4([]string{"token_embd.weight"}, []int{0})
			return
		}
		lmStage := 3 * 4 * int64(cfg.Vocab) * int64(cfg.HiddenSize)
		got := loadGate.acquire(lmStage)
		defer loadGate.release(got)
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
			if arch == "gemma3" {
				// Five of every six layers attend over a sliding window
				// with the local rope base; the sixth is global. Sandwich
				// norms and the gelu-tanh gate round out the block. (The
				// converter already folds Gemma's +1 into norm weights.)
				b.geglu = true
				if (i+1)%6 != 0 {
					b.window = cfg.SlidingWin
					b.ropeTheta = 10000
				}
			}
			p := fmt.Sprintf("blk.%d.", i)
			b.ln1 = tensor(p + "attn_norm.weight").Data
			b.ln2 = tensor(p + "ffn_norm.weight").Data
			b.qNorm = vecOpt(p + "attn_q_norm.weight")
			b.kNorm = vecOpt(p + "attn_k_norm.weight")
			b.postAttn = vecOpt(p + "post_attention_norm.weight")
			b.postFFN = vecOpt(p + "post_ffw_norm.weight")
			layerNames := []string{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "attn_output.weight", p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"}
			if allQ8(layerNames...) {
				b.qQKV = linDirect(
					[]string{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"},
					[]int{qPerm, kPerm, 0})
				b.qo = linDirect([]string{p + "attn_output.weight"}, []int{0})
				b.qGU = linDirect([]string{p + "ffn_gate.weight", p + "ffn_up.weight"}, []int{0, 0})
				b.qDown = linDirect([]string{p + "ffn_down.weight"}, []int{0})
			} else if allQ4(layerNames...) {
				b.qQKV = linDirect4(
					[]string{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"},
					[]int{qPerm, kPerm, 0})
				b.qo = linDirect4([]string{p + "attn_output.weight"}, []int{0})
				b.qGU = linDirect4([]string{p + "ffn_gate.weight", p + "ffn_up.weight"}, []int{0, 0})
				b.qDown = linDirect4([]string{p + "ffn_down.weight"}, []int{0})
			} else {
				b.wQKV, b.qQKV = quant(hcat([]*tensai.Matrix{
					trans(p+"attn_q.weight", qPerm),
					trans(p+"attn_k.weight", kPerm),
					trans(p+"attn_v.weight", 0),
				}))
				b.wo, b.qo = lin(p+"attn_output.weight", 0)
				b.wGU, b.qGU = quant(hcat([]*tensai.Matrix{
					trans(p+"ffn_gate.weight", 0),
					trans(p+"ffn_up.weight", 0),
				}))
				b.wDown, b.qDown = lin(p+"ffn_down.weight", 0)
			}
			b.bQKV = catVec(
				unpermuteVec(vecOpt(p+"attn_q.bias"), qPerm),
				unpermuteVec(vecOpt(p+"attn_k.bias"), kPerm),
				vecOpt(p+"attn_v.bias"))
		}(i)
	}
	wg.Wait()
	return m, tok, nil
}

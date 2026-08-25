package main

// Qwen2-, Qwen3-, and Llama-family inference: pre-norm transformer
// blocks with RMSNorm, rotary position embeddings, grouped-query
// attention, and a SwiGLU MLP, decoded one token at a time with a KV
// cache. The architectures share this block: Llama drops the attention
// biases, and Qwen3 drops them too while adding per-head QK-norm and an
// explicit head_dim (so the query dimension need not equal the hidden
// size). Dimensions come from config.json and any such checkpoint whose
// tokenizer.json is byte-level BPE works; every matvec runs on tensai's
// Dot kernel or, with -q8, the int8 kernel.

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
	HeadDim      int     `json:"head_dim"`
	NoRopeLayers []int   `json:"no_rope_layers"` // smollm3: 1 = RoPE, 0 = NoPE
	SlidingWin   int     `json:"sliding_window"` // gemma3: local-attention span
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
	// ChatStyle overrides the template family when it differs from the
	// architecture — DeepSeek's R1 distills are qwen2/llama blocks that
	// speak DeepSeek's turn markers. Set by the GGUF loader, never JSON.
	ChatStyle string `json:"-"`
	// MoE dimensions (qwen2moe/qwen3moe/gpt-oss), from GGUF metadata.
	NExpert     int `json:"-"`
	NExpertUsed int `json:"-"`
	MoeFF       int `json:"-"`
	SharedFF    int `json:"-"`
	// YaRN rope scaling (gpt-oss), from GGUF metadata; Factor 0 means
	// plain rope.
	YarnFactor   float64 `json:"-"`
	YarnOrigCtx  int     `json:"-"`
	YarnBetaFast float64 `json:"-"`
	YarnBetaSlow float64 `json:"-"`
}

// qmat abstracts the int8 and int4 twins behind one matvec call, plus the
// batched form prefill uses.
type qmat struct {
	cols int
	f    func(x, out []float32) error
	mm   func(x, out *tensai.Matrix) error
	q8   *tensai.QMatrix  // retained for GPU upload
	q4   *tensai.Q4Matrix // likewise, for the int4 twin
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
		return &qmat{cols: q.Cols, f: q.MatVec, mm: q.MatMul, q4: q}
	}
	panic("unsupported quantization width")
}

// qblock holds one layer's weights. The q, k, and v projections fuse
// into one matrix (columns [q | k | v]) and gate/up into another, so a
// decode step streams four weight matrices instead of seven — quantization
// is per column, so the fused results are bit-identical to separate calls.
type qblock struct {
	ln1, ln2     []float32
	qNorm, kNorm []float32 // Qwen3/Gemma per-head QK-norm; nil otherwise
	postAttn     []float32 // Gemma sandwich norms; nil otherwise
	postFFN      []float32
	noPE         bool           // smollm3: every fourth layer skips RoPE
	window       int            // gemma3: sliding-attention span; 0 = full
	ropeTheta    float64        // per-layer rope base; 0 = the config default
	ropeFreq     []float64      // precomputed theta^(-2*i/head_dim)
	geglu        bool           // gemma3: gelu-tanh gate instead of silu
	wQKV, wo     *tensai.Matrix // [in, out] after transposing HF's [out, in]
	bQKV         []float32
	wGU, wDown   *tensai.Matrix
	qQKV, qo     *qmat
	qGU, qDown   *qmat
	// MoE: the router picks topK of the experts, whose small FFNs run in
	// place of the dense gate/up/down; qwen2moe adds an always-on shared
	// expert scaled by a sigmoid gate.
	router     *tensai.Matrix // [hidden, nExpert], float — tiny
	routerBias []float32
	experts    []expertFFN
	topK       int
	normTopK   bool      // qwen3moe renormalizes the top-k weights
	softmaxK   bool      // gpt-oss: softmax over the top-k logits, not all
	oaiGLU     bool      // gpt-oss: clamped swiglu with the (up+1) linear term
	sinks      []float32 // gpt-oss: per-head extra softmax logit
	bo         []float32 // attention output bias (gpt-oss)
	sharedGU   *qmat
	sharedDown *qmat
	sharedGate []float32   // [hidden]; sigmoid(dot) scales the shared expert
	kc, vc     [][]float32 // KV cache, kvHeads*headDim per position
}

// expertFFN is one expert's SwiGLU: fused gate|up and down projections,
// with gpt-oss carrying per-expert biases.
type expertFFN struct {
	qGU, qDown       *qmat
	guBias, downBias []float32
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
	switch c.ModelType {
	case "qwen2", "qwen3", "llama", "smollm3":
	default:
		return c, fmt.Errorf("unsupported model_type %q (this example speaks qwen2, qwen3, llama, and smollm3)", c.ModelType)
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
	// linqFused transposes and column-concatenates several [out, in]
	// weights into one [in, sum(out)] matrix before quantizing.
	linqFused := func(names ...string) (*tensai.Matrix, *qmat) {
		var parts []*tensai.Matrix
		for _, name := range names {
			t, err := f.Tensor(name)
			if err != nil {
				panic(err)
			}
			m, err := t.Matrix()
			if err != nil {
				panic(err)
			}
			parts = append(parts, m.T())
		}
		w := hcat(parts)
		if bits == 0 {
			return w, nil
		}
		return nil, quantizeMat(w, bits)
	}

	headSz := cfg.HiddenSize / cfg.Heads
	if cfg.HeadDim != 0 {
		headSz = cfg.HeadDim
	}
	m := &qwen{cfg: cfg, headSz: headSz}
	m.embed, err = f.Tensor("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	m.normW = vec("model.norm.weight")
	m.blocks = make([]qblock, cfg.Layers)
	// Layers load concurrently: reads are ReadAt against one descriptor
	// and the convert/transpose/quantize chain per tensor is CPU-bound, so
	// this is where an otherwise serial load spends nearly all its time.
	// The worker cap bounds the float32 staging copies alive at once.
	var wg sync.WaitGroup
	sem := make(chan struct{}, min(runtime.NumCPU(), 8))
	stage := layerStage(cfg, m.headSz)
	// The lm head — the largest single tensor — transposes and quantizes
	// alongside the layers (its own column loop is parallel too).
	wg.Add(1)
	go func() {
		defer wg.Done()
		lmStage := 3 * 4 * int64(cfg.Vocab) * int64(cfg.HiddenSize)
		got := loadGate.acquire(lmStage)
		defer loadGate.release(got)
		if cfg.TieEmbedding {
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
		} else {
			m.lmT, m.qLmT = linq("lm_head.weight")
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
			b.noPE = i < len(cfg.NoRopeLayers) && cfg.NoRopeLayers[i] == 0
			p := fmt.Sprintf("model.layers.%d.", i)
			b.ln1 = vec(p + "input_layernorm.weight")
			b.ln2 = vec(p + "post_attention_layernorm.weight")
			b.qNorm = vecOpt(p + "self_attn.q_norm.weight")
			b.kNorm = vecOpt(p + "self_attn.k_norm.weight")
			b.wQKV, b.qQKV = linqFused(p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight", p+"self_attn.v_proj.weight")
			b.bQKV = catVec(vecOpt(p+"self_attn.q_proj.bias"), vecOpt(p+"self_attn.k_proj.bias"), vecOpt(p+"self_attn.v_proj.bias"))
			b.wo, b.qo = linq(p + "self_attn.o_proj.weight")
			b.wGU, b.qGU = linqFused(p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight")
			b.wDown, b.qDown = linq(p + "mlp.down_proj.weight")
		}(i)
	}
	wg.Wait()
	m.initRopeFreqs()
	return m, nil
}

func (m *qwen) initRopeFreqs() {
	half := m.headSz / 2
	for i := range m.blocks {
		b := &m.blocks[i]
		if b.noPE || half == 0 {
			continue
		}
		theta := b.ropeTheta
		if theta == 0 {
			theta = m.cfg.RopeTheta
		}
		b.ropeFreq = make([]float64, half)
		for j := range b.ropeFreq {
			b.ropeFreq[j] = math.Pow(theta, -2*float64(j)/float64(m.headSz))
		}
	}
}

// memGate bounds the float32 staging bytes alive at once during a
// parallel load: workers acquire their estimated footprint and block
// until it fits the budget. A request larger than the whole budget is
// capped, so it still runs — alone. Without this, eight workers each
// staging a 7B-class layer (over a gigabyte of sources, transposes, and
// fused copies apiece, and the untied lm head several times that) can
// exhaust the machine outright.
type memGate struct {
	mu    sync.Mutex
	cond  *sync.Cond
	avail int64
	total int64
}

func newMemGate(budget int64) *memGate {
	g := &memGate{avail: budget, total: budget}
	g.cond = sync.NewCond(&g.mu)
	return g
}

func (g *memGate) acquire(n int64) int64 {
	n = min(n, g.total)
	g.mu.Lock()
	for g.avail < n {
		g.cond.Wait()
	}
	g.avail -= n
	g.mu.Unlock()
	return n
}

func (g *memGate) release(n int64) {
	g.mu.Lock()
	g.avail += n
	g.mu.Unlock()
	g.cond.Broadcast()
}

// loadGate is shared by both loaders; 2GB of in-flight staging keeps a
// 16-core load fast for small models and a 7B load bounded.
var loadGate = newMemGate(2 << 30)

// layerStage estimates one layer's peak float32 staging: the largest
// fused matrix (sources, transposes, and the concatenated copy overlap,
// so roughly three times its bytes).
func layerStage(cfg config, headSz int) int64 {
	qkv := int64(cfg.HiddenSize) * int64(cfg.Heads*headSz+2*cfg.KVHeads*headSz)
	gu := int64(cfg.HiddenSize) * int64(2*cfg.Intermediate)
	return 3 * 4 * max(qkv, gu)
}

// hcat concatenates matrices with equal rows along the columns.
func hcat(parts []*tensai.Matrix) *tensai.Matrix {
	rows := parts[0].Rows
	cols := 0
	for _, p := range parts {
		cols += p.Cols
	}
	out := tensai.NewMatrix(rows, cols)
	off := 0
	for _, p := range parts {
		for r := 0; r < rows; r++ {
			copy(out.Data[r*cols+off:], p.Data[r*p.Cols:(r+1)*p.Cols])
		}
		off += p.Cols
	}
	return out
}

// catVec concatenates bias vectors; all-nil stays nil (llama).
func catVec(vs ...[]float32) []float32 {
	var out []float32
	any := false
	for _, v := range vs {
		any = any || v != nil
		out = append(out, v...)
	}
	if !any {
		return nil
	}
	return out
}

func rmsnormInto(out, x, w []float32, eps float64) {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/float64(len(x))+eps)
	for i, v := range x {
		out[i] = float32(float64(v)*inv) * w[i]
	}
}

// activate applies the gated activation in place: silu(gate)*up, or
// Gemma's tanh-approximated gelu when geglu is set.
// swigluOAI is gpt-oss's clamped SwiGLU: gate = min(gate, 7),
// up in [-7, 7], out = gate*sigmoid(1.702*gate) * (up + 1).
func swigluOAI(gate, up []float32) {
	const alpha, limit = 1.702, 7.0
	for i, g := range gate {
		gd := math.Min(float64(g), limit)
		u := math.Min(math.Max(float64(up[i]), -limit), limit)
		gate[i] = float32(gd / (1 + math.Exp(-alpha*gd)) * (u + 1))
	}
}

func activate(gate, up []float32, geglu bool) {
	if !geglu {
		tensai.SiluMul(gate, up)
		return
	}
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i := range gate {
		g := float64(gate[i])
		gate[i] = float32(0.5*g*(1+math.Tanh(c*(g+0.044715*g*g*g)))) * up[i]
	}
}

// moeFFN runs one token's activation through a block's sparse FFN: the
// router's softmax picks topK experts, each contributes its SwiGLU
// output scaled by its routing weight, and qwen2moe's shared expert
// adds its sigmoid-gated output on top.
// moePick is one routed expert and its mixing weight.
type moePick struct {
	e int
	w float32
}

// route turns one row's router logits into its top-k experts and
// weights; logits are consumed as scratch.
func (b *qblock) route(logits []float32) []moePick {
	picks := make([]moePick, 0, b.topK)
	if b.softmaxK {
		// gpt-oss: pick the top-k logits, softmax over just those.
		for e, w := range logits {
			if len(picks) < b.topK {
				picks = append(picks, moePick{e, w})
				continue
			}
			lo := 0
			for i := 1; i < len(picks); i++ {
				if picks[i].w < picks[lo].w {
					lo = i
				}
			}
			if w > picks[lo].w {
				picks[lo] = moePick{e, w}
			}
		}
		maxL := picks[0].w
		for _, p := range picks[1:] {
			maxL = max(maxL, p.w)
		}
		var sum float64
		for i := range picks {
			picks[i].w = float32(math.Exp(float64(picks[i].w - maxL)))
			sum += float64(picks[i].w)
		}
		for i := range picks {
			picks[i].w = float32(float64(picks[i].w) / sum)
		}
	} else {
		// Qwen MoE: softmax over all experts, then pick from the
		// probabilities.
		maxL := logits[0]
		for _, v := range logits[1:] {
			maxL = max(maxL, v)
		}
		var sum float64
		for i, v := range logits {
			e := math.Exp(float64(v - maxL))
			logits[i] = float32(e)
			sum += e
		}
		inv := float32(1 / sum)
		for i := range logits {
			logits[i] *= inv
		}
		for e, w := range logits {
			if len(picks) < b.topK {
				picks = append(picks, moePick{e, w})
				continue
			}
			lo := 0
			for i := 1; i < len(picks); i++ {
				if picks[i].w < picks[lo].w {
					lo = i
				}
			}
			if w > picks[lo].w {
				picks[lo] = moePick{e, w}
			}
		}
		if b.normTopK {
			var ws float32
			for _, p := range picks {
				ws += p.w
			}
			for i := range picks {
				picks[i].w /= ws
			}
		}
	}
	return picks
}

func (m *qwen) moeFFN(b *qblock, a []float32) []float32 {
	picks := b.route(mv(a, b.router, nil, b.routerBias))
	out := make([]float32, m.cfg.HiddenSize)
	moeFF := m.cfg.MoeFF
	for _, p := range picks {
		ex := &b.experts[p.e]
		gu := mv(a, nil, ex.qGU, ex.guBias)
		gate, up := gu[:moeFF], gu[moeFF:]
		if b.oaiGLU {
			swigluOAI(gate, up)
		} else {
			tensai.SiluMul(gate, up)
		}
		d := mv(gate, nil, ex.qDown, ex.downBias)
		tensai.Axpy(p.w, d, out)
	}
	if b.sharedGU != nil {
		gu := mv(a, nil, b.sharedGU, nil)
		sf := m.cfg.SharedFF
		gate, up := gu[:sf], gu[sf:]
		tensai.SiluMul(gate, up)
		d := mv(gate, nil, b.sharedDown, nil)
		g := float32(1 / (1 + math.Exp(-float64(tensai.DotVec(b.sharedGate, a)))))
		tensai.Axpy(g, d, out)
	}
	return out
}

// mv computes x @ W (+ bias), on the quantized twin when it exists.
func mv(x []float32, w *tensai.Matrix, q *qmat, bias []float32) []float32 {
	cols := 0
	if q != nil {
		cols = q.cols
	} else {
		cols = w.Cols
	}
	out := make([]float32, cols)
	mvInto(out, x, w, q, bias)
	return out
}

// mvInto computes x @ W (+ bias) into out, on the quantized twin when it exists.
func mvInto(out, x []float32, w *tensai.Matrix, q *qmat, bias []float32) {
	if q != nil {
		if err := q.f(x, out); err != nil {
			panic(err)
		}
	} else {
		xm := &tensai.Matrix{Rows: 1, Cols: len(x), Data: x}
		om := &tensai.Matrix{Rows: 1, Cols: len(out), Data: out}
		if err := tensai.DotInto(om, xm, w); err != nil {
			panic(err)
		}
	}
	if bias != nil {
		for i := range out {
			out[i] += bias[i]
		}
	}
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
	// Sliding-window layers (Gemma) see only the last window positions.
	start := 0
	if b.window > 0 && steps > b.window {
		start = steps - b.window
	}
	qOff := h * m.headSz
	kvOff := (h / group) * m.headSz
	scale := 1 / math.Sqrt(float64(m.headSz))
	qh := q[qOff : qOff+m.headSz]
	maxs := math.Inf(-1)
	for t := start; t < steps; t++ {
		s := float64(tensai.DotVec(qh, b.kc[t][kvOff:kvOff+m.headSz])) * scale
		scores[t] = s
		if s > maxs {
			maxs = s
		}
	}
	if b.sinks != nil && float64(b.sinks[h]) > maxs {
		maxs = float64(b.sinks[h])
	}
	var sum float64
	for t := start; t < steps; t++ {
		scores[t] = math.Exp(scores[t] - maxs)
		sum += scores[t]
	}
	if b.sinks != nil {
		// The sink is an extra softmax slot with no value: it only
		// absorbs probability mass.
		sum += math.Exp(float64(b.sinks[h]) - maxs)
	}
	out := attn[qOff : qOff+m.headSz]
	for t := start; t < steps; t++ {
		tensai.Axpy(float32(scores[t]/sum), b.vc[t][kvOff:kvOff+m.headSz], out)
	}
}

// qkNorm applies Qwen3's per-head RMS normalization in place; w has one
// weight per head channel. A nil w (qwen2, llama) is a no-op.
func (m *qwen) qkNorm(v, w []float32) {
	if w == nil {
		return
	}
	for o := 0; o < len(v); o += m.headSz {
		rmsnormInto(v[o:o+m.headSz], v[o:o+m.headSz], w, m.cfg.RMSEps)
	}
}

// rope rotates one head in place, half-split style: pair (i, i+dh/2).
func (m *qwen) rope(h []float32, pos int, b *qblock) {
	theta := b.ropeTheta
	if theta == 0 {
		theta = m.cfg.RopeTheta
	}
	freqs := b.ropeFreq
	half := m.headSz / 2
	yarn := m.cfg.YarnFactor > 1
	var low, high, mscale float64
	if yarn {
		// ggml's YaRN: dimensions below `low` keep their train-time
		// frequencies (extrapolation), above `high` divide by the factor
		// (interpolation), with a linear ramp between; the magnitudes
		// scale by 1 + 0.1*ln(factor).
		corr := func(rot float64) float64 {
			return float64(m.headSz) * math.Log(float64(m.cfg.YarnOrigCtx)/(rot*2*math.Pi)) / (2 * math.Log(theta))
		}
		low = math.Max(0, math.Floor(corr(m.cfg.YarnBetaFast)))
		high = math.Min(float64(m.headSz-1), math.Ceil(corr(m.cfg.YarnBetaSlow)))
		mscale = 1 + 0.1*math.Log(m.cfg.YarnFactor)
	}
	for i := 0; i < half; i++ {
		freq := freqs[i]
		angle := float64(pos) * freq
		scale := 1.0
		if yarn {
			ramp := (float64(i) - low) / math.Max(0.001, high-low)
			ramp = 1 - math.Min(1, math.Max(0, ramp))
			angle = angle/m.cfg.YarnFactor*(1-ramp) + angle*ramp
			scale = mscale
		}
		sv, cv := math.Sincos(angle)
		sv, cv = sv*scale, cv*scale
		a, b := float64(h[i]), float64(h[i+half])
		h[i] = float32(a*cv - b*sv)
		h[i+half] = float32(b*cv + a*sv)
	}
}

// prefill feeds a batch of tokens starting at startPos, extending the KV
// cache, and returns the next-token logits after the last one. It computes
// exactly what feeding the tokens through step one by one would — the
// batched quantized matmul quantizes each activation row identically and
// the attention loops match — but streams each weight matrix once per
// batch instead of once per token. The lm_head runs only on the final
// position.
func (m *qwen) prefill(tokens []int, startPos int) []float32 {
	x := m.forwardBatch(tokens, startPos)
	hs := m.cfg.HiddenSize
	last := x.Data[(len(tokens)-1)*hs : len(tokens)*hs]
	a := make([]float32, hs)
	rmsnormInto(a, last, m.normW, m.cfg.RMSEps)
	return mv(a, m.lmT, m.qLmT, nil)
}

// prefillLogits is prefill with the lm_head applied to every position:
// row t holds the next-token logits after tokens[0..t]. Speculative
// decoding uses it to score a draft's proposals in one batched pass.
func (m *qwen) prefillLogits(tokens []int, startPos int) *tensai.Matrix {
	x := m.forwardBatch(tokens, startPos)
	hs := m.cfg.HiddenSize
	a := tensai.NewMatrix(x.Rows, hs)
	for t := 0; t < x.Rows; t++ {
		rmsnormInto(a.Data[t*hs:(t+1)*hs], x.Data[t*hs:(t+1)*hs], m.normW, m.cfg.RMSEps)
	}
	return mmb(a, m.lmT, m.qLmT, nil)
}

// truncate rolls the KV cache back to n positions — how speculative
// decoding discards the tail a rejected draft left behind.
func (m *qwen) truncate(n int) {
	for i := range m.blocks {
		b := &m.blocks[i]
		if len(b.kc) > n {
			b.kc = b.kc[:n]
			b.vc = b.vc[:n]
		}
	}
}

// forwardBatch runs the transformer blocks over a batch of tokens,
// extending the KV cache, and returns the hidden states.
func (m *qwen) forwardBatch(tokens []int, startPos int) *tensai.Matrix {
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
			rmsnormInto(a.Data[t*hs:(t+1)*hs], x.Data[t*hs:(t+1)*hs], w, cfg.RMSEps)
		}
	}
	qDim := cfg.Heads * m.headSz
	qkvW := qDim + 2*kvDim
	for li := range m.blocks {
		b := &m.blocks[li]
		norm(b.ln1)
		qkv := mmb(a, b.wQKV, b.qQKV, b.bQKV)
		for t := 0; t < n; t++ {
			pos := startPos + t
			row := qkv.Data[t*qkvW : (t+1)*qkvW]
			qr := row[:qDim]
			kr := row[qDim : qDim+kvDim]
			m.qkNorm(qr, b.qNorm)
			m.qkNorm(kr, b.kNorm)
			if !b.noPE {
				for h := 0; h < cfg.Heads; h++ {
					m.rope(qr[h*m.headSz:(h+1)*m.headSz], pos, b)
				}
				for h := 0; h < cfg.KVHeads; h++ {
					m.rope(kr[h*m.headSz:(h+1)*m.headSz], pos, b)
				}
			}
			// Copies detach the cache rows from the wide fused buffer.
			b.kc = append(b.kc, append(make([]float32, 0, kvDim), kr...))
			b.vc = append(b.vc, append(make([]float32, 0, kvDim), row[qDim+kvDim:]...))
		}

		// Causal attention: row t sees cache positions [0, startPos+t].
		// Rows are independent, so they fan out across CPUs.
		attn := tensai.NewMatrix(n, qDim)
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
					qr := qkv.Data[t*qkvW : t*qkvW+qDim]
					ar := attn.Data[t*qDim : (t+1)*qDim]
					for h := 0; h < cfg.Heads; h++ {
						m.attendHead(b, qr, ar, h, group, steps, scores[:steps])
					}
				}
			}()
		}
		wg.Wait()

		proj := mmb(attn, b.wo, b.qo, b.bo)
		if b.postAttn != nil {
			for t := 0; t < n; t++ {
				rmsnormInto(proj.Data[t*hs:(t+1)*hs], proj.Data[t*hs:(t+1)*hs], b.postAttn, cfg.RMSEps)
			}
		}
		for i := range x.Data {
			x.Data[i] += proj.Data[i]
		}

		norm(b.ln2)
		var down *tensai.Matrix
		if len(b.experts) > 0 {
			// Each row routes to its own experts, so the batch runs
			// row-wise, rows spread across CPUs. (An expert-grouped GEMM
			// was tried and lost: the four-row kernel's multiply work
			// scales with rows x weights either way, the per-expert
			// weights sit in L3 across rows, and the orchestration cost
			// is real — a deeper batch kernel is the actual lever.)
			down = tensai.NewMatrix(n, hs)
			var wg sync.WaitGroup
			rowCh := make(chan int, n)
			for t := 0; t < n; t++ {
				rowCh <- t
			}
			close(rowCh)
			for w := 0; w < min(runtime.NumCPU(), n); w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for t := range rowCh {
						copy(down.Data[t*hs:(t+1)*hs], m.moeFFN(b, a.Data[t*hs:(t+1)*hs]))
					}
				}()
			}
			wg.Wait()
		} else {
			gu := mmb(a, b.wGU, b.qGU, nil)
			inter := cfg.Intermediate
			gate := tensai.NewMatrix(n, inter)
			for t := 0; t < n; t++ {
				row := gu.Data[t*2*inter:]
				copy(gate.Data[t*inter:(t+1)*inter], row[:inter])
				activate(gate.Data[t*inter:(t+1)*inter], row[inter:2*inter], b.geglu)
			}
			down = mmb(gate, b.wDown, b.qDown, nil)
		}
		if b.postFFN != nil {
			for t := 0; t < n; t++ {
				rmsnormInto(down.Data[t*hs:(t+1)*hs], down.Data[t*hs:(t+1)*hs], b.postFFN, cfg.RMSEps)
			}
		}
		for i := range x.Data {
			x.Data[i] += down.Data[i]
		}
	}
	return x
}

// step feeds one token at position pos and returns the next-token logits.
func (m *qwen) step(token, pos int) []float32 {
	cfg := m.cfg
	hs := cfg.HiddenSize
	group := cfg.Heads / cfg.KVHeads

	x := make([]float32, hs)
	copy(x, m.embed.Data[token*hs:(token+1)*hs])
	a := make([]float32, hs)

	kvDim := cfg.KVHeads * m.headSz
	qDim := cfg.Heads * m.headSz
	qkvW := qDim + 2*kvDim
	qkv := make([]float32, qkvW)
	attn := make([]float32, qDim)
	proj := make([]float32, hs)
	gu := make([]float32, 2*cfg.Intermediate)
	downBuf := make([]float32, hs)
	for li := range m.blocks {
		b := &m.blocks[li]
		rmsnormInto(a, x, b.ln1, cfg.RMSEps)
		mvInto(qkv, a, b.wQKV, b.qQKV, b.bQKV)
		q := qkv[:qDim]
		k := qkv[qDim : qDim+kvDim]
		v := qkv[qDim+kvDim:]
		m.qkNorm(q, b.qNorm)
		m.qkNorm(k, b.kNorm)
		if !b.noPE {
			for h := 0; h < cfg.Heads; h++ {
				m.rope(q[h*m.headSz:(h+1)*m.headSz], pos, b)
			}
			for h := 0; h < cfg.KVHeads; h++ {
				m.rope(k[h*m.headSz:(h+1)*m.headSz], pos, b)
			}
		}
		// Copy k and v out of the fused row so the cache does not retain
		// the whole qkv buffer per position.
		b.kc = append(b.kc, append(make([]float32, 0, kvDim), k...))
		b.vc = append(b.vc, append(make([]float32, 0, kvDim), v...))

		clear(attn)
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
		mvInto(proj, attn, b.wo, b.qo, b.bo)
		if b.postAttn != nil {
			rmsnormInto(proj, proj, b.postAttn, cfg.RMSEps)
		}
		for i := range x {
			x[i] += proj[i]
		}

		rmsnormInto(a, x, b.ln2, cfg.RMSEps)
		var down []float32
		if len(b.experts) > 0 {
			down = m.moeFFN(b, a)
		} else {
			mvInto(gu, a, b.wGU, b.qGU, nil)
			gate, up := gu[:cfg.Intermediate], gu[cfg.Intermediate:]
			activate(gate, up, b.geglu)
			mvInto(downBuf, gate, b.wDown, b.qDown, nil)
			down = downBuf
		}
		if b.postFFN != nil {
			rmsnormInto(down, down, b.postFFN, cfg.RMSEps)
		}
		for i := range x {
			x[i] += down[i]
		}
	}

	rmsnormInto(a, x, m.normW, cfg.RMSEps)
	return mv(a, m.lmT, m.qLmT, nil)
}

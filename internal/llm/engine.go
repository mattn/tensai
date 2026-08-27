// Package llm wires tensai's kernels into a runnable language model:
// checkpoint download and loading, chat templates, sampling, generation
// (plain and speculative), the GPU decode path, and the OpenAI-compatible
// server. cmd/tensai and _example/qwen are thin flag parsers over Engine.
package llm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/tokenizer"
)

// DefaultRepo is the checkpoint Options.Repo defaults to.
const DefaultRepo = "Qwen/Qwen2.5-0.5B-Instruct"

// DefaultSystem is the system prompt Qwen-family models get; other
// families swap it for a neutral one (see Open), exactly when the caller
// left it at this default.
const DefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

// DefaultDataDir is where a repo's files live when Options.Data is
// empty: a per-repo directory under the user cache (~/.cache/tensai/...
// on Linux).
func DefaultDataDir(repo string) string {
	if repo == "" {
		repo = DefaultRepo
	}
	if dir, err := os.UserCacheDir(); err == nil {
		return filepath.Join(dir, "tensai", path.Base(repo))
	}
	return filepath.Join("tensai-data", path.Base(repo))
}

// Options selects and configures a model. The zero value is not runnable:
// fill Data or GGUF, and usually Bits.
type Options struct {
	Data    string // directory for downloaded model files
	Repo    string // Hugging Face repo for missing files; DefaultRepo if empty
	GGUF    string // load model and tokenizer from a single .gguf instead
	Bits    int    // decode weights: 0 float32, 8 int8, 4 int4
	GPU     bool   // decode on the GPU (needs Bits and a wgpu build tag)
	Requant bool   // requantize gguf weights through float32
	NoCache bool   // skip the gguf repack cache file
	Draft   string // data directory of a smaller draft model (speculative decoding)
	SpecK   int    // draft tokens per speculative step
	Think   bool   // let Qwen3-family models reason in a <think> block
	System  string // system prompt; DefaultSystem adapts per model family
	Temp    float64
	TopP    float64
	Seed    int64
	Log     io.Writer // load/timing chatter; nil silences it
}

// Engine is a loaded model ready to generate: the tokenizer, the chat
// template, optionally a draft model and a GPU residency.
type Engine struct {
	opts    Options
	model   *qwen
	draft   *qwen
	tok     *tokenizer.Tokenizer
	tm      tmpl
	system  string
	imEnd   int
	eot     int
	nCtx    int
	g       *tensai.GPU
	gq      *gpuQwen
	prefill func([]int, int) []float32
	step    func(int, int) []float32
	reset   func()
	rng     *rand.Rand

	steps  int
	logits []float32
}

// Open downloads (if needed) and loads the model, the optional draft
// model, and the GPU residency, returning an Engine positioned at an
// empty context.
func Open(o Options) (*Engine, error) {
	if o.Repo == "" {
		o.Repo = DefaultRepo
	}
	if o.Log == nil {
		o.Log = io.Discard
	}
	base := "https://huggingface.co/" + o.Repo + "/resolve/main/"

	var tok *tokenizer.Tokenizer
	var model *qwen
	var err error
	start := time.Now()
	if o.GGUF != "" {
		model, tok, err = loadGGUF(o.GGUF, o.Bits, !o.GPU && !o.Requant, !o.NoCache)
		if err != nil {
			return nil, err
		}
	} else {
		weights, err := fetchWeights(base, o.Data)
		if err != nil {
			return nil, err
		}
		var paths [2]string
		for i, name := range []string{"tokenizer.json", "config.json"} {
			if paths[i], err = fetch(base, o.Data, name); err != nil {
				return nil, err
			}
		}
		if tok, err = tokenizer.Load(paths[0]); err != nil {
			return nil, err
		}
		if model, err = loadQwen(paths[1], weights, o.Bits); err != nil {
			return nil, err
		}
	}
	how := "float32"
	if o.Bits != 0 {
		how = fmt.Sprintf("int%d", o.Bits)
	}
	fmt.Fprintf(o.Log, "loaded %s (%d layers, hidden %d) as %s in %v\n",
		model.cfg.ModelType, model.cfg.Layers, model.cfg.HiddenSize, how, time.Since(start).Round(time.Millisecond))

	// The draft model shares the tokenizer, so it must come from the same
	// family (0.5B drafting for 7B, say).
	var draftM *qwen
	if o.Draft != "" {
		if o.GPU {
			return nil, fmt.Errorf("a draft model requires CPU decoding")
		}
		dw, err := fetchWeights(base, o.Draft)
		if err != nil {
			return nil, err
		}
		dc, err := fetch(base, o.Draft, "config.json")
		if err != nil {
			return nil, err
		}
		start := time.Now()
		if draftM, err = loadQwen(dc, dw, o.Bits); err != nil {
			return nil, err
		}
		fmt.Fprintf(o.Log, "loaded draft (%d layers, hidden %d) in %v\n",
			draftM.cfg.Layers, draftM.cfg.HiddenSize, time.Since(start).Round(time.Millisecond))
	}

	style := model.cfg.ChatStyle
	if style == "" {
		style = model.cfg.ModelType
	}
	system := o.System
	if system == DefaultSystem || system == "" {
		switch {
		case style == "gpt-oss":
			// The harmony system block: identity, reasoning effort, and
			// the channel contract the model was trained on.
			system = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."
		case style == "deepseek":
			// DeepSeek recommends no system prompt for the R1 distills.
			system = ""
		case style != "qwen2" && style != "qwen3":
			system = "You are a helpful assistant."
		default:
			system = DefaultSystem
		}
	}
	tm := templateFor(style, o.Think)
	stopID := func(i int) int {
		if i < len(tm.stops) {
			if id, ok := tok.ID(tm.stops[i]); ok {
				return id
			}
		}
		return -1
	}

	e := &Engine{
		opts: o, model: model, draft: draftM, tok: tok, tm: tm,
		system: system, imEnd: stopID(0), eot: stopID(1),
		rng: rand.New(rand.NewSource(o.Seed)),
	}
	e.nCtx = model.cfg.MaxPos
	if e.nCtx == 0 {
		e.nCtx = 4096
	}
	e.prefill, e.step = model.prefill, model.step
	e.reset = model.reset

	if o.GPU {
		if o.Bits == 0 {
			return nil, fmt.Errorf("GPU decoding requires quantized weights")
		}
		g, err := tensai.OpenGPU(tensai.GPUHighPerformance)
		if err != nil {
			return nil, err
		}
		// The resident KV cache is two buffers per layer sized by the
		// context; a long-context model (Qwen3 declares 40960 positions)
		// would allocate several GB, so clamp to a 2GB total budget and to
		// the device's per-buffer storage limit.
		kvDim := model.cfg.KVHeads * model.headSz
		maxCtx := (2 << 30) / (2 * model.cfg.Layers * kvDim * 4)
		if lim := g.StorageLimit(); lim > 0 {
			if perBuf := int(lim / uint64(kvDim*4)); perBuf < maxCtx {
				maxCtx = perBuf
			}
		}
		if maxCtx < e.nCtx {
			fmt.Fprintf(o.Log, "gpu cache limited to %d positions by device memory\n", maxCtx)
			e.nCtx = maxCtx
		}
		start := time.Now()
		gq, err := newGPUQwen(model, g, e.nCtx)
		if err != nil {
			g.Close()
			return nil, err
		}
		fmt.Fprintf(o.Log, "uploaded to %s in %v\n", g.Name(), time.Since(start).Round(time.Millisecond))
		e.g, e.gq = g, gq
		e.prefill, e.step = gq.prefill, gq.step
		e.reset = func() {
			model.reset()
			gq.gpuLen = 0
		}
	}
	return e, nil
}

// Close releases the GPU residency, if any.
func (e *Engine) Close() {
	if e.g != nil {
		e.g.Close()
	}
}

// feed pushes tokens through the model, extending the KV cache; generate
// then samples until an end token, which is also fed so the cache stays
// aligned with the template for the next turn.
func (e *Engine) feed(ids []int) {
	if len(ids) > 1 {
		if e.draft != nil {
			e.draft.prefill(ids, e.steps)
		}
		e.logits = e.prefill(ids, e.steps)
		e.steps += len(ids)
		return
	}
	for _, id := range ids {
		if e.draft != nil {
			e.draft.step(id, e.steps)
		}
		e.logits = e.step(id, e.steps)
		e.steps++
	}
}

// generate emits up to limit sampled tokens to w, speculatively when a
// draft model is loaded, and reports throughput to the Log writer.
func (e *Engine) generate(w io.Writer, limit int) {
	start := time.Now()
	gen := 0
	if e.draft != nil {
		var stats specStats
		e.logits, e.steps, _, stats = generateSpeculative(e.model, e.draft, e.logits, e.steps,
			limit, e.nCtx, e.opts.SpecK, e.opts.Temp, e.opts.TopP, func(id int) bool {
				return id == e.imEnd || id == e.eot
			}, e.rng, func(id int) bool {
				fmt.Fprint(w, e.tok.Decode([]int{id}))
				gen++
				return true
			})
		fmt.Fprintln(w)
		fmt.Fprintf(e.opts.Log, "(%d tokens, %.1f tok/s, %d/%d drafts accepted)\n",
			gen, float64(gen)/time.Since(start).Seconds(), stats.accepted, stats.proposed)
		return
	}
	for ; gen < limit && e.steps < e.nCtx-1; gen++ {
		next := sample(e.logits, e.opts.Temp, e.opts.TopP, e.rng)
		if next == e.imEnd || next == e.eot {
			e.feed([]int{next})
			break
		}
		fmt.Fprint(w, e.tok.Decode([]int{next}))
		e.feed([]int{next})
	}
	fmt.Fprintln(w)
	fmt.Fprintf(e.opts.Log, "(%d tokens, %.1f tok/s)\n",
		gen, float64(gen)/time.Since(start).Seconds())
}

// Generate runs one completion: the prompt goes through the model's chat
// template (or verbatim when raw), and up to n sampled tokens stream to w.
func (e *Engine) Generate(w io.Writer, prompt string, raw bool, n int) {
	text := prompt
	if !raw {
		if e.tm.foldSystem {
			text = e.tm.bos + e.tm.userOpen + e.system + "\n\n" + prompt + e.tm.userClose + e.tm.asstOpen
		} else {
			text = e.tm.bos + e.tm.sysOpen + e.system + e.tm.sysClose +
				e.tm.userOpen + prompt + e.tm.userClose + e.tm.asstOpen
		}
	}
	ids := e.tok.Encode(text)
	fmt.Fprintf(e.opts.Log, "prompt: %d tokens\n", len(ids))
	start := time.Now()
	e.feed(ids)
	fmt.Fprintf(e.opts.Log, "prefill: %v\n", time.Since(start).Round(time.Millisecond))
	e.generate(w, n)
	fmt.Fprintf(e.opts.Log, "%d tokens total in %v\n",
		e.steps, time.Since(start).Round(time.Millisecond))
}

// Chat runs an interactive multi-turn loop: one line of in per turn, the
// KV cache carrying the conversation, until EOF or an empty line.
func (e *Engine) Chat(in io.Reader, w io.Writer, n int) {
	pre := e.tm.bos
	if !e.tm.foldSystem {
		pre += e.tm.sysOpen + e.system + e.tm.sysClose
	}
	if pre != "" {
		e.feed(e.tok.Encode(pre))
	}
	fmt.Fprintln(e.opts.Log, "chat mode: type a message, empty line or Ctrl-D to quit")
	sc := bufio.NewScanner(in)
	first := true
	for {
		fmt.Fprint(w, "> ")
		if !sc.Scan() || strings.TrimSpace(sc.Text()) == "" {
			break
		}
		text := sc.Text()
		if e.tm.foldSystem && first {
			text = e.system + "\n\n" + text
		}
		first = false
		e.feed(e.tok.Encode(e.tm.userOpen + text + e.tm.userClose + e.tm.asstOpen))
		e.generate(w, n)
		e.feed(e.tok.Encode("\n"))
		if e.steps >= e.nCtx-64 {
			fmt.Fprintln(e.opts.Log, "context window exhausted")
			break
		}
	}
}

// Serve blocks on an OpenAI-compatible /v1/chat/completions server.
func (e *Engine) Serve(addr string) error {
	s := &server{
		model: e.model, tok: e.tok, system: e.system, nCtx: e.nCtx,
		temp: e.opts.Temp, topP: e.opts.TopP, imEnd: e.imEnd, eot: e.eot,
		tm: e.tm, prefill: e.prefill, step: e.step, reset: e.reset,
		draft: e.draft, specK: e.opts.SpecK,
	}
	return s.listen(addr)
}

func fetch(base, dir, name string) (string, error) {
	path := filepath.Join(dir, name)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	resp, err := http.Get(base + name)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("downloading %s: %s", name, resp.Status)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(f, &progressReader{r: resp.Body, name: name, total: resp.ContentLength}); err != nil {
		f.Close()
		os.Remove(tmp)
		return "", err
	}
	fmt.Fprintln(os.Stderr)
	if err := f.Close(); err != nil {
		return "", err
	}
	return path, os.Rename(tmp, path)
}

// progressReader reports download progress on stderr, at most twice a
// second so slow terminals don't throttle the transfer.
type progressReader struct {
	r     io.Reader
	name  string
	total int64
	done  int64
	last  time.Time
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.done += int64(n)
	if now := time.Now(); now.Sub(p.last) > 500*time.Millisecond || err != nil {
		p.last = now
		if p.total > 0 {
			fmt.Fprintf(os.Stderr, "\rdownloading %s... %d%% (%d/%d MB)",
				p.name, p.done*100/p.total, p.done>>20, p.total>>20)
		} else {
			fmt.Fprintf(os.Stderr, "\rdownloading %s... %d MB", p.name, p.done>>20)
		}
	}
	return n, err
}

// fetchWeights returns the checkpoint path: a plain model.safetensors, or
// for sharded models the index file after downloading every shard.
func fetchWeights(base, dir string) (string, error) {
	if p := filepath.Join(dir, "model.safetensors"); exists(p) {
		return p, nil
	}
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if !exists(idx) {
		// Try the single file first; fall back to the sharded index.
		if p, err := fetch(base, dir, "model.safetensors"); err == nil {
			return p, nil
		}
		if _, err := fetch(base, dir, "model.safetensors.index.json"); err != nil {
			return "", err
		}
	}
	raw, err := os.ReadFile(idx)
	if err != nil {
		return "", err
	}
	var parsed struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", err
	}
	shards := map[string]bool{}
	for _, s := range parsed.WeightMap {
		shards[s] = true
	}
	names := make([]string, 0, len(shards))
	for s := range shards {
		names = append(names, s)
	}
	sort.Strings(names)
	for _, s := range names {
		if _, err := fetch(base, dir, s); err != nil {
			return "", err
		}
	}
	return idx, nil
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

type sampleCandidate struct {
	id int
	p  float64
}

type sampleScratch struct {
	ps      []float64
	buckets []uint16
	cands   []sampleCandidate
}

var sampleScratchPool sync.Pool

// sample picks the next token: greedy at temp <= 0, otherwise
// temperature softmax restricted to the nucleus — the smallest
// probability-sorted prefix whose mass reaches topP — so the long
// low-probability tail (where repetition loops and derailments live)
// never gets a lottery ticket. topP >= 1 keeps the whole distribution.
//
// Sorting all 150k vocabulary entries per token would dominate sampling,
// so the nucleus is found through a histogram over probability exponents:
// only the tokens above the crossing bucket — usually a few hundred —
// ever get sorted, and the result is exactly the sorted-prefix nucleus.
func sample(logits []float32, temp, topP float64, rng *rand.Rand) int {
	if temp <= 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return best
	}
	scratch, _ := sampleScratchPool.Get().(*sampleScratch)
	if scratch == nil {
		scratch = &sampleScratch{}
	}
	if cap(scratch.ps) < len(logits) {
		scratch.ps = make([]float64, len(logits))
	}
	ps := scratch.ps[:len(logits)]
	if cap(scratch.buckets) < len(logits) {
		scratch.buckets = make([]uint16, len(logits))
	}
	buckets := scratch.buckets[:len(logits)]
	cands := scratch.cands[:0]
	defer func() {
		scratch.cands = cands[:0]
		sampleScratchPool.Put(scratch)
	}()
	maxl := logits[0]
	for _, v := range logits {
		if v > maxl {
			maxl = v
		}
	}
	// Bucket the probability mass by exponent (p in (0, 1], so Ilogb is
	// 0, -1, -2, ...); the crossing search below finds the bucket where
	// the running mass reaches the nucleus target.
	const nb = 1100
	bucket := func(p float64) int {
		b := -math.Ilogb(p)
		if b < 0 {
			b = 0
		} else if b >= nb {
			b = nb - 1
		}
		return b
	}
	// Exp over the vocabulary dominates sampling, so it fans out across
	// CPUs — each element (and its bucket) is independent — while the
	// mass accumulation stays a serial in-order pass, keeping the result
	// bit-identical to the fused serial loop.
	needBuckets := topP < 1
	expRange := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			p := math.Exp(float64(logits[i]-maxl) / temp)
			ps[i] = p
			if needBuckets && p > 0 {
				buckets[i] = uint16(bucket(p))
			}
		}
	}
	if workers := min(runtime.NumCPU(), len(logits)/4096); workers > 1 {
		var wg sync.WaitGroup
		chunk := (len(logits) + workers - 1) / workers
		for lo := chunk; lo < len(logits); lo += chunk {
			hi := min(lo+chunk, len(logits))
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				expRange(lo, hi)
			}(lo, hi)
		}
		expRange(0, chunk)
		wg.Wait()
	} else {
		expRange(0, len(logits))
	}
	var sum float64
	for _, p := range ps {
		sum += p
	}

	if topP >= 1 {
		// The full distribution needs no order at all — one CDF walk.
		r := rng.Float64() * sum
		for i, p := range ps {
			r -= p
			if r <= 0 {
				return i
			}
		}
		return len(ps) - 1
	}

	// Sum the mass per precomputed bucket and find the bucket where the
	// running mass crosses the nucleus target: every nucleus member has
	// p in a bucket at or above the crossing.
	var bsum [nb]float64
	for i, p := range ps {
		if p > 0 {
			bsum[buckets[i]] += p
		}
	}
	target := topP * sum
	var mass float64
	cut := nb - 1
	for b := 0; b < nb; b++ {
		mass += bsum[b]
		if mass >= target {
			cut = b
			break
		}
	}
	for i, p := range ps {
		if p > 0 && int(buckets[i]) <= cut {
			cands = append(cands, sampleCandidate{i, p})
		}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].p > cands[b].p })
	mass = 0
	n := len(cands)
	for i, c := range cands {
		mass += c.p
		if mass >= target {
			n = i + 1
			break
		}
	}
	cands = cands[:n]
	r := rng.Float64() * mass
	for _, c := range cands {
		r -= c.p
		if r <= 0 {
			return c.id
		}
	}
	return cands[0].id
}

// tmpl is the chat template family a model speaks: ChatML for the Qwen
// and SmolLM lines, Gemma's turn markers for gemma3 (which has no system
// role — the system prompt folds into the first user turn).
type tmpl struct {
	bos                 string
	sysOpen, sysClose   string
	userOpen, userClose string
	asstOpen, asstClose string
	foldSystem          bool
	stops               []string
}

func templateFor(modelType string, think bool) tmpl {
	if modelType == "gpt-oss" {
		// The harmony format: role blocks between <|start|> and <|end|>,
		// the assistant answering in channels (analysis for reasoning,
		// final for the reply) and finishing with <|return|>.
		return tmpl{
			sysOpen: "<|start|>system<|message|>", sysClose: "<|end|>",
			userOpen: "<|start|>user<|message|>", userClose: "<|end|>",
			asstOpen: "<|start|>assistant", asstClose: "<|return|>",
			stops: []string{"<|return|>"},
		}
	}
	if modelType == "mistral" {
		// Mistral instruct: user turns bracketed by [INST], no system
		// role (it folds into the first user turn), </s> closing each
		// assistant reply.
		return tmpl{
			bos:      "<s>",
			userOpen: "[INST] ", userClose: " [/INST]",
			asstClose:  "</s>",
			foldSystem: true,
			stops:      []string{"</s>"},
		}
	}
	if modelType == "deepseek" {
		// DeepSeek R1 distills: the system prompt (rarely used — DeepSeek
		// recommends none) sits bare after BOS, user turns have no closing
		// marker, and the model opens its answer with a <think> block on
		// its own.
		return tmpl{
			bos:      "<｜begin▁of▁sentence｜>",
			userOpen: "<｜User｜>",
			asstOpen: "<｜Assistant｜>", asstClose: "<｜end▁of▁sentence｜>",
			stops: []string{"<｜end▁of▁sentence｜>"},
		}
	}
	if modelType == "phi3" {
		// Phi-3's template has no system role either; its official
		// guidance folds the system prompt into the first user turn.
		return tmpl{
			bos:      "<s>",
			userOpen: "<|user|>\n", userClose: "<|end|>\n",
			asstOpen: "<|assistant|>\n", asstClose: "<|end|>\n",
			foldSystem: true,
			stops:      []string{"<|end|>", "<|endoftext|>"},
		}
	}
	if modelType == "gemma3" {
		return tmpl{
			bos:      "<bos>",
			userOpen: "<start_of_turn>user\n", userClose: "<end_of_turn>\n",
			asstOpen: "<start_of_turn>model\n", asstClose: "<end_of_turn>\n",
			foldSystem: true,
			stops:      []string{"<end_of_turn>"},
		}
	}
	t := tmpl{
		sysOpen: "<|im_start|>system\n", sysClose: "<|im_end|>\n",
		userOpen: "<|im_start|>user\n", userClose: "<|im_end|>\n",
		asstOpen: "<|im_start|>assistant\n", asstClose: "<|im_end|>\n",
		stops: []string{"<|im_end|>", "<|endoftext|>"},
	}
	// Qwen3 and SmolLM3 disable their thinking mode by opening the
	// assistant turn with an empty think block; -think leaves it open.
	if (modelType == "qwen3" || modelType == "smollm3") && !think {
		t.asstOpen += "<think>\n\n</think>\n\n"
	}
	return t
}

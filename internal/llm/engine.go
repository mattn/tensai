// Package llm wires tensai's kernels into a runnable language model:
// checkpoint download and loading, chat templates, sampling, generation
// (plain and speculative), the GPU decode path, and the OpenAI-compatible
// server. cmd/tensai is a thin flag parser over Engine.
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
	"os/user"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/tokenizer"
)

// DefaultRepo is the checkpoint Options.Repo defaults to.
const DefaultRepo = "Qwen/Qwen2.5-0.5B-Instruct"

// DefaultSystem is the system prompt Qwen-family models get; other
// families swap it for a neutral one (see Open), exactly when the caller
// left it at this default.
const DefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

// CacheRoot is the directory model caches live under
// (~/.cache/tensai on Linux).
func CacheRoot() string {
	if dir, err := os.UserCacheDir(); err == nil {
		return filepath.Join(dir, "tensai")
	}
	// UserCacheDir only fails when the environment does not say where
	// the home directory is. The account itself still knows, and a
	// multi-gigabyte download belongs there rather than in whatever
	// directory the command happened to start in.
	if u, err := user.Current(); err == nil && u.HomeDir != "" {
		return filepath.Join(u.HomeDir, ".cache", "tensai")
	}
	return "tensai-data"
}

// DefaultDataDir is where a repo's files live when Options.Data is
// empty: a per-repo directory under CacheRoot.
func DefaultDataDir(repo string) string {
	if repo == "" {
		repo = DefaultRepo
	}
	return filepath.Join(CacheRoot(), path.Base(repo))
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
	g       *gpu.Device
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
	// An empty Repo means the caller named a model already on disk, so
	// nothing here may reach the network: the default repo that fills in
	// below addresses a different checkpoint, and a file fetched from it
	// would be another model's.
	local := o.Repo == ""
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
		model.cfg.ChatTemplate = chatTemplate(base, o.Data, local)
		if !local {
			recordOrigin(o.Data, o.Repo)
		}
	}
	// A delta layer's state is not a KV cache: nothing in it can be
	// indexed by position, so speculative decoding cannot roll it back,
	// and the GPU path has no kernel for it. Both say so rather than
	// producing nonsense.
	if model.hasDelta() {
		if o.GPU {
			return nil, fmt.Errorf("%s runs its linear-attention layers on the CPU only", model.cfg.ModelType)
		}
		if o.Draft != "" {
			return nil, fmt.Errorf("%s keeps a recurrent state that cannot be rolled back, so -draft cannot verify against it", model.cfg.ModelType)
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
	// The family table says which markers the model speaks, and only a
	// convention tensai renders can be offered at all -- so the
	// checkpoint's own template only ever takes the capability away,
	// never grants one the server could not write. A template that never
	// branches on tools is the checkpoint saying it was not prepared for
	// them, whatever its family does.
	if tpl := model.cfg.ChatTemplate; tpl != "" && !templateTakesTools(tpl) {
		tm.toolCalls = ""
	}
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
		g, err := gpu.Open(gpu.HighPerformance)
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
		gq, err := newGPUQwen(model, g, e.nCtx, o.Log)
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

// Reset drops the KV cache and rewinds the position counter, so the next
// Generate starts from an empty context instead of continuing the last
// one. Benchmarks use it to repeat the same prompt.
func (e *Engine) Reset() {
	e.reset()
	if e.draft != nil {
		e.draft.reset()
	}
	e.steps = 0
	e.logits = nil
}

// GPUName reports the adapter the engine is running on, empty when
// decoding on the CPU. Backend names the binding generation the binary
// was built with.
func (e *Engine) GPUName() string {
	if e.g == nil {
		return ""
	}
	return e.g.Name()
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
// draft model is loaded, and reports throughput to the Log writer. It
// returns the token count and why generation ended.
func (e *Engine) generate(w io.Writer, limit int) (int, string) {
	start := time.Now()
	gen := 0
	if e.draft != nil {
		var stats specStats
		var finish string
		e.logits, e.steps, finish, stats = generateSpeculative(e.model, e.draft, e.logits, e.steps,
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
		return gen, finish
	}
	finish := "length"
	for ; gen < limit && e.steps < e.nCtx-1; gen++ {
		next := sample(e.logits, e.opts.Temp, e.opts.TopP, e.rng)
		if next == e.imEnd || next == e.eot {
			e.feed([]int{next})
			finish = "stop"
			break
		}
		fmt.Fprint(w, e.tok.Decode([]int{next}))
		e.feed([]int{next})
	}
	fmt.Fprintln(w)
	fmt.Fprintf(e.opts.Log, "(%d tokens, %.1f tok/s)\n",
		gen, float64(gen)/time.Since(start).Seconds())
	return gen, finish
}

// RunResult reports what one Generate call did.
type RunResult struct {
	PromptTokens     int
	CompletionTokens int
	Finish           string // "stop" or "length"
	Prefill          time.Duration
	Total            time.Duration
}

// Generate runs one completion: the prompt goes through the model's chat
// template (or verbatim when raw), and up to n sampled tokens stream to w.
func (e *Engine) Generate(w io.Writer, prompt string, raw bool, n int) RunResult {
	text := prompt
	if !raw {
		if e.tm.foldSystem {
			text = e.tm.bos + e.tm.userOpen + e.system + "\n\n" + prompt + e.tm.userClose +
				e.tm.asstOpen + e.tm.asstPrefill
		} else {
			text = e.tm.bos + e.tm.sysOpen + e.system + e.tm.sysClose +
				e.tm.userOpen + prompt + e.tm.userClose + e.tm.asstOpen + e.tm.asstPrefill
		}
	}
	ids := e.tok.Encode(text)
	fmt.Fprintf(e.opts.Log, "prompt: %d tokens\n", len(ids))
	start := time.Now()
	e.feed(ids)
	prefill := time.Since(start)
	fmt.Fprintf(e.opts.Log, "prefill: %v\n", prefill.Round(time.Millisecond))
	gen, finish := e.generate(w, n)
	fmt.Fprintf(e.opts.Log, "%d tokens total in %v\n",
		e.steps, time.Since(start).Round(time.Millisecond))
	return RunResult{
		PromptTokens: len(ids), CompletionTokens: gen, Finish: finish,
		Prefill: prefill, Total: time.Since(start),
	}
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
		e.feed(e.tok.Encode(e.tm.userOpen + text + e.tm.userClose + e.tm.asstOpen + e.tm.asstPrefill))
		e.generate(w, n)
		e.feed(e.tok.Encode("\n"))
		if e.steps >= e.nCtx-64 {
			fmt.Fprintln(e.opts.Log, "context window exhausted")
			break
		}
	}
}

// Serve blocks on an OpenAI-compatible /v1/chat/completions server,
// with a demo page on GET /. A non-empty apiKey guards the /v1 routes
// behind an Authorization: Bearer header.
func (e *Engine) Serve(addr, apiKey string) error {
	s := &server{
		apiKey: apiKey,
		model:  e.model, tok: e.tok, system: e.system, nCtx: e.nCtx,
		temp: e.opts.Temp, topP: e.opts.TopP, imEnd: e.imEnd, eot: e.eot,
		tm: e.tm, prefill: e.prefill, step: e.step, reset: e.reset,
		draft: e.draft, specK: e.opts.SpecK,
	}
	// An agent resends its system prompt and its tool definitions with
	// every question; caching them is the difference between paying for
	// thousands of tokens each time and paying once. The GPU path keeps
	// its own resident cache, which this does not reach.
	s.cache.enabled = !e.opts.GPU
	s.cache.hasDelta = e.model.hasDelta()
	return s.listen(addr)
}

// fetchAttempts is how many times a download is tried before giving up.
// The pause between them widens, so a server that is briefly unhappy gets
// a little room without a wedged one holding the load up for long.
const fetchAttempts = 5

// fetch downloads name into dir unless it is already there. A partial
// download stays on disk as name.tmp and the next attempt resumes it with
// a Range request rather than starting a multi-gigabyte file over. The
// resume is guarded by If-Range against the recorded ETag, so a file that
// changed upstream restarts instead of splicing two versions together;
// without a recorded tag the partial is discarded rather than trusted.
func fetch(base, dir, name string) (string, error) {
	path := filepath.Join(dir, name)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	tmp := path + ".tmp"
	var err error
	for attempt := 0; attempt < fetchAttempts; attempt++ {
		if attempt > 0 {
			pause := time.Duration(1<<uint(attempt-1)) * time.Second
			fmt.Fprintf(os.Stderr, "\n%s: %v; retrying in %v\n", name, err, pause)
			time.Sleep(pause)
		}
		var retry bool
		retry, err = fetchOnce(base, dir, name, tmp)
		if err == nil {
			fmt.Fprintln(os.Stderr)
			os.Remove(tmp + ".etag")
			return path, os.Rename(tmp, path)
		}
		if !retry {
			return "", err
		}
	}
	return "", fmt.Errorf("downloading %s: %w (after %d attempts)", name, err, fetchAttempts)
}

// fetchOnce makes one attempt at finishing tmp, reporting whether the
// failure is the kind another attempt could get past. A partial file is
// left where it is either way: it is what the next attempt resumes from.
func fetchOnce(base, dir, name, tmp string) (retry bool, err error) {
	req, err := http.NewRequest(http.MethodGet, base+name, nil)
	if err != nil {
		return false, err
	}
	var off int64
	tag, _ := os.ReadFile(tmp + ".etag")
	if fi, err := os.Stat(tmp); err == nil && fi.Size() > 0 && len(tag) > 0 {
		off = fi.Size()
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", off))
		req.Header.Set("If-Range", string(tag))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return true, err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusPartialContent:
		// The server kept our place; anything else means starting over.
	case http.StatusOK:
		off = 0
	case http.StatusRequestedRangeNotSatisfiable:
		// The partial is at least as long as the file: it is not a prefix
		// of anything the server will send, so drop it and try again.
		os.Remove(tmp)
		os.Remove(tmp + ".etag")
		return true, fmt.Errorf("downloading %s: %s", name, resp.Status)
	default:
		// 5xx and 429 may pass; a 404 or a 403 will not.
		retry = resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests
		return retry, fmt.Errorf("downloading %s: %s", name, resp.Status)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if off > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		if etag := resp.Header.Get("ETag"); etag != "" {
			os.WriteFile(tmp+".etag", []byte(etag), 0o644)
		} else {
			os.Remove(tmp + ".etag")
		}
	}
	f, err := os.OpenFile(tmp, flags, 0o644)
	if err != nil {
		return false, err
	}
	total := resp.ContentLength
	if total >= 0 {
		total += off
	}
	_, err = io.Copy(f, &progressReader{r: resp.Body, name: name, total: total, done: off})
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		// Whatever landed stays: the next attempt picks up from there.
		return true, err
	}
	return false, nil
}

// FetchGGUF downloads one GGUF file from a Hugging Face repo — a
// reference of the form org/repo/file.gguf — into the cache root and
// returns the local path. The file keeps its own name and lands beside
// the other cached ggufs, so "tensai models" lists it and a later bare
// file name finds it. An already-downloaded file returns immediately,
// and a partial one resumes.
func FetchGGUF(ref string) (string, error) {
	parts := strings.Split(ref, "/")
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" ||
		!strings.HasSuffix(parts[2], ".gguf") || parts[2] == ".gguf" {
		return "", fmt.Errorf("gguf reference %q is not org/repo/file.gguf", ref)
	}
	base := "https://huggingface.co/" + parts[0] + "/" + parts[1] + "/resolve/main/"
	p, err := fetch(base, CacheRoot(), parts[2])
	if err == nil {
		recordGGUFOrigin(p, parts[0]+"/"+parts[1])
	}
	return p, err
}

// recordGGUFOrigin is recordOrigin for a single cached gguf file, which
// has no directory of its own: the repo lands in a sidecar next to it.
func recordGGUFOrigin(path, repo string) {
	if path == "" || !strings.Contains(repo, "/") {
		return
	}
	p := path + originSidecar
	if b, err := os.ReadFile(p); err == nil && string(b) == repo {
		return
	}
	os.WriteFile(p, []byte(repo), 0o644)
}

// originSidecar suffixes a cached gguf's origin marker.
const originSidecar = ".tensai-origin"

// GGUFOrigin returns the org/repo a cached gguf was downloaded from, or
// "" when no valid record exists. The full reference the listing prints
// — org/repo/file.gguf — downloads the same file on another machine.
func GGUFOrigin(path string) string {
	b, err := os.ReadFile(path + originSidecar)
	if err != nil {
		return ""
	}
	repo := strings.TrimSpace(string(b))
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" ||
		strings.ContainsAny(repo, `\`) || strings.Contains(repo, "..") {
		return ""
	}
	return repo
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

// originFile names the repo a cached model was downloaded from. The
// cache directory is named after the repo's last element, which is what
// makes a listed name typeable, but that drops the organization -- and
// with it the only way to fetch the same checkpoint on another machine.
const originFile = ".tensai-origin"

// recordOrigin remembers where a download came from, so a listing can
// name something the user can act on somewhere else. Failing to write it
// costs the listing an organization, not the model, so the error is not
// worth failing a load over.
func recordOrigin(dir, repo string) {
	if dir == "" || !strings.Contains(repo, "/") {
		return
	}
	p := filepath.Join(dir, originFile)
	if b, err := os.ReadFile(p); err == nil && string(b) == repo {
		return
	}
	os.WriteFile(p, []byte(repo), 0o644)
}

// Origin returns the repo a cached model was downloaded from, or "" for
// one that predates the record or was never downloaded at all.
func Origin(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, originFile))
	if err != nil {
		return ""
	}
	repo := strings.TrimSpace(string(b))
	// Only a plain org/name is worth handing back: it is going to be
	// printed and then typed at -model.
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" ||
		strings.ContainsAny(repo, `\`) || strings.Contains(repo, "..") {
		return ""
	}
	return repo
}

// chatTemplate returns the model's own Jinja chat template, or "" when
// the checkpoint does not ship one. Older checkpoints keep it in
// tokenizer_config.json; newer ones moved it to a file of its own and
// leave the JSON field empty. Neither file is needed to run the model,
// so a miss costs nothing but the answer -- and for a model the caller
// named on disk the files are read where they are, never downloaded,
// since the repo in hand is not the one they came from.
func chatTemplate(base, dir string, local bool) string {
	get := func(name string) (string, bool) {
		if local {
			p := filepath.Join(dir, name)
			if _, err := os.Stat(p); err != nil {
				return "", false
			}
			raw, err := os.ReadFile(p)
			return string(raw), err == nil
		}
		p, err := fetch(base, dir, name)
		if err != nil {
			return "", false
		}
		raw, err := os.ReadFile(p)
		return string(raw), err == nil
	}
	if raw, ok := get("tokenizer_config.json"); ok {
		var cfg struct {
			ChatTemplate string `json:"chat_template"`
		}
		if json.Unmarshal([]byte(raw), &cfg) == nil && cfg.ChatTemplate != "" {
			return cfg.ChatTemplate
		}
	}
	if raw, ok := get("chat_template.jinja"); ok {
		return raw
	}
	return ""
}

// templateTakesTools reports whether a chat template branches on the
// tool definitions it may be handed. A template prepared for calling has
// to name that variable to render the signatures at all, so its absence
// is the checkpoint saying it was never prepared for this. The match is
// on the Jinja idioms that read the variable rather than the bare word,
// which also appears in templates that only mention tools in prose.
func templateTakesTools(tpl string) bool {
	flat := strings.ToLower(strings.Join(strings.Fields(tpl), " "))
	for _, idiom := range []string{
		"if tools",         // Qwen, Hermes, Mistral
		"if tools is",      // Llama 3.1 ("is not none")
		"in tools",         // "for tool in tools"
		"tools is defined", // Gemma-style guards
		"tools %}",         // bare interpolation of the list
		"tools|",           // filtered, e.g. "tools|tojson"
		"tools |",
	} {
		if strings.Contains(flat, idiom) {
			return true
		}
	}
	return false
}

// tmpl is the chat template family a model speaks: ChatML for the Qwen
// and SmolLM lines, Gemma's turn markers for gemma3 (which has no system
// role — the system prompt folds into the first user turn).
type tmpl struct {
	bos                 string
	sysOpen, sysClose   string
	userOpen, userClose string
	asstOpen, asstClose string
	// asstPrefill is what the template writes into an assistant turn
	// before the model speaks -- the empty think block a Qwen3 opens
	// with when thinking is off. It belongs to the turn being generated,
	// not to every turn in the history.
	asstPrefill string
	foldSystem  bool
	stops       []string
	// toolCalls names the function-calling convention the family was
	// trained on, empty when it has none. "hermes" is the one the ChatML
	// families speak: tool signatures inside a <tools> block appended to
	// the system turn, calls emitted as <tool_call> JSON, and results fed
	// back as <tool_response> inside a user turn.
	toolCalls string
	// reasonOpen and reasonClose bracket the block a thinking model fills
	// before it answers, empty for families that do not think out loud.
	// What is between them is the model reasoning, not its reply, and the
	// API keeps the two apart.
	reasonOpen, reasonClose string
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
			stops:       []string{"<｜end▁of▁sentence｜>"},
			reasonOpen:  "<think>",
			reasonClose: "</think>",
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
		stops:     []string{"<|im_end|>", "<|endoftext|>"},
		toolCalls: "hermes",
	}
	// Qwen3 and SmolLM3 disable their thinking mode by opening the
	// assistant turn with an empty think block; -think leaves it open.
	if modelType == "qwen3" || modelType == "qwen3_5" || modelType == "smollm3" {
		if think {
			t.reasonOpen, t.reasonClose = "<think>", "</think>"
		} else {
			t.asstPrefill = "<think>\n\n</think>\n\n"
		}
	}
	// Qwen3.5 left the Hermes convention its predecessors speak: its
	// tools block is a system turn of its own, and a call is nested XML
	// rather than JSON.
	if modelType == "qwen3_5" {
		t.toolCalls = "qwen3xml"
	}
	return t
}

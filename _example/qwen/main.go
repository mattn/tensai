// Command qwen runs the published Qwen2.5-0.5B-Instruct checkpoint in pure
// Go: BF16 weights load through encoding/safetensors, text goes through
// the tokenizer package, and the Qwen2 architecture — RMSNorm, rotary
// embeddings, grouped-query attention, SwiGLU — decodes with a KV cache.
// Build with GOEXPERIMENT=simd and pass -q8 for int8 decode weights.
//
// On first run it downloads the checkpoint (~1GB), tokenizer, and config
// from Hugging Face into -data. The prompt is wrapped in the ChatML
// template Qwen was instruction-tuned on; -raw skips the template for
// plain completion:
//
//	GOEXPERIMENT=simd go run ./_example/qwen -q8 -prompt "What is the capital of France?"
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime/pprof"
	"sort"
	"strings"
	"time"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/tokenizer"
)

const defaultRepo = "Qwen/Qwen2.5-0.5B-Instruct"

// hfBase is derived from the -repo flag before any download starts.
var hfBase = "https://huggingface.co/" + defaultRepo + "/resolve/main/"

func fetch(dir, name string) (string, error) {
	path := filepath.Join(dir, name)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	fmt.Fprintf(os.Stderr, "downloading %s...\n", name)
	resp, err := http.Get(hfBase + name)
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
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return "", err
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	return path, os.Rename(tmp, path)
}

// fetchWeights returns the checkpoint path: a plain model.safetensors, or
// for sharded models the index file after downloading every shard.
func fetchWeights(dir string) (string, error) {
	if p := filepath.Join(dir, "model.safetensors"); exists(p) {
		return p, nil
	}
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if !exists(idx) {
		// Try the single file first; fall back to the sharded index.
		if p, err := fetch(dir, "model.safetensors"); err == nil {
			return p, nil
		}
		if _, err := fetch(dir, "model.safetensors.index.json"); err != nil {
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
		if _, err := fetch(dir, s); err != nil {
			return "", err
		}
	}
	return idx, nil
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

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
	maxl := logits[0]
	for _, v := range logits {
		if v > maxl {
			maxl = v
		}
	}
	ps := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		p := math.Exp(float64(v-maxl) / temp)
		ps[i] = p
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

	// Bucket the probability mass by exponent (p in (0, 1], so Ilogb is
	// 0, -1, -2, ...) and find the bucket where the running mass crosses
	// the nucleus target: every nucleus member has p in a bucket at or
	// above the crossing.
	const nb = 1100
	var bsum [nb]float64
	bucket := func(p float64) int {
		b := -math.Ilogb(p)
		if b < 0 {
			b = 0
		} else if b >= nb {
			b = nb - 1
		}
		return b
	}
	for _, p := range ps {
		if p > 0 {
			bsum[bucket(p)] += p
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
	type cand struct {
		id int
		p  float64
	}
	var cands []cand
	for i, p := range ps {
		if p > 0 && bucket(p) <= cut {
			cands = append(cands, cand{i, p})
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

func main() {
	dataDir := flag.String("data", "_example/qwen/data", "directory for the downloaded model files")
	repo := flag.String("repo", defaultRepo, "Hugging Face repo to download missing model files from")
	prompt := flag.String("prompt", "What is the capital of France?", "user message (or raw prompt with -raw)")
	system := flag.String("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "system message for the chat template")
	raw := flag.Bool("raw", false, "skip the chat template, complete the prompt as-is")
	n := flag.Int("n", 256, "max tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 = greedy")
	topP := flag.Float64("topp", 0.9, "nucleus sampling: keep the smallest set of tokens with this much probability mass (1 disables)")
	seed := flag.Int64("seed", 1, "sampling seed for -temp > 0")
	q8 := flag.Bool("q8", false, "decode against int8-quantized weights")
	q4 := flag.Bool("q4", false, "decode against int4-quantized weights (group-wise)")
	chat := flag.Bool("chat", false, "interactive multi-turn chat on stdin (the KV cache carries the conversation)")
	gpu := flag.Bool("gpu", false, "decode on the GPU (requires -q8 or -q4 and a wgpu build tag)")
	cpuprofile := flag.String("cpuprofile", "", "write a CPU profile of generation to this file")
	ggufPath := flag.String("gguf", "", "load model and tokenizer from a single .gguf file instead of -data/-repo")
	serveAddr := flag.String("serve", "", "serve an OpenAI-compatible /v1/chat/completions API on this address (e.g. :8080)")
	think := flag.Bool("think", false, "let Qwen3 models reason in a <think> block before answering")
	draftDir := flag.String("draft", "", "data directory of a smaller draft model: speculative decoding (greedy only)")
	specK := flag.Int("spec", 3, "draft tokens proposed per speculative step (3 fills one 4-row verification block)")
	flag.Parse()
	hfBase = "https://huggingface.co/" + *repo + "/resolve/main/"

	bits := 0
	if *q8 {
		bits = 8
	}
	if *q4 {
		bits = 4
	}

	var tok *tokenizer.Tokenizer
	var model *qwen
	var err error
	start := time.Now()
	if *ggufPath != "" {
		model, tok, err = loadGGUF(*ggufPath, bits, !*gpu)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	} else {
		weights, err := fetchWeights(*dataDir)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		var paths [2]string
		for i, name := range []string{"tokenizer.json", "config.json"} {
			p, err := fetch(*dataDir, name)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			paths[i] = p
		}
		if tok, err = tokenizer.Load(paths[0]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if model, err = loadQwen(paths[1], weights, bits); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	how := "float32"
	if bits != 0 {
		how = fmt.Sprintf("int%d", bits)
	}
	fmt.Fprintf(os.Stderr, "loaded %s (%d layers, hidden %d) as %s in %v\n",
		model.cfg.ModelType, model.cfg.Layers, model.cfg.HiddenSize, how, time.Since(start).Round(time.Millisecond))

	// The draft model shares the tokenizer, so it must come from the same
	// family (0.5B drafting for 7B, say). Speculation verifies greedily.
	var draftM *qwen
	if *draftDir != "" {
		if *temp > 0 || *gpu || *serveAddr != "" {
			fmt.Fprintln(os.Stderr, "-draft requires greedy CPU decoding (-temp 0, no -gpu/-serve)")
			os.Exit(1)
		}
		dw, err := fetchWeights(*draftDir)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		dc, err := fetch(*draftDir, "config.json")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		start := time.Now()
		if draftM, err = loadQwen(dc, dw, bits); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "loaded draft (%d layers, hidden %d) in %v\n",
			draftM.cfg.Layers, draftM.cfg.HiddenSize, time.Since(start).Round(time.Millisecond))
	}

	imEnd, _ := tok.ID("<|im_end|>")
	eot, _ := tok.ID("<|endoftext|>")
	rng := rand.New(rand.NewSource(*seed))

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	nCtx := model.cfg.MaxPos
	if nCtx == 0 {
		nCtx = 4096
	}

	// With -gpu the transformer blocks decode on the device; the prompt
	// still prefills on the CPU, and syncCache carries it over below.
	var gq *gpuQwen
	if *gpu {
		if bits == 0 {
			fmt.Fprintln(os.Stderr, "-gpu requires -q8 or -q4")
			os.Exit(1)
		}
		g, err := tensai.OpenGPU(tensai.GPUHighPerformance)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer g.Close()
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
		if maxCtx < nCtx {
			fmt.Fprintf(os.Stderr, "gpu cache limited to %d positions by device memory\n", maxCtx)
			nCtx = maxCtx
		}
		start := time.Now()
		if gq, err = newGPUQwen(model, g, nCtx); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "uploaded to %s in %v\n", g.Name(), time.Since(start).Round(time.Millisecond))
	}

	// feed pushes tokens through the model, extending the KV cache;
	// generate then samples until an end token, which is also fed so the
	// cache stays aligned with the template for the next turn.
	// Qwen3's template disables its thinking mode by opening the assistant
	// turn with an empty think block; -think leaves the block open so the
	// model reasons first. Other models just open the turn.
	asst := "<|im_start|>assistant\n"
	if (model.cfg.ModelType == "qwen3" || model.cfg.ModelType == "smollm3") && !*think {
		asst += "<think>\n\n</think>\n\n"
	}

	stepFn := model.step
	prefillFn := model.prefill
	reset := func() {
		model.reset()
	}
	if gq != nil {
		stepFn = gq.step
		prefillFn = gq.prefill
		reset = func() {
			model.reset()
			gq.gpuLen = 0
		}
	}

	if *serveAddr != "" {
		s := &server{
			model: model, tok: tok, system: *system, nCtx: nCtx,
			temp: *temp, topP: *topP, imEnd: imEnd, eot: eot,
			asst: asst, prefill: prefillFn, step: stepFn, reset: reset,
		}
		if err := s.listen(*serveAddr); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	steps := 0
	var logits []float32
	feed := func(ids []int) {
		if len(ids) > 1 {
			if draftM != nil {
				draftM.prefill(ids, steps)
			}
			logits = prefillFn(ids, steps)
			steps += len(ids)
			return
		}
		for _, id := range ids {
			if draftM != nil {
				draftM.step(id, steps)
			}
			logits = stepFn(id, steps)
			steps++
		}
	}
	// generateSpec emits tokens speculatively: the target's pending logits
	// pick a token, the draft greedily proposes specK continuations, and
	// one batched target pass scores them all — every agreed position is
	// accepted, the first disagreement's row supplies the corrected next
	// pick, and both KV caches roll back to the accepted length.
	generateSpec := func(limit int) {
		start := time.Now()
		gen, accepted, proposed := 0, 0, 0
		for gen < limit && steps < nCtx-2-*specK {
			c0 := sample(logits, 0, 1, rng)
			if c0 == imEnd || c0 == eot {
				feed([]int{c0})
				break
			}
			fmt.Print(tok.Decode([]int{c0}))
			gen++
			props := make([]int, 0, *specK)
			dl := draftM.step(c0, steps)
			for i := 0; i < *specK; i++ {
				d := sample(dl, 0, 1, rng)
				props = append(props, d)
				dl = draftM.step(d, steps+1+i)
			}
			lm := model.prefillLogits(append([]int{c0}, props...), steps)
			j, stop := 0, false
			for ; j < len(props); j++ {
				row := lm.Data[j*lm.Cols : (j+1)*lm.Cols]
				if sample(row, 0, 1, rng) != props[j] {
					break
				}
				if props[j] == imEnd || props[j] == eot {
					j++
					stop = true
					break
				}
				fmt.Print(tok.Decode([]int{props[j]}))
				gen++
			}
			accepted += j
			proposed += len(props)
			model.truncate(steps + 1 + j)
			draftM.truncate(steps + 1 + j)
			steps += 1 + j
			logits = lm.Data[j*lm.Cols : (j+1)*lm.Cols]
			if stop {
				break
			}
		}
		fmt.Println()
		fmt.Fprintf(os.Stderr, "(%d tokens, %.1f tok/s, %d/%d drafts accepted)\n",
			gen, float64(gen)/time.Since(start).Seconds(), accepted, proposed)
	}

	generate := func(limit int) {
		if draftM != nil {
			generateSpec(limit)
			return
		}
		start := time.Now()
		gen := 0
		for ; gen < limit && steps < nCtx-1; gen++ {
			next := sample(logits, *temp, *topP, rng)
			if next == imEnd || next == eot {
				feed([]int{next})
				break
			}
			fmt.Print(tok.Decode([]int{next}))
			feed([]int{next})
		}
		fmt.Println()
		fmt.Fprintf(os.Stderr, "(%d tokens, %.1f tok/s)\n",
			gen, float64(gen)/time.Since(start).Seconds())
	}

	if *chat {
		feed(tok.Encode("<|im_start|>system\n" + *system + "<|im_end|>\n"))
		fmt.Fprintln(os.Stderr, "chat mode: type a message, empty line or Ctrl-D to quit")
		sc := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("> ")
			if !sc.Scan() || strings.TrimSpace(sc.Text()) == "" {
				break
			}
			feed(tok.Encode("<|im_start|>user\n" + sc.Text() + "<|im_end|>\n" + asst))
			generate(*n)
			feed(tok.Encode("\n"))
			if steps >= nCtx-64 {
				fmt.Fprintln(os.Stderr, "context window exhausted")
				break
			}
		}
		return
	}

	text := *prompt
	if !*raw {
		text = "<|im_start|>system\n" + *system + "<|im_end|>\n" +
			"<|im_start|>user\n" + *prompt + "<|im_end|>\n" + asst
	}
	ids := tok.Encode(text)
	fmt.Fprintf(os.Stderr, "prompt: %d tokens\n", len(ids))
	start = time.Now()
	feed(ids)
	fmt.Fprintf(os.Stderr, "prefill: %v\n", time.Since(start).Round(time.Millisecond))
	generate(*n)
	fmt.Fprintf(os.Stderr, "%d tokens total in %v\n",
		steps, time.Since(start).Round(time.Millisecond))
}

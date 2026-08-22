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
	"time"

	"github.com/mattn/tensai/tokenizer"
)

const hfBase = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/"

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

func sample(logits []float32, temp float64, rng *rand.Rand) int {
	if temp <= 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return best
	}
	type cand struct {
		id int
		p  float64
	}
	cands := make([]cand, len(logits))
	maxl := logits[0]
	for _, v := range logits {
		if v > maxl {
			maxl = v
		}
	}
	var sum float64
	for i, v := range logits {
		p := math.Exp(float64(v-maxl) / temp)
		cands[i] = cand{i, p}
		sum += p
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].p > cands[b].p })
	r := rng.Float64() * sum
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
	prompt := flag.String("prompt", "What is the capital of France?", "user message (or raw prompt with -raw)")
	system := flag.String("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "system message for the chat template")
	raw := flag.Bool("raw", false, "skip the chat template, complete the prompt as-is")
	n := flag.Int("n", 256, "max tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 = greedy")
	seed := flag.Int64("seed", 1, "sampling seed for -temp > 0")
	q8 := flag.Bool("q8", false, "decode against int8-quantized weights")
	q4 := flag.Bool("q4", false, "decode against int4-quantized weights (group-wise)")
	cpuprofile := flag.String("cpuprofile", "", "write a CPU profile of generation to this file")
	flag.Parse()

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

	tok, err := tokenizer.Load(paths[0])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	bits := 0
	if *q8 {
		bits = 8
	}
	if *q4 {
		bits = 4
	}
	start := time.Now()
	model, err := loadQwen(paths[1], weights, bits)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	how := "float32"
	if bits != 0 {
		how = fmt.Sprintf("int%d", bits)
	}
	fmt.Fprintf(os.Stderr, "loaded qwen2 (%d layers, hidden %d) as %s in %v\n",
		model.cfg.Layers, model.cfg.HiddenSize, how, time.Since(start).Round(time.Millisecond))

	text := *prompt
	if !*raw {
		text = "<|im_start|>system\n" + *system + "<|im_end|>\n" +
			"<|im_start|>user\n" + *prompt + "<|im_end|>\n" +
			"<|im_start|>assistant\n"
	}
	ids := tok.Encode(text)
	fmt.Fprintf(os.Stderr, "prompt: %d tokens\n", len(ids))
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
	start = time.Now()
	var logits []float32
	for pos, id := range ids {
		logits = model.step(id, pos)
	}
	steps := len(ids)
	for i := 0; i < *n; i++ {
		next := sample(logits, *temp, rng)
		if next == imEnd || next == eot {
			break
		}
		fmt.Print(tok.Decode([]int{next}))
		logits = model.step(next, steps)
		steps++
	}
	fmt.Println()
	fmt.Fprintf(os.Stderr, "%d tokens in %v (%.1f tok/s)\n",
		steps, time.Since(start).Round(time.Millisecond),
		float64(steps)/time.Since(start).Seconds())
}

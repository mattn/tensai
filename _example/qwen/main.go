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
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
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
	flag.Parse()

	var paths [3]string
	for i, name := range []string{"model.safetensors", "tokenizer.json", "config.json"} {
		p, err := fetch(*dataDir, name)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		paths[i] = p
	}

	tok, err := tokenizer.Load(paths[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	start := time.Now()
	model, err := loadQwen(paths[2], paths[0])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded %s (%d layers, hidden %d) in %v\n",
		"qwen2.5-0.5b-instruct", model.cfg.Layers, model.cfg.HiddenSize,
		time.Since(start).Round(time.Millisecond))
	if *q8 {
		start = time.Now()
		model.quantize()
		fmt.Fprintf(os.Stderr, "quantized to int8 in %v\n", time.Since(start).Round(time.Millisecond))
	}

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

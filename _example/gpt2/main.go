// Command gpt2 runs the real, published GPT-2 small (124M) checkpoint in
// pure Go: the weights load through tensai's encoding/safetensors reader,
// the byte-level BPE tokenizer is implemented from vocab.json and
// merges.txt, and every matvec in the transformer runs on tensai's Dot
// kernel — build with GOEXPERIMENT=simd for the AVX2 version.
//
// On first run it downloads the checkpoint (~550MB), vocabulary, and
// merges from Hugging Face into -data. Then:
//
//	GOEXPERIMENT=simd go run ./_example/gpt2 -prompt "The meaning of life is" -n 40
//
// -temp 0 (the default) decodes greedily and deterministically; a positive
// temperature samples.
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
)

const hfBase = "https://huggingface.co/openai-community/gpt2/resolve/main/"

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

// sample picks the next token: argmax at temperature 0, otherwise a draw
// from the tempered distribution.
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
	var maxl float32 = logits[0]
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
	dataDir := flag.String("data", "_example/gpt2/data", "directory for the downloaded model files")
	prompt := flag.String("prompt", "Hello, I'm a language model,", "prompt text")
	n := flag.Int("n", 40, "tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 = greedy")
	seed := flag.Int64("seed", 1, "sampling seed for -temp > 0")
	flag.Parse()

	var paths [3]string
	for i, name := range []string{"model.safetensors", "vocab.json", "merges.txt"} {
		p, err := fetch(*dataDir, name)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		paths[i] = p
	}

	tok, err := newTokenizer(paths[1], paths[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	start := time.Now()
	model, err := loadModel(paths[0])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded gpt2 (124M) in %v\n", time.Since(start).Round(time.Millisecond))

	ids := tok.Encode(*prompt)
	if len(ids) == 0 || len(ids) >= nCtx {
		fmt.Fprintf(os.Stderr, "prompt must be 1..%d tokens, got %d\n", nCtx-1, len(ids))
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "prompt tokens: %v\n", ids)
	rng := rand.New(rand.NewSource(*seed))

	fmt.Print(*prompt)
	start = time.Now()
	var logits []float32
	for pos, id := range ids {
		logits = model.step(id, pos)
	}
	steps := len(ids)
	for i := 0; i < *n && steps < nCtx; i++ {
		next := sample(logits, *temp, rng)
		fmt.Print(tok.Decode([]int{next}))
		logits = model.step(next, steps)
		steps++
	}
	fmt.Println()
	fmt.Fprintf(os.Stderr, "%d tokens in %v (%.1f tok/s)\n",
		steps, time.Since(start).Round(time.Millisecond),
		float64(steps)/time.Since(start).Seconds())
}

// Command gpt2 runs the real, published GPT-2 small (124M) checkpoint in
// pure Go: the weights load through tensai's encoding/safetensors reader,
// the text goes through tensai's tokenizer package (byte-level BPE from
// tokenizer.json), and every matvec in the transformer runs on tensai's
// Dot kernel — build with GOEXPERIMENT=simd for the AVX2 version.
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

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/tokenizer"
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

// sample picks the next token: argmax at temperature 0, otherwise a
// tempered draw restricted to the nucleus — the smallest
// probability-sorted prefix holding topP of the mass, found through a
// histogram over probability exponents so only that prefix ever gets
// sorted (the same scheme as the qwen example). topP >= 1 keeps the
// whole distribution with a plain CDF walk.
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
	var maxl float32 = logits[0]
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
		r := rng.Float64() * sum
		for i, p := range ps {
			r -= p
			if r <= 0 {
				return i
			}
		}
		return len(ps) - 1
	}

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
	dataDir := flag.String("data", "_example/gpt2/data", "directory for the downloaded model files")
	prompt := flag.String("prompt", "Hello, I'm a language model,", "prompt text")
	n := flag.Int("n", 40, "tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 = greedy")
	topP := flag.Float64("topp", 0.9, "nucleus sampling: keep the smallest set of tokens with this much probability mass (1 disables)")
	seed := flag.Int64("seed", 1, "sampling seed for -temp > 0")
	useGPU := flag.Bool("gpu", false, "run prompt-prefill attention on the GPU (build with -tags wgpu or wgpu24)")
	q8 := flag.Bool("q8", false, "decode against int8-quantized weights (weight-only, per-column scales)")
	flag.Parse()

	var gpu *tensai.GPU
	if *useGPU {
		g, err := tensai.OpenGPU()
		if err != nil {
			fmt.Fprintf(os.Stderr, "gpu unavailable, using cpu: %v\n", err)
		} else {
			defer g.Close()
			fmt.Fprintf(os.Stderr, "gpu: %s\n", g.Name())
			gpu = g
		}
	}

	var paths [2]string
	for i, name := range []string{"model.safetensors", "tokenizer.json"} {
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
	model, err := loadModel(paths[0])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded gpt2 (124M) in %v\n", time.Since(start).Round(time.Millisecond))
	if *q8 {
		start = time.Now()
		model.quantize()
		fmt.Fprintf(os.Stderr, "quantized decode weights to int8 in %v\n", time.Since(start).Round(time.Millisecond))
	}

	ids := tok.Encode(*prompt)
	if len(ids) == 0 || len(ids) >= nCtx {
		fmt.Fprintf(os.Stderr, "prompt must be 1..%d tokens, got %d\n", nCtx-1, len(ids))
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "prompt tokens: %v\n", ids)
	rng := rand.New(rand.NewSource(*seed))

	fmt.Print(*prompt)
	start = time.Now()
	logits := model.prefill(ids, gpu)
	fmt.Fprintf(os.Stderr, "prefill: %d tokens in %v\n", len(ids), time.Since(start).Round(time.Millisecond))
	steps := len(ids)
	for i := 0; i < *n && steps < nCtx; i++ {
		next := sample(logits, *temp, *topP, rng)
		fmt.Print(tok.Decode([]int{next}))
		logits = model.step(next, steps)
		steps++
	}
	fmt.Println()
	fmt.Fprintf(os.Stderr, "%d tokens in %v (%.1f tok/s)\n",
		steps, time.Since(start).Round(time.Millisecond),
		float64(steps)/time.Since(start).Seconds())
}

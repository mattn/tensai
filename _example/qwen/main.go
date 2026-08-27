// Command qwen runs the published Qwen2.5-0.5B-Instruct checkpoint in pure
// Go: BF16 weights load through encoding/safetensors, text goes through
// the tokenizer package, and the Qwen2 architecture — RMSNorm, rotary
// embeddings, grouped-query attention, SwiGLU — decodes with a KV cache.
// Build with GOEXPERIMENT=simd and pass -q8 for int8 decode weights.
//
// The engine lives in internal/llm — cmd/tensai is the subcommand
// interface over the same code — and this example keeps the original
// flat flags:
//
//	GOEXPERIMENT=simd go run ./_example/qwen -q8 -prompt "What is the capital of France?"
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"

	"github.com/mattn/tensai/internal/llm"
)

func main() {
	o := llm.Options{Log: os.Stderr}
	flag.StringVar(&o.Data, "data", "", "directory for the downloaded model files (default: the per-repo directory under the user cache)")
	flag.StringVar(&o.Repo, "repo", llm.DefaultRepo, "Hugging Face repo to download missing model files from")
	prompt := flag.String("prompt", "What is the capital of France?", "user message (or raw prompt with -raw)")
	flag.StringVar(&o.System, "system", llm.DefaultSystem, "system message for the chat template (Qwen's branding swaps for a neutral one on other models)")
	raw := flag.Bool("raw", false, "skip the chat template, complete the prompt as-is")
	n := flag.Int("n", 256, "max tokens to generate")
	flag.Float64Var(&o.Temp, "temp", 0, "sampling temperature; 0 = greedy")
	flag.Float64Var(&o.TopP, "topp", 0.9, "nucleus sampling: keep the smallest set of tokens with this much probability mass (1 disables)")
	flag.Int64Var(&o.Seed, "seed", 1, "sampling seed for -temp > 0")
	q8 := flag.Bool("q8", false, "decode against int8-quantized weights")
	q4 := flag.Bool("q4", false, "decode against int4-quantized weights (group-wise)")
	chat := flag.Bool("chat", false, "interactive multi-turn chat on stdin (the KV cache carries the conversation)")
	flag.BoolVar(&o.GPU, "gpu", false, "decode on the GPU (requires -q8 or -q4 and a wgpu build tag)")
	cpuprofile := flag.String("cpuprofile", "", "write a CPU profile of generation to this file")
	flag.StringVar(&o.GGUF, "gguf", "", "load model and tokenizer from a single .gguf file instead of -data/-repo")
	serveAddr := flag.String("serve", "", "serve an OpenAI-compatible /v1/chat/completions API on this address (e.g. :8080)")
	flag.BoolVar(&o.Think, "think", false, "let Qwen3 models reason in a <think> block before answering")
	flag.StringVar(&o.Draft, "draft", "", "data directory of a smaller draft model for speculative decoding")
	flag.IntVar(&o.SpecK, "spec", 3, "draft tokens proposed per speculative step (3 fills one 4-row verification block)")
	flag.BoolVar(&o.Requant, "requant", false, "requantize gguf weights through float32 instead of repacking their stored blocks (slower load, but coarser scale tables decode faster)")
	flag.BoolVar(&o.NoCache, "nocache", false, "neither write nor reuse the repack cache file the first -gguf load leaves next to the model")
	flag.Parse()
	if *q8 {
		o.Bits = 8
	}
	if *q4 {
		o.Bits = 4
	}
	if o.Data == "" && o.GGUF == "" {
		o.Data = llm.DefaultDataDir(o.Repo)
	}

	e, err := llm.Open(o)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer e.Close()

	if *serveAddr != "" {
		if err := e.Serve(*serveAddr, ""); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if *chat {
		e.Chat(os.Stdin, os.Stdout, *n)
		return
	}
	e.Generate(os.Stdout, *prompt, *raw, *n)
}

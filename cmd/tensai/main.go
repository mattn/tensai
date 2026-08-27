// Command tensai runs GGUF and safetensors language models on tensai's
// pure-Go kernels. Build with GOEXPERIMENT=simd (and -tags wgpu24 for
// the GPU path):
//
//	tensai run -q8 "What is the capital of France?"
//	tensai chat -q8 -gguf model.gguf
//	tensai serve -q8 -addr :8080
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"runtime/pprof"

	"github.com/mattn/tensai/internal/llm"
)

const usage = `usage: tensai <command> [flags]

commands:
  run      generate a completion for a prompt
  chat     interactive multi-turn chat on stdin
  serve    OpenAI-compatible /v1/chat/completions server
  version  print the version

Run "tensai <command> -h" for the command's flags.`

// modelFlags registers the flags every command shares and returns the
// Options they fill.
func modelFlags(fs *flag.FlagSet) (*llm.Options, func()) {
	o := &llm.Options{Log: os.Stderr}
	fs.StringVar(&o.Data, "data", "", "directory for the downloaded model files (default: a per-repo directory under the user cache)")
	fs.StringVar(&o.Repo, "repo", llm.DefaultRepo, "Hugging Face repo to download missing model files from")
	fs.StringVar(&o.GGUF, "gguf", "", "load model and tokenizer from a single .gguf file instead of -data/-repo")
	q8 := fs.Bool("q8", false, "decode against int8-quantized weights")
	q4 := fs.Bool("q4", false, "decode against int4-quantized weights (group-wise)")
	fs.BoolVar(&o.GPU, "gpu", false, "decode on the GPU (requires -q8 or -q4 and a wgpu build tag)")
	fs.BoolVar(&o.Requant, "requant", false, "requantize gguf weights through float32 instead of repacking their stored blocks")
	fs.BoolVar(&o.NoCache, "nocache", false, "neither write nor reuse the repack cache file the first -gguf load leaves next to the model")
	fs.StringVar(&o.Draft, "draft", "", "data directory of a smaller draft model for speculative decoding")
	fs.IntVar(&o.SpecK, "spec", 3, "draft tokens proposed per speculative step")
	fs.BoolVar(&o.Think, "think", false, "let Qwen3 models reason in a <think> block before answering")
	fs.StringVar(&o.System, "system", llm.DefaultSystem, "system message for the chat template")
	fs.Float64Var(&o.Temp, "temp", 0, "sampling temperature; 0 = greedy")
	fs.Float64Var(&o.TopP, "topp", 0.9, "nucleus sampling: keep the smallest set of tokens with this much probability mass (1 disables)")
	fs.Int64Var(&o.Seed, "seed", 1, "sampling seed for -temp > 0")
	// Bits and the default data directory resolve only after Parse.
	finish := func() {
		if *q8 {
			o.Bits = 8
		}
		if *q4 {
			o.Bits = 4
		}
		if o.Data == "" && o.GGUF == "" {
			o.Data = llm.DefaultDataDir(o.Repo)
		}
	}
	return o, finish
}

func openEngine(o *llm.Options, finish func()) *llm.Engine {
	finish()
	e, err := llm.Open(*o)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return e
}

func profileTo(name string) func() {
	if name == "" {
		return func() {}
	}
	f, err := os.Create(name)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	pprof.StartCPUProfile(f)
	return pprof.StopCPUProfile
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]
	switch cmd {
	case "run":
		fs := flag.NewFlagSet("tensai run", flag.ExitOnError)
		o, finish := modelFlags(fs)
		prompt := fs.String("prompt", "", "user message (or raw prompt with -raw); positional arguments join into one")
		raw := fs.Bool("raw", false, "skip the chat template, complete the prompt as-is")
		n := fs.Int("n", 256, "max tokens to generate")
		cpuprofile := fs.String("cpuprofile", "", "write a CPU profile of generation to this file")
		fs.Parse(args)
		text := *prompt
		if rest := fs.Args(); len(rest) > 0 {
			if text != "" {
				fmt.Fprintln(os.Stderr, "give the prompt either with -prompt or as arguments, not both")
				os.Exit(2)
			}
			text = joinArgs(rest)
		}
		if text == "" {
			fmt.Fprintln(os.Stderr, "usage: tensai run [flags] <prompt>")
			os.Exit(2)
		}
		e := openEngine(o, finish)
		defer e.Close()
		stop := profileTo(*cpuprofile)
		defer stop()
		e.Generate(os.Stdout, text, *raw, *n)
	case "chat":
		fs := flag.NewFlagSet("tensai chat", flag.ExitOnError)
		o, finish := modelFlags(fs)
		n := fs.Int("n", 256, "max tokens per reply")
		fs.Parse(args)
		e := openEngine(o, finish)
		defer e.Close()
		e.Chat(os.Stdin, os.Stdout, *n)
	case "serve":
		fs := flag.NewFlagSet("tensai serve", flag.ExitOnError)
		o, finish := modelFlags(fs)
		addr := fs.String("addr", ":8080", "address to listen on")
		fs.Parse(args)
		e := openEngine(o, finish)
		defer e.Close()
		if err := e.Serve(*addr); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "version":
		if bi, ok := debug.ReadBuildInfo(); ok {
			fmt.Println("tensai", bi.Main.Version)
			return
		}
		fmt.Println("tensai (unknown)")
	case "-h", "--help", "help":
		fmt.Println(usage)
	default:
		fmt.Fprintf(os.Stderr, "tensai: unknown command %q\n\n%s\n", cmd, usage)
		os.Exit(2)
	}
}

func joinArgs(a []string) string {
	s := a[0]
	for _, w := range a[1:] {
		s += " " + w
	}
	return s
}

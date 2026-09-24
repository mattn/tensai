// Command tensai runs GGUF and safetensors language models on tensai's
// pure-Go kernels. Build with GOEXPERIMENT=simd (and -tags wgpu24 for
// the GPU path):
//
//	tensai run -q8 "What is the capital of France?"
//	tensai chat -q8 -model ./model.gguf
//	tensai serve -q8 -addr :8080
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/pprof"
	"sort"
	"strings"
	"time"

	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/llm"
	"github.com/mattn/tensai/internal/qwenimage"
	"github.com/mattn/tensai/internal/simd"
	"image/png"
	"math/rand/v2"
)

const version = "0.0.30"

// revision is stamped by the release build (-X main.revision=...).
var revision = "HEAD"

const usage = `usage: tensai <command> [flags]

commands:
  run      generate a completion for a prompt
  chat     interactive multi-turn chat on stdin
  serve    OpenAI-compatible /v1/chat/completions server
  ask      answer a question by scoring options, no generation
  bench    compare CPU and GPU prefill and decode speed
  image    generate a picture from a prompt with Qwen-Image
  models   list cached models; "models rm <name>" deletes one
  version  print the version

Run "tensai <command> -h" for the command's flags.`

// modelFlags registers the flags every command shares and returns the
// Options they fill.
func modelFlags(fs *flag.FlagSet) (*llm.Options, func()) {
	o := &llm.Options{Log: os.Stderr}
	model := fs.String("model", "", `which model to run: a name from "tensai models", a path to a directory or .gguf, a Hugging Face repo to download, or org/repo/file.gguf for one of its gguf files`)
	q8 := fs.Bool("q8", false, "decode against int8-quantized weights")
	q4 := fs.Bool("q4", false, "decode against int4-quantized weights (group-wise)")
	f32 := fs.Bool("f32", false, "decode against float32 weights")
	fs.BoolVar(&o.GPU, "gpu", false, "decode on the GPU (quantized weights and a wgpu build tag)")
	fs.BoolVar(&o.Verbose, "v", false, "narrate what the model is doing: what the file says it is, how it is read, the prompt it was handed, and where a request's time went")
	fs.BoolVar(&o.Verbose, "verbose", false, "same as -v")
	fs.BoolVar(&o.Requant, "requant", false, "requantize gguf weights through float32 instead of repacking their stored blocks")
	fs.BoolVar(&o.NoCache, "nocache", false, "neither write nor reuse the repack cache file the first .gguf load leaves next to the model")
	draft := fs.String("draft", "", "a smaller same-family model for speculative decoding, named the way -model is")
	fs.IntVar(&o.SpecK, "spec", 3, "draft tokens proposed per speculative step")
	fs.StringVar(&o.Tools, "tool", "", "tools the model may call, comma-separated: "+strings.Join(llm.ToolNames(), ", "))
	fs.BoolVar(&o.Think, "think", false, "let Qwen3 models reason in a <think> block before answering")
	fs.StringVar(&o.System, "system", llm.DefaultSystem, `system message for the chat template; empty ("") sends no system turn at all`)
	fs.Float64Var(&o.Temp, "temp", 0, "sampling temperature; 0 = greedy")
	fs.Float64Var(&o.TopP, "topp", 0.9, "nucleus sampling: keep the smallest set of tokens with this much probability mass (1 disables)")
	fs.Int64Var(&o.Seed, "seed", 1, "sampling seed for -temp > 0")
	fs.Float64Var(&o.Repeat, "repeat", 1, "repeat penalty over the recent context, llama.cpp style: 1 = off, 1.1 = mild")
	fs.IntVar(&o.RepeatLastN, "repeat-last", 64, "how many recent tokens the repeat penalty looks back over")
	fs.Float64Var(&o.Presence, "presence", 0, "presence penalty on tokens already generated (OpenAI style)")
	fs.Float64Var(&o.Frequency, "frequency", 0, "frequency penalty per occurrence of a generated token (OpenAI style)")
	// Bits and the model reference resolve only after Parse.
	finish := func() {
		// Without a width the loader picks one: what the file stores,
		// narrowed to int4 when int8 would not fit the machine.
		o.Bits = llm.BitsAuto
		switch {
		case *q8:
			o.Bits = 8
		case *q4:
			o.Bits = 4
		case *f32:
			o.Bits = 0
		}
		if err := resolveModel(o, *model); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		if *draft != "" {
			d := &llm.Options{}
			if err := resolveModel(d, *draft); err != nil || d.GGUF != "" {
				fmt.Fprintf(os.Stderr, "-draft wants a model directory or a cached name, not %q\n", *draft)
				os.Exit(2)
			}
			o.Draft = d.Data
		}
	}
	return o, finish
}

// resolveModel turns the one thing a caller says about a model into the
// three the loader wants. A reference is, in order: empty for the default
// checkpoint; an existing path, to a .gguf or to a directory; a bare name
// in the cache, exactly as "tensai models" prints it; or, failing all of
// those, a Hugging Face repo to download. A download lands in the cache
// under CacheRoot; to put a model elsewhere, fetch it and name its path.
func resolveModel(o *llm.Options, ref string) error {
	if ref == "" {
		o.Repo = llm.DefaultRepo
		o.Data = llm.DefaultDataDir(o.Repo)
		return nil
	}
	local := func(p string) error {
		info, err := os.Stat(p)
		if err != nil {
			return err
		}
		if info.IsDir() {
			// The same test the listing uses: a model directory is one
			// the loader can read a config.json out of, which keeps a
			// dataset directory from being mistaken for a checkpoint.
			if _, err := os.Stat(filepath.Join(p, "config.json")); err != nil {
				return fmt.Errorf("%s has no config.json, so it is not a model directory", p)
			}
			o.Data = p
			return nil
		}
		if !strings.HasSuffix(p, ".gguf") {
			return fmt.Errorf("%s is not a model directory or a .gguf file", p)
		}
		o.GGUF = p
		return nil
	}
	// A path the user spelled out wins, so a directory named like a repo
	// still resolves to the directory in front of them. Something that
	// exists but is not a model is an error: falling through would send
	// a typo to the network as if it were a repo.
	if strings.ContainsAny(ref, `/\`) || ref == "." || ref == ".." {
		if _, err := os.Stat(ref); err == nil {
			return local(ref)
		}
	}
	// A cached model by bare name — gguf files included, so the name
	// "tensai models" prints is always a valid reference.
	root := llm.CacheRoot()
	if ref == filepath.Base(ref) {
		for _, cand := range []string{ref, ref + ".gguf"} {
			p := filepath.Join(root, cand)
			if _, err := os.Stat(p); err != nil {
				continue
			}
			return local(p)
		}
	}
	// org/repo/file.gguf names one file in a Hugging Face repo: download
	// it into the cache root, where the listing and a bare name find it.
	if strings.HasSuffix(ref, ".gguf") && !filepath.IsAbs(ref) && strings.Count(ref, "/") == 2 {
		p, err := llm.FetchGGUF(ref)
		if err != nil {
			return err
		}
		return local(p)
	}
	// A reference that can only be a filesystem path must exist: letting
	// an absolute path or a .gguf name fall through would send it to the
	// network as if it were a repo and leave a junk directory named after
	// the typo in the cache.
	if filepath.IsAbs(ref) || strings.HasSuffix(ref, ".gguf") {
		return fmt.Errorf("no model at %s", ref)
	}
	// Nothing local: the last reading that can still work is a repo, and
	// a repo has an org, so a bare name here is simply not found.
	if !strings.Contains(ref, "/") {
		return fmt.Errorf("no cached model %q under %s (see \"tensai models\", or give an org/name to download)", ref, root)
	}
	o.Repo = ref
	o.Data = llm.DefaultDataDir(ref)
	return nil
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
		jsonOut := fs.Bool("json", false, "print one JSON object with the completion and usage instead of streaming text")
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
		if *raw && o.Tools != "" {
			fmt.Fprintln(os.Stderr, "-raw skips the chat template, which is where a tool call lives; drop one of the two")
			os.Exit(2)
		}
		e := openEngine(o, finish)
		defer e.Close()
		stop := profileTo(*cpuprofile)
		defer stop()
		generate := func(w io.Writer) llm.RunResult {
			if o.Tools != "" {
				return e.GenerateTools(w, text, *n)
			}
			return e.Generate(w, text, *raw, *n)
		}
		if *jsonOut {
			var sb strings.Builder
			res := generate(&sb)
			content := strings.TrimSuffix(sb.String(), "\n")
			if res.Content != "" {
				// A tool run streamed its calls through the same writer;
				// what the caller asked for is the answer.
				content = res.Content
			}
			out, _ := json.Marshal(map[string]any{
				"content":           content,
				"finish":            res.Finish,
				"prompt_tokens":     res.PromptTokens,
				"completion_tokens": res.CompletionTokens,
				"prefill_ms":        res.Prefill.Milliseconds(),
				"total_ms":          res.Total.Milliseconds(),
				"tok_per_sec":       float64(res.CompletionTokens) / (res.Total - res.Prefill).Seconds(),
			})
			fmt.Println(string(out))
		} else {
			generate(os.Stdout)
		}
	case "chat":
		fs := flag.NewFlagSet("tensai chat", flag.ExitOnError)
		o, finish := modelFlags(fs)
		n := fs.Int("n", 256, "max tokens per reply")
		fs.Parse(args)
		e := openEngine(o, finish)
		defer e.Close()
		if o.Tools != "" {
			e.ChatTools(os.Stdin, os.Stdout, *n)
			return
		}
		e.Chat(os.Stdin, os.Stdout, *n)
	case "serve":
		fs := flag.NewFlagSet("tensai serve", flag.ExitOnError)
		o, finish := modelFlags(fs)
		defAddr := os.Getenv("TENSAI_ADDR")
		if defAddr == "" {
			defAddr = "127.0.0.1:8080"
		}
		addr := fs.String("addr", defAddr, "address to listen on (or $TENSAI_ADDR); loopback only unless widened")
		apiKey := fs.String("api-key", os.Getenv("TENSAI_API_KEY"), "require this bearer token on the /v1 API (or $TENSAI_API_KEY)")
		fs.Parse(args)
		e := openEngine(o, finish)
		defer e.Close()
		if err := e.Serve(*addr, *apiKey); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "ask":
		fs := flag.NewFlagSet("tensai ask", flag.ExitOnError)
		o, finish := modelFlags(fs)
		choice := fs.String("choice", "", "comma-separated answers to choose among")
		yesno := fs.Bool("yesno", false, "score yes against no")
		state := fs.String("state", "", "the situation the question is asked about, given ahead of it")
		label := fs.Bool("label", false, "list the options under the question lettered A, B, C and score the letter, one token each, instead of the option text")
		batch := fs.Bool("batch", false, `read a System One request from stdin: {"state": ..., "questions": {id: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}}`)
		jsonOut := fs.Bool("json", false, "print the probabilities as one JSON object")
		fs.Parse(args)
		question := joinArgs(fs.Args())
		if *batch {
			if *yesno || *choice != "" || *label || question != "" {
				fmt.Fprintln(os.Stderr, "-batch takes its questions from stdin, and -state only when the request has none")
				os.Exit(2)
			}
			var req llm.SystemOneRequest
			if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
				fmt.Fprintln(os.Stderr, "reading the request:", err)
				os.Exit(2)
			}
			if len(req.State) == 0 && *state != "" {
				req.State, _ = json.Marshal(*state)
			}
			e := openEngine(o, finish)
			defer e.Close()
			resp, err := e.SystemOne(req)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			answersPrint(resp, *jsonOut)
			return
		}
		var options []string
		switch {
		case *yesno && *choice != "":
			fmt.Fprintln(os.Stderr, "give -yesno or -choice, not both")
			os.Exit(2)
		case *yesno:
			options = []string{"yes", "no"}
		case *choice != "":
			for _, c := range strings.Split(*choice, ",") {
				if c = strings.TrimSpace(c); c != "" {
					options = append(options, c)
				}
			}
		}
		// The state is the context a decision is made in, and the model
		// reads it as the first part of the user turn. A state with no
		// question after it is the question.
		if *state != "" && question == "" {
			question, *state = *state, ""
		}
		if question == "" || len(options) < 2 {
			fmt.Fprintln(os.Stderr, "usage: tensai ask [flags] (-choice a,b,c | -yesno | -batch) [-state <situation>] [<question>]")
			os.Exit(2)
		}
		e := openEngine(o, finish)
		defer e.Close()
		res, err := e.ScoreMany(*state, []llm.Question{{Text: question, Options: options}}, *label)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		askPrint(options, res.Probs[0], *jsonOut)
	case "bench":
		fs := flag.NewFlagSet("tensai bench", flag.ExitOnError)
		o, finish := modelFlags(fs)
		p := fs.Int("p", 512, "approximate prompt tokens to prefill")
		n := fs.Int("n", 32, "tokens to decode")
		reps := fs.Int("r", 5, "timed repetitions per side, after one warm-up")
		fs.Parse(args)
		finish()
		if o.Bits <= 0 {
			// The GPU path needs quantized weights; bench both sides
			// the same way.
			o.Bits = 8
		}
		benchCmd(o, *p, *n, *reps)
	case "models":
		if err := modelsCmd(args); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "image":
		fs := flag.NewFlagSet("tensai image", flag.ExitOnError)
		model := fs.String("model", imageRepo, `which checkpoint to run, named the way run and chat take a model: a repo (Qwen/Qwen-Image-2.1, or Comfy-Org/Qwen-Image-2.1 for ComfyUI's int8 repackaging), downloaded on first use; a name from "tensai models"; or a path to a directory in either layout`)
		prompt := fs.String("prompt", "", "what to draw; positional arguments join into one")
		out := fs.String("o", "out.png", "where to write the picture")
		size := fs.Int("size", 256, "width and height in pixels; rounded down to a multiple of 32")
		steps := fs.Int("steps", 20, "denoising steps")
		seed := fs.Int64("seed", 1, "noise seed")
		f32 := fs.Bool("f32", false, "keep the weights as floats, which needs about 42GB of memory")
		q4 := fs.Bool("q4", false, "quantize the weights to four bits instead of eight: half the memory, about half again the error")
		negative := fs.String("negative", "", "what to steer away from; needs -cfg above 1")
		cfg := fs.Float64("cfg", 1, "how far to steer away from -negative: 1 is off, and anything above doubles what a step costs")
		quiet := fs.Bool("q", false, "print nothing but errors")
		fs.Bool("fetch", false, "no longer needed: a repo named by -model downloads, or finishes downloading, on its own")
		useGPU := fs.Bool("gpu", false, "run feed-forward and attention on the GPU (needs a wgpu build tag and quantized weights)")
		budget := fs.Float64("gpu-budget", 4, "gigabytes of weights the GPU may hold; past what a device can take it is dropped, and nothing reports that")
		gpuProjections := fs.Bool("gpu-projections", false, "stream attention projection weights to the GPU (requires -gpu)")
		gpuVAE := fs.Bool("gpu-vae", false, "run decoder convolutions on the GPU")
		turboLoRA := fs.String("turbo-lora", "", "path to a Viggle Qwen-Image-2.1 six-step LoRA; uses 6 steps and cfg=1")
		cpuprofile := fs.String("cpuprofile", "", "write a CPU profile of image generation to this file")
		fs.Parse(os.Args[2:])
		if *turboLoRA != "" {
			explicitSteps := false
			fs.Visit(func(f *flag.Flag) {
				if f.Name == "steps" {
					explicitSteps = true
				}
			})
			if !explicitSteps {
				*steps = 6
			}
		}
		defer profileTo(*cpuprofile)()
		text := strings.TrimSpace(*prompt + " " + strings.Join(fs.Args(), " "))
		if text == "" {
			fmt.Fprintln(os.Stderr, "tensai image: give it something to draw")
			os.Exit(2)
		}
		bits := 8
		switch {
		case *f32:
			bits = 0
		case *q4:
			bits = 4
		}
		if err := generateImage(*model, text, *negative, *out, *size, *steps, *seed, bits, *cfg, *budget, *useGPU, *quiet, imageAcceleration{*gpuProjections, *gpuVAE, *turboLoRA}); err != nil {
			fmt.Fprintln(os.Stderr, "tensai image:", err)
			os.Exit(1)
		}
	case "version":
		fmt.Printf("tensai v%s (%s)\n", version, revision)
	case "-h", "--help", "help":
		fmt.Println(usage)
	default:
		fmt.Fprintf(os.Stderr, "tensai: unknown command %q\n\n%s\n", cmd, usage)
		os.Exit(2)
	}
}

// answersPrint shows a System One response: as the JSON a program
// reads, or one block per question for the eye, in id order.
func answersPrint(resp *llm.SystemOneResponse, asJSON bool) {
	if asJSON {
		out, _ := json.Marshal(resp)
		fmt.Println(string(out))
		return
	}
	ids := make([]string, 0, len(resp.Answers))
	for id := range resp.Answers {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for i, id := range ids {
		if i > 0 {
			fmt.Println()
		}
		a := resp.Answers[id]
		switch a.Type {
		case "noul":
			fmt.Printf("%s: %5.1f%%  yes\n", id, 100**a.Noul)
			continue
		case "choice":
			fmt.Printf("%s: %s  (confidence %.2f)\n", id, a.Choice, *a.Confidence)
		case "score":
			fmt.Printf("%s: %.2f  (confidence %.2f)\n", id, *a.Score, *a.Confidence)
		}
		names := make([]string, 0, len(a.Probabilities))
		for n := range a.Probabilities {
			names = append(names, n)
		}
		sort.SliceStable(names, func(x, y int) bool {
			if a.Type == "score" {
				return names[x] < names[y]
			}
			return a.Probabilities[names[x]] > a.Probabilities[names[y]]
		})
		for _, n := range names {
			line := n
			if a.Legend != nil {
				line += "  " + a.Legend[n]
			}
			fmt.Printf("%5.1f%%  %s\n", 100*a.Probabilities[n], line)
		}
	}
}

// askPrint lists the options by probability, the way a reader wants
// them, or as one JSON object in the order given, the way a program
// does: the text form is for the eye, the JSON for the caller that asked
// a typed question and wants a typed answer back.
func askPrint(options []string, probs []float64, asJSON bool) {
	if asJSON {
		m := make(map[string]float64, len(options))
		for i, o := range options {
			m[o] = probs[i]
		}
		best := 0
		for i := range probs {
			if probs[i] > probs[best] {
				best = i
			}
		}
		out, _ := json.Marshal(struct {
			Answer        string             `json:"answer"`
			Probabilities map[string]float64 `json:"probabilities"`
		}{options[best], m})
		fmt.Println(string(out))
		return
	}
	idx := make([]int, len(options))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	// The number leads: it has a fixed width, and the label after it
	// needs none, which spares this code guessing at how wide a string
	// is on a terminal.
	for _, i := range idx {
		fmt.Printf("%5.1f%%  %s\n", 100*probs[i], options[i])
	}
}

// benchCmd compares prefill and decode speed on the CPU and the GPU.
// Each side runs in its own child process so the second measurement never
// pays for the first one's heap: a freed model keeps its pages resident
// long enough to distort a back-to-back run in one process.
func benchCmd(o *llm.Options, p, n, reps int) {
	if side := os.Getenv("TENSAI_BENCH_CHILD"); side != "" {
		benchChild(side)
		return
	}
	var sb strings.Builder
	for sb.Len() < p*4 {
		sb.WriteString("The quick brown fox jumps over the lazy dog while considering cache coherency protocols and memory hierarchies in modern processors. ")
	}
	cfg := benchConfig{Opts: *o, Prompt: sb.String(), N: n, Reps: reps}
	cfg.Opts.Log = nil
	blob, err := json.Marshal(&cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	run := func(side string) (benchResult, error) {
		cmd := exec.Command(exe, "bench")
		cmd.Env = append(os.Environ(), "TENSAI_BENCH_CHILD="+side, "TENSAI_BENCH_CONFIG="+string(blob))
		cmd.Stderr = os.Stderr
		out, err := cmd.Output()
		if err != nil {
			return benchResult{}, err
		}
		var r benchResult
		if err := json.Unmarshal(out, &r); err != nil {
			return benchResult{}, err
		}
		if r.Err != "" {
			return benchResult{}, errors.New(r.Err)
		}
		return r, nil
	}
	cpu, err := run("cpu")
	if err != nil {
		fmt.Fprintln(os.Stderr, "cpu:", err)
		os.Exit(1)
	}
	dev, gpuErr := run("gpu")

	fmt.Printf("prefill %d tokens, decode %d tokens, int%d weights\n", cpu.Tokens, n, o.Bits)
	// A binary built without GOEXPERIMENT=simd runs the portable kernels
	// and measures an order of magnitude slower, which is easy to mistake
	// for a slow machine.
	switch {
	case simd.HasAVX2:
		fmt.Println("cpu: AVX2 kernels")
	case simd.HasNEON:
		fmt.Println("cpu: NEON kernels")
	default:
		fmt.Println("cpu: portable kernels (rebuild with GOEXPERIMENT=simd on amd64 or arm64 to vectorize)")
	}
	// The two binding generations reach different adapters and differ in
	// speed on the same one, so the table names which build measured.
	tag := gpu.Backend()
	if tag == "" {
		tag = "no gpu build tag"
	}
	if gpuErr == nil && dev.Adapter != "" {
		fmt.Printf("gpu: %s via -tags %s\n", dev.Adapter, tag)
	} else {
		fmt.Printf("gpu build: %s\n", tag)
	}
	// Each side names the layout it measured: the two sides do not always
	// run the same one, and a cache an earlier run wrote can change which
	// one the next run gets, which is invisible in the numbers alone.
	row := func(name string, r benchResult) {
		fmt.Printf("%-8s %9.1f %-14s %8.1f %-10s %s\n", name,
			median(r.Prefill), spread(r.Prefill),
			median(r.Decode), spread(r.Decode), r.Layout)
	}
	fmt.Printf("\nmedian of %d runs after one warm-up, tokens/sec\n\n", reps)
	fmt.Printf("%-8s %9s %-14s %8s %-10s %s\n", "", "prefill", "", "decode", "", "weights")
	row("cpu", cpu)
	if gpuErr != nil {
		fmt.Printf("%-8s unavailable: %v\n", "gpu", gpuErr)
		return
	}
	row("gpu", dev)
	fmt.Printf("%-8s %8.2fx %-14s %7.2fx\n", "gpu/cpu",
		median(dev.Prefill)/median(cpu.Prefill), "", median(dev.Decode)/median(cpu.Decode))
}

type benchConfig struct {
	Opts   llm.Options
	Prompt string
	N      int
	Reps   int
}

type benchResult struct {
	Tokens  int
	Prefill []float64 // one sample per repetition
	Decode  []float64
	Adapter string `json:",omitempty"`
	Layout  string `json:",omitempty"`
	Err     string `json:",omitempty"`
}

// median returns the middle sample; samples are sorted in place.
func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	return v[len(v)/2]
}

// spread renders the low-high range of the samples.
func spread(v []float64) string {
	if len(v) < 2 {
		return ""
	}
	sort.Float64s(v)
	return fmt.Sprintf("(%.0f-%.0f)", v[0], v[len(v)-1])
}

// benchChild measures one side and prints the result as one JSON line.
func benchChild(side string) {
	var cfg benchConfig
	if err := json.Unmarshal([]byte(os.Getenv("TENSAI_BENCH_CONFIG")), &cfg); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// The engine's progress lines would interleave with the parent's
	// table; the measurement is the output here.
	cfg.Opts.Log = io.Discard
	cfg.Opts.GPU = side == "gpu"
	var r benchResult
	e, err := llm.Open(cfg.Opts)
	if err != nil {
		r.Err = err.Error()
	} else {
		defer e.Close()
		r.Adapter = e.GPUName()
		r.Layout = e.WeightLayout()
		// The model stays loaded across repetitions and the first pass is
		// discarded, so the samples describe steady state rather than a
		// cold cache and a ramping clock — the same thing llama-bench
		// reports.
		for i := 0; i <= cfg.Reps; i++ {
			e.Reset()
			res := e.Generate(io.Discard, cfg.Prompt, true, cfg.N)
			if i == 0 {
				r.Tokens = res.PromptTokens
				continue
			}
			r.Prefill = append(r.Prefill, float64(res.PromptTokens)/res.Prefill.Seconds())
			r.Decode = append(r.Decode, float64(res.CompletionTokens)/(res.Total-res.Prefill).Seconds())
		}
	}
	out, _ := json.Marshal(&r)
	fmt.Println(string(out))
}

// modelsCmd lists the model cache, or deletes entries with
// "models rm <name>...". Names are the directory names "models" prints.
func modelsCmd(args []string) error {
	root := llm.CacheRoot()
	if len(args) > 0 && args[0] == "rm" {
		if len(args) < 2 {
			return fmt.Errorf("usage: tensai models rm <name>...")
		}
		for _, name := range args[1:] {
			// The listing may print an org/name — or org/repo/file.gguf
			// for a downloaded gguf — which is what a user copies; the
			// cached entry is only ever the last element.
			if strings.Count(name, "/") == 2 && strings.HasSuffix(name, ".gguf") {
				name = name[strings.LastIndex(name, "/")+1:]
			}
			if org, base, ok := strings.Cut(name, "/"); ok {
				if org == "" || org != filepath.Base(org) || org == "." || org == ".." ||
					base == "" || base != filepath.Base(base) || base == "." || base == ".." {
					return fmt.Errorf("invalid model name %q", name)
				}
				// A checkpoint fetched under its org/repo name sits in
				// a directory for the org; one fetched otherwise is
				// cached under the repo alone.
				if nested := filepath.Join(root, org, base); isDir(nested) {
					if err := os.RemoveAll(nested); err != nil {
						return err
					}
					fmt.Println("removed", nested)
					// The org directory goes too once nothing is left in it.
					if rest, err := os.ReadDir(filepath.Join(root, org)); err == nil && len(rest) == 0 {
						os.Remove(filepath.Join(root, org))
					}
					continue
				}
				name = base
			}
			if name != filepath.Base(name) || name == "." || name == ".." {
				return fmt.Errorf("invalid model name %q", name)
			}
			target := filepath.Join(root, name)
			if _, err := os.Stat(target); err != nil {
				return fmt.Errorf("no cached model %q (see \"tensai models\")", name)
			}
			if err := os.RemoveAll(target); err != nil {
				return err
			}
			fmt.Println("removed", target)
			if strings.HasSuffix(name, ".gguf") {
				// The repack caches and the origin sidecar live beside
				// their gguf.
				caches, _ := filepath.Glob(target + ".tensai-*")
				for _, c := range caches {
					if os.Remove(c) == nil {
						fmt.Println("removed", c)
					}
				}
			}
		}
		return nil
	}
	if len(args) > 0 {
		return fmt.Errorf("usage: tensai models [rm <name>...]")
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "no cached models under %s\n", root)
			return nil
		}
		return err
	}
	var total int64
	found := false
	others := 0
	// Rows are collected before printing: naming a model by the repo it
	// came from moves it out of directory order, and a listing that
	// looks unsorted is harder to read than one that is.
	// The name column widens to the longest entry when it prints, so a
	// long org/repo/file.gguf reference does not shear the columns after
	// it out of line.
	type row struct{ name, rest string }
	var rows []row
	emit := func(name, rest string) {
		rows = append(rows, row{name, rest})
	}
	for _, ent := range entries {
		if !ent.IsDir() {
			if !strings.HasSuffix(ent.Name(), ".gguf") {
				continue
			}
			var size int64
			newest := time.Time{}
			names, _ := filepath.Glob(filepath.Join(root, ent.Name()) + ".tensai-*.cache")
			for _, n := range append(names, filepath.Join(root, ent.Name())) {
				if info, err := os.Stat(n); err == nil {
					size += info.Size()
					if info.ModTime().After(newest) {
						newest = info.ModTime()
					}
				}
			}
			name := ent.Name()
			// A downloaded gguf lists under its repo, and the full
			// org/repo/file.gguf it prints is itself a -model reference
			// that fetches the same file elsewhere.
			if repo := llm.GGUFOrigin(filepath.Join(root, ent.Name())); repo != "" {
				name = repo + "/" + ent.Name()
			}
			emit(name, fmt.Sprintf("%8s  %-9s %-11s %s", humanSize(size), "gguf",
				llm.Inspect(filepath.Join(root, ent.Name())), newest.Format("2006-01-02")))
			total += size
			found = true
			continue
		}
		dir := filepath.Join(root, ent.Name())
		if rest, size, ok := describeModelDir(dir); ok {
			// Naming the repo rather than the directory keeps the
			// listing usable on another machine, where the model is not
			// cached yet; -model takes either form against a cache that
			// already has it.
			name := ent.Name()
			if repo := llm.Origin(dir); repo != "" {
				name = repo
			}
			emit(name, rest)
			total += size
			found = true
			continue
		}
		// A checkpoint fetched by its org/repo name (tensai image does
		// this) sits one level down, under a directory named for the
		// org; it lists under that same org/repo, which is what -model
		// and "models rm" take.
		nested := false
		subs, _ := os.ReadDir(dir)
		for _, sub := range subs {
			if !sub.IsDir() {
				continue
			}
			if rest, size, ok := describeModelDir(filepath.Join(dir, sub.Name())); ok {
				emit(ent.Name()+"/"+sub.Name(), rest)
				total += size
				found, nested = true, true
			}
		}
		if !nested {
			others++
		}
	}
	sort.Slice(rows, func(i, j int) bool {
		a, b := strings.ToLower(rows[i].name), strings.ToLower(rows[j].name)
		if a != b {
			return a < b
		}
		return rows[i].name < rows[j].name
	})
	width := 40
	for _, r := range rows {
		width = max(width, len(r.name))
	}
	for _, r := range rows {
		fmt.Printf("%-*s %s\n", width, r.name, r.rest)
	}
	if found {
		fmt.Printf("%-*s %8s  (%s)\n", width, "total", humanSize(total), root)
	} else {
		fmt.Fprintf(os.Stderr, "no cached models under %s\n", root)
	}
	if others > 0 {
		// Said plainly so the count cannot be read as models that failed
		// to list; "models rm" still accepts them by name.
		what := fmt.Sprintf("%d other directories here hold example datasets, not models", others)
		if others == 1 {
			what = "1 other directory here holds an example dataset, not a model"
		}
		fmt.Fprintln(os.Stderr, what)
	}
	return nil
}

// describeModelDir reports whether dir holds a model and, if so, its
// listing columns and size on disk. The cache root is shared with the
// examples, which park their datasets (iris, mnist) beside the
// checkpoints, so a directory counts only when something can load it: a
// config.json for run, chat and serve, or the transformer directory a
// diffusers checkpoint keeps for tensai image.
func describeModelDir(dir string) (rest string, size int64, ok bool) {
	kind, caps := "", "-"
	if raw, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
		kind = "?"
		var cfg struct {
			ModelType string `json:"model_type"`
		}
		if json.Unmarshal(raw, &cfg) == nil && cfg.ModelType != "" {
			kind = cfg.ModelType
		}
		caps = llm.Inspect(dir).String()
	} else if isImageModel(dir) {
		// What it can do sits in the same column as tools and think do
		// for a language model: it makes images.
		kind, caps = "diffusers", "image"
		if qwenimage.ModelDir(dir).Comfy() {
			kind = "comfyui"
		}
	} else {
		return "", 0, false
	}
	newest := time.Time{}
	filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if info, err := d.Info(); err == nil {
			size += info.Size()
			if info.ModTime().After(newest) {
				newest = info.ModTime()
			}
		}
		return nil
	})
	date := "-" // a checkpoint whose download has not written a file yet
	if !newest.IsZero() {
		date = newest.Format("2006-01-02")
	}
	return fmt.Sprintf("%8s  %-9s %-11s %s", humanSize(size), kind, caps, date), size, true
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func humanSize(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0fMB", float64(n)/(1<<20))
	default:
		return fmt.Sprintf("%dB", n)
	}
}

func joinArgs(a []string) string {
	return strings.Join(a, " ")
}

type imageAcceleration struct {
	projections, vae bool
	turboLoRA        string
}

// generateImage runs Qwen-Image-2.1 from a text prompt to a square RGBA
// PNG. The two models are seven gigabytes apiece and only one is held
// at a time: the prompt is encoded and the encoder released before the
// denoising transformer loads.
func generateImage(model, prompt, negative, out string, size, steps int, seed int64, bits int, cfg, budget float64, useGPU, quiet bool, accel imageAcceleration) error {
	if accel.projections && !useGPU {
		return fmt.Errorf("-gpu-projections requires -gpu")
	}
	if (useGPU || accel.vae) && gpu.Backend() == "" {
		return fmt.Errorf("GPU image generation requires a build with -tags wgpu or wgpu24")
	}
	if steps < 2 {
		return fmt.Errorf("-steps must be at least 2")
	}
	if accel.turboLoRA != "" && (steps != 6 || cfg != 1 || negative != "") {
		return fmt.Errorf("-turbo-lora requires 6 steps, -cfg 1 and no negative prompt")
	}
	say := func(format string, args ...any) {
		if !quiet {
			fmt.Fprintf(os.Stderr, format+"\n", args...)
		}
	}
	dir, err := imageModelDir(model, say)
	if err != nil {
		return err
	}
	// The decoder turns each latent position into a sixteen-pixel
	// square, and the checkpoint's own layout groups those in twos.
	side := size / 32 * 2
	if side < 2 {
		return fmt.Errorf("a size of %d leaves nothing to draw; 64 is the smallest that works", size)
	}

	// Both prompts go through the encoder in one load, since it is seven
	// gigabytes and guidance needs the second one.
	prompts := []string{prompt}
	guide := &qwenimage.Guidance{Scale: cfg}
	if cfg > 1 {
		if negative == "" {
			// Qwen has no beginning-of-sequence token, so the encoder
			// needs something to read.
			negative = " "
		}
		prompts = append(prompts, negative)
	} else if negative != "" {
		return fmt.Errorf("-negative does nothing without -cfg above 1")
	}
	start := time.Now()
	hidden, err := qwenimage.EncodePrompts(dir, prompts, bits)
	if err != nil {
		return err
	}
	text := hidden[0]
	if len(hidden) > 1 {
		guide.Text = hidden[1]
	}
	say("prompt: %d tokens in %v", text.Rows, time.Since(start).Round(time.Second))

	start = time.Now()
	m, err := qwenimage.LoadTransformer(dir.Transformer(), bits)
	if err != nil {
		return err
	}
	defer m.Close()
	say("transformer: loaded in %v", time.Since(start).Round(time.Second))
	if accel.turboLoRA != "" {
		start = time.Now()
		if err := qwenimage.LoadTurboLoRA(m, accel.turboLoRA); err != nil {
			return err
		}
		say("turbo: adapter loaded in %v", time.Since(start).Round(time.Second))
	}
	if useGPU {
		start = time.Now()
		name, held, err := qwenimage.UseGPU(m, uint64(budget*(1<<30)))
		if err != nil {
			return err
		}
		say("gpu: %s holding %.1fGiB of feed-forward weights in %v", name, float64(held)/(1<<30), time.Since(start).Round(time.Second))
		if accel.projections {
			if err := qwenimage.UseGPUProjections(m, uint64(budget*(1<<30))); err != nil {
				return err
			}
			say("gpu: streaming attention projections")
		}
	}

	latents := qwenimage.Noise(rand.New(rand.NewPCG(uint64(seed), 0)), side, side)
	layout := qwenimage.NewLayout(text.Rows, side, side)
	start = time.Now()
	schedule := qwenimage.NewSchedule(steps, side*side)
	if accel.turboLoRA != "" {
		schedule = qwenimage.NewTurboSchedule(side * side)
	}
	err = qwenimage.Generate(m, latents, text, layout, schedule, guide, func(i int) {
		say("step %d/%d in %v", i+1, steps, time.Since(start).Round(time.Second))
	})
	if err != nil {
		return err
	}
	say("%d steps in %v", steps, time.Since(start).Round(time.Second))
	// The decoder no longer needs the transformer. Release its mapped
	// weights and GPU allocations before allocating full-resolution maps.
	start = time.Now()
	if err := m.Close(); err != nil {
		return err
	}
	say("transformer: released in %v", time.Since(start).Round(time.Second))

	start = time.Now()
	stats, err := qwenimage.LoadStats(dir.VAEConfig())
	if err != nil {
		return err
	}
	dec, err := qwenimage.LoadDecoder(dir.VAE())
	if err != nil {
		return err
	}
	decode := qwenimage.Decode
	if accel.vae {
		decode = qwenimage.DecodeGPU
	}
	px, err := decode(dec, stats.Denormalize(latents, side, side))
	if err != nil {
		return err
	}
	img, err := qwenimage.Image(px)
	if err != nil {
		return err
	}
	say("decoder: loaded and decoded in %v", time.Since(start).Round(time.Second))
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return err
	}
	say("wrote %s, %dx%d", out, side*16, side*16)
	return nil
}

// imageModelDir resolves -model the way resolveModel does for run and
// chat: a path that exists, a name "tensai models" prints, or a repo.
// The two repos tensai image knows how to download are fetched when
// anything is missing, which also finishes an interrupted download; with
// everything in place that is a stat per file and nothing on the wire.
func imageModelDir(ref string, say func(string, ...any)) (qwenimage.ModelDir, error) {
	root := llm.CacheRoot()
	local := func(p string) (qwenimage.ModelDir, error) {
		if !isImageModel(p) {
			return "", fmt.Errorf("%s holds no image checkpoint: want text_encoder, transformer and vae (diffusers) or text_encoders, diffusion_models and vae (ComfyUI)", p)
		}
		return qwenimage.ModelDir(p), nil
	}
	if strings.ContainsAny(ref, `/\`) || ref == "." || ref == ".." {
		if _, err := os.Stat(ref); err == nil {
			return local(ref)
		}
	}
	if ref == filepath.Base(ref) {
		if p := filepath.Join(root, ref); isDir(p) {
			return local(p)
		}
		return "", fmt.Errorf("no cached image model %q under %s (see \"tensai models\", or give %s or %s to download)", ref, root, imageRepo, comfyImageRepo)
	}
	if filepath.IsAbs(ref) {
		return "", fmt.Errorf("no model at %s", ref)
	}
	dir := imageCacheDir(ref)
	fetch := map[string]func(string, func(string, ...any)) error{
		imageRepo:      fetchImageModel,
		comfyImageRepo: fetchComfyImageModel,
	}[ref]
	if fetch == nil {
		if isImageModel(dir) {
			return qwenimage.ModelDir(dir), nil
		}
		return "", fmt.Errorf("%s is not cached, and tensai image can download only %s and %s", ref, imageRepo, comfyImageRepo)
	}
	if err := fetch(dir, say); err != nil {
		return "", err
	}
	if ref == imageRepo {
		// It sits under its bare name, as run and chat cache a repo;
		// the record lets the listing name it by the repo.
		llm.RecordOrigin(dir, ref)
	}
	return qwenimage.ModelDir(dir), nil
}

// isImageModel reports whether dir holds a checkpoint tensai image can
// run, in the diffusers layout or ComfyUI's.
func isImageModel(dir string) bool {
	return isDir(filepath.Join(dir, "transformer")) || qwenimage.ModelDir(dir).Comfy()
}

// imageCacheDir is where a repo's checkpoint is cached. Qwen's sits under
// its bare name, as run and chat cache a repo. ComfyUI's has the same
// bare name, so it goes under its org instead, which the listing prints
// as the repo.
func imageCacheDir(repo string) string {
	root := llm.CacheRoot()
	if repo == imageRepo {
		return filepath.Join(root, filepath.Base(repo))
	}
	return filepath.Join(root, filepath.FromSlash(repo))
}

// fetchFile downloads one file of a repo into dir/sub unless it is
// already there, naming it only when there is something to fetch.
func fetchFile(repo, dir, sub, name string, say func(string, ...any)) (string, error) {
	p := filepath.Join(dir, sub, name)
	if _, err := os.Stat(p); err == nil {
		return p, nil
	}
	say("fetching %s/%s from %s", sub, name, repo)
	return llm.Fetch("https://huggingface.co/"+repo+"/resolve/main/"+sub+"/", filepath.Join(dir, sub), name)
}

// imageRepo is where the checkpoint lives, and comfyImageRepo is
// ComfyUI's repackaging of it.
const (
	imageRepo      = "Qwen/Qwen-Image-2.1"
	comfyImageRepo = "Comfy-Org/Qwen-Image-2.1"
)

// fetchComfyImageModel downloads ComfyUI's int8 files into dir, laid out
// as the repo has them, and the two small files it does not ship (the
// VAE's config, whose latent statistics the decoder needs, and the
// tokenizer) from Qwen's.
func fetchComfyImageModel(dir string, say func(string, ...any)) error {
	for _, f := range []struct{ repo, sub, name string }{
		{imageRepo, "processor", "tokenizer.json"},
		{imageRepo, "vae", "config.json"},
		{comfyImageRepo, "vae", "qwen_image_2.1_vae_bf16.safetensors"},
		{comfyImageRepo, "text_encoders", "qwen3vl_8b_int8_convrot.safetensors"},
		{comfyImageRepo, "diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"},
	} {
		if _, err := fetchFile(f.repo, dir, f.sub, f.name, say); err != nil {
			return err
		}
	}
	return nil
}

// fetchImageModel downloads the checkpoint's parts into dir. The shard
// names come from each component's index rather than a list here, so a
// repository that re-splits its weights still resolves.
func fetchImageModel(dir string, say func(string, ...any)) error {
	for _, f := range []struct{ sub, name string }{
		{"processor", "tokenizer.json"},
		{"vae", "config.json"},
		{"vae", "diffusion_pytorch_model.safetensors"},
	} {
		if _, err := fetchFile(imageRepo, dir, f.sub, f.name, say); err != nil {
			return err
		}
	}
	for _, c := range []struct{ sub, index string }{
		{"transformer", "diffusion_pytorch_model.safetensors.index.json"},
		{"text_encoder", "model.safetensors.index.json"},
	} {
		path, err := fetchFile(imageRepo, dir, c.sub, c.index, say)
		if err != nil {
			return err
		}
		shards, err := shardNames(path)
		if err != nil {
			return err
		}
		for _, n := range shards {
			if _, err := fetchFile(imageRepo, dir, c.sub, n, say); err != nil {
				return err
			}
		}
	}
	return nil
}

// shardNames reads the weight files a safetensors index points at, in a
// stable order.
func shardNames(index string) ([]string, error) {
	b, err := os.ReadFile(index)
	if err != nil {
		return nil, err
	}
	var idx struct {
		Map map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("%s: %w", index, err)
	}
	seen := map[string]bool{}
	var out []string
	for _, v := range idx.Map {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Strings(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("%s names no weight files", index)
	}
	return out, nil
}

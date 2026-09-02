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
	"github.com/mattn/tensai/internal/simd"
)

const version = "0.0.25"

// revision is stamped by the release build (-X main.revision=...).
var revision = "HEAD"

const usage = `usage: tensai <command> [flags]

commands:
  run      generate a completion for a prompt
  chat     interactive multi-turn chat on stdin
  serve    OpenAI-compatible /v1/chat/completions server
  bench    compare CPU and GPU prefill and decode speed
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
	fs.BoolVar(&o.GPU, "gpu", false, "decode on the GPU (requires -q8 or -q4 and a wgpu build tag)")
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
	// Bits and the model reference resolve only after Parse.
	finish := func() {
		if *q8 {
			o.Bits = 8
		}
		if *q4 {
			o.Bits = 4
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
	case "bench":
		fs := flag.NewFlagSet("tensai bench", flag.ExitOnError)
		o, finish := modelFlags(fs)
		p := fs.Int("p", 512, "approximate prompt tokens to prefill")
		n := fs.Int("n", 32, "tokens to decode")
		reps := fs.Int("r", 5, "timed repetitions per side, after one warm-up")
		fs.Parse(args)
		finish()
		if o.Bits == 0 {
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
	case "version":
		fmt.Printf("tensai v%s (%s)\n", version, revision)
	case "-h", "--help", "help":
		fmt.Println(usage)
	default:
		fmt.Fprintf(os.Stderr, "tensai: unknown command %q\n\n%s\n", cmd, usage)
		os.Exit(2)
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
	if simd.HasAVX2 {
		fmt.Println("cpu: AVX2 kernels")
	} else {
		fmt.Println("cpu: portable kernels (rebuild with GOEXPERIMENT=simd on amd64 for AVX2)")
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
	row := func(name string, r benchResult) {
		fmt.Printf("%-8s %9.1f %-14s %8.1f %s\n", name,
			median(r.Prefill), spread(r.Prefill),
			median(r.Decode), spread(r.Decode))
	}
	fmt.Printf("\nmedian of %d runs after one warm-up, tokens/sec\n\n", reps)
	fmt.Printf("%-8s %9s %-14s %8s\n", "", "prefill", "", "decode")
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
					strings.Contains(base, "/") {
					return fmt.Errorf("invalid model name %q", name)
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
			emit(name, fmt.Sprintf("%8s  %-8s %-11s %s", humanSize(size), "gguf",
				llm.Inspect(filepath.Join(root, ent.Name())), newest.Format("2006-01-02")))
			total += size
			found = true
			continue
		}
		dir := filepath.Join(root, ent.Name())
		// The cache root is shared with the examples, which park their
		// datasets (iris, mnist) beside the checkpoints. What run, chat,
		// and serve can load is a directory with a config.json, so
		// anything without one is not a model and does not belong in a
		// model listing.
		raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
		if err != nil {
			others++
			continue
		}
		kind := "?"
		var cfg struct {
			ModelType string `json:"model_type"`
		}
		if json.Unmarshal(raw, &cfg) == nil && cfg.ModelType != "" {
			kind = cfg.ModelType
		}
		var size int64
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
		// Naming the repo rather than the directory keeps the listing
		// usable on another machine, where the model is not cached yet;
		// -model takes either form against a cache that already has it.
		name := ent.Name()
		if repo := llm.Origin(dir); repo != "" {
			name = repo
		}
		emit(name, fmt.Sprintf("%8s  %-8s %-11s %s", humanSize(size), kind,
			llm.Inspect(dir), newest.Format("2006-01-02")))
		total += size
		found = true
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
	s := a[0]
	for _, w := range a[1:] {
		s += " " + w
	}
	return s
}

# LLM Inference

The same kernels that train a XOR network run real language models. `_example/gpt2` is a complete inference engine in pure Go, and the `tensai` command runs nine model families over the same kernels.

## GPT-2

`_example/gpt2` downloads the published GPT-2 small (124M) checkpoint from Hugging Face, loads the weights through `encoding/safetensors`, tokenizes with a from-scratch byte-level BPE, and decodes with a KV cache — every matvec running on the same `Dot` kernel as the rest of tensai, at ~30 tok/s with the AVX2 build:

```
$ GOEXPERIMENT=simd go run ./_example/gpt2 -n 20
Hello, I'm a language model, not a programming language. I'm a language model. ...
```

The greedy continuation matches GPT-2's well-known reference output token for token, which pins the whole pipeline — reader, tokenizer, and forward pass — in one check.

- `-q8` quantizes the decode-path weights to int8 and doubles generation (23 → 46 tok/s on the same machine), because decode streams the whole checkpoint per token
- `-gpu` (built with `-tags wgpu` or `wgpu24`) runs every block's causal multi-head attention as a single masked dispatch on the GPU

## Qwen and friends: nine model families

The `tensai` command runs modern instruction-tuned models: RMSNorm, rotary position embeddings, grouped-query attention, and a SwiGLU MLP, loaded from safetensors (config.json drives the dimensions, sharded checkpoints come through their index.json) or from a single llama.cpp GGUF that carries config, tokenizer, and weights in one file. One runtime speaks nine architectures:

| family | models | what it adds |
|---|---|---|
| qwen2 | Qwen 1.5/2/2.5, Qwen2.5-Coder, the R1-Distill-Qwen line | attention biases |
| qwen3 | Qwen3 dense | per-head QK-norm, explicit head_dim, `-think` |
| llama | Llama 2/3, SmolLM2, Mistral, R1-Distill-Llama | the block everyone forked |
| smollm3 | SmolLM3-3B | RoPE skipped every fourth layer |
| gemma3 | Gemma 3 | sliding windows on 5/6 layers, sandwich norms, gelu-tanh gate, SentencePiece |
| phi3 | Phi-3/3.5-mini | q/k/v and gate/up shipped pre-fused |
| qwen2moe / qwen3moe | Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B | top-k routed experts, a shared expert on qwen2moe |
| gpt-oss | gpt-oss-20b | MXFP4 experts, attention sinks, YaRN rope, harmony channels |

The DeepSeek-R1 distills are stock qwen2/llama blocks wearing DeepSeek's turn markers, which the loader spots in the embedded chat template and switches automatically, `<think>` reasoning included.

```
$ tensai run -q8 "What is the capital of France?"
The capital of France is Paris.
43 tokens in 1.3s (33.1 tok/s)
```

## Quantized loading

With `-q8`/`-q4` each weight quantizes as it loads and its float32 copy dies immediately, so the full-precision model never has to fit in memory. Quantized GGUF checkpoints skip the float32 detour entirely: Q8_0, Q4_0, Q5_0, the Q4_K/Q5_K/Q6_K K-quant family, and MXFP4 repack straight from the memory-mapped file, keeping llama.cpp's own quantization intact. A 1.5B Q4_K_M loads in about 3 seconds instead of 8; a 3B Q8_0 opens in 5 seconds instead of 32 (`-requant` restores the float detour, trading a much slower load for about 10% more decode speed).

The first `.gguf` load also writes the repacked weights to a cache file next to the model (`-nocache` opts out), and every later load just memory-maps it: the 1.5B Q4_K_M reopens in ~0.3 seconds, a Mistral 7B in well under a second, and gpt-oss-20b in under two. Mapped weights are clean file-backed pages the kernel can drop and re-read at will — on a machine where the model barely fits, that replaces swap thrashing with ordinary page cache behavior.

On a 15GB machine the ladder looks like: a 0.5B at ~40 tok/s with `-q8`, a 1.5B Q4_K_M at ~25 tok/s with `-q4` (tiled integer kernels, native Windows), and Qwen2.5-**7B**-Instruct — 15GB of BF16 shards, int4-quantized on the fly during a two-minute load into ~6GB resident — answering correctly at 3.5 tok/s.

## Prefill, speculative decoding, sampling

- **Batched prefill** — prompts feed through the model in blocks of eight token rows, streaming the weights once per block instead of once per token, cutting time-to-first-token by around 6x
- **Speculative decoding** — `-draft` points at a smaller same-family model (greedy only): the draft proposes a few tokens, one batched pass of the big model verifies them, and rejections roll the caches back, so the output is exactly what the big model alone would produce
- **Sampling** — `-temp` above 0 samples from the nucleus: `-topp 0.9` keeps the smallest probability-sorted set of tokens holding 90% of the mass, so the long tail where repetition loops live never gets a lottery ticket

## The `tensai` command

```bash
GOEXPERIMENT=simd go install github.com/mattn/tensai/cmd/tensai@latest
```

```
usage: tensai <command> [flags]

commands:
  run      generate a completion for a prompt
  chat     interactive multi-turn chat on stdin
  serve    OpenAI-compatible /v1/chat/completions server
  bench    compare CPU and GPU prefill and decode speed
  models   list cached models; "models rm <name>" deletes one
  version  print the version
```

All model commands share the same flags: `-model` (which model to run), `-q8`/`-q4`, `-gpu`, `-draft`, `-think`, `-system`, `-temp`, `-topp`, `-seed`, and more — run `tensai <command> -h` for the full list.

`-model` is the only thing that says which model to run, and it reads whichever
form you hand it, in this order:

| Form | Example |
|---|---|
| a name from `tensai models` | `-model Qwen3-0.6B`, `-model qwen2.5-0.5b-instruct-q8_0` |
| a path to a directory or `.gguf` | `-model ./model.gguf`, `-model /srv/checkpoints/qwen` |
| a Hugging Face repo, downloaded on first use | `-model Qwen/Qwen3-4B-Instruct-2507` |

Omit it for the default checkpoint. A local reference never downloads: a name
that is not cached, and carries no org to fetch it from, is an error pointing
back at the listing. A download lands in the user cache directory
(`~/.cache/tensai` on Linux); to keep a model anywhere else, fetch it there and
name its path. `-draft` takes the same forms, minus `.gguf`.

```bash
tensai run -q8 "What is the capital of France?"
tensai run -q8 -model Qwen3-0.6B "Explain RoPE briefly"   # a name from "tensai models"
tensai run -q8 -json "Explain RoPE briefly"      # one JSON object with usage counts
tensai chat -q8 -model ./model.gguf              # multi-turn; the KV cache carries the dialogue
tensai models                                    # list the cache; "models rm <name>" deletes
tensai bench -q8                                 # CPU vs GPU, prefill and decode
```

### Measuring CPU against GPU

`bench` prefills a synthetic prompt and decodes a few tokens twice — once on
the CPU, once on the GPU — and prints both with the ratio. Each side runs in
its own child process, so a freed model's pages never distort the other
measurement. The header names the kernels and the adapter each side is using, which
matters:
the two binding generations reach different adapters (only `wgpu24` sees
non-conformant drivers such as Mesa's dozen inside WSL2), so a build without
that tag may silently fall back to a CPU Vulkan implementation, and a binary
built without `GOEXPERIMENT=simd` measures the portable kernels, an order of
magnitude below the AVX2 ones.

```
$ GOEXPERIMENT=simd go run -tags wgpu24 ./cmd/tensai bench -q8
prefill 401 tokens, decode 32 tokens, int8 weights
cpu: AVX2 kernels
gpu: Microsoft Direct3D12 (AMD Radeon(TM) Graphics) (integrated) via -tags wgpu24

median of 5 runs after one warm-up, tokens/sec

           prefill                  decode
cpu          430.7 (357-446)          38.3 (38-39)
gpu         2241.5 (1663-2295)        28.7 (25-29)
gpu/cpu      5.20x                   0.75x
```

`-p` sets the approximate prompt length, `-n` the tokens to decode, and `-r`
the timed repetitions. Without a GPU build tag the GPU row reports why it is
unavailable. The model stays loaded across repetitions and the first pass is
discarded, so the samples describe steady state — a cold prefill on this path
can read 30% low, which is what makes an unwarmed number unfair to compare
against a tool that reports steady state. Prefill throughput still falls as
the prompt grows, since attention is quadratic, so compare at one length.

### Serving an OpenAI-compatible API

```bash
tensai serve -q8 -addr 127.0.0.1:8080
```

`models` lists only what `run`, `chat`, and `serve` can load — a directory
with a `config.json`, or a `.gguf` file. The examples cache their datasets in
the same place, and those are counted separately rather than listed as models;
`models rm` still removes them by name.

`serve` exposes `/v1/chat/completions` (messages array, SSE streaming, usage counts), so any OpenAI client pointed at it chats with a pure-Go model. A built-in chat demo page is served on `GET /`.

- The default bind is loopback only (`127.0.0.1:8080`, or `$TENSAI_ADDR`); widen it explicitly if you mean to
- `-api-key` (or `$TENSAI_API_KEY`) requires a bearer token on the `/v1` routes; the demo page stays open

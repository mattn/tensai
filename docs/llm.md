# LLM Inference

The same kernels that train a XOR network run real language models. `_example/gpt2` is a complete inference engine in pure Go, and the `tensai` command runs ten model families over the same kernels.

## GPT-2

`_example/gpt2` downloads the published GPT-2 small (124M) checkpoint from Hugging Face, loads the weights through `encoding/safetensors`, tokenizes with a from-scratch byte-level BPE, and decodes with a KV cache — every matvec running on the same `Dot` kernel as the rest of tensai, at ~30 tok/s with the AVX2 build:

```
$ GOEXPERIMENT=simd go run ./_example/gpt2 -n 20
Hello, I'm a language model, not a programming language. I'm a language model. ...
```

The greedy continuation matches GPT-2's well-known reference output token for token, which pins the whole pipeline — reader, tokenizer, and forward pass — in one check.

- `-q8` quantizes the decode-path weights to int8 and doubles generation (23 → 46 tok/s on the same machine), because decode streams the whole checkpoint per token
- `-gpu` (built with `-tags wgpu` or `wgpu24`) runs every block's causal multi-head attention as a single masked dispatch on the GPU

## Qwen and friends: ten model families

The `tensai` command runs modern instruction-tuned models: RMSNorm, rotary position embeddings, grouped-query attention, and a SwiGLU MLP, loaded from safetensors (config.json drives the dimensions, sharded checkpoints come through their index.json) or from a single llama.cpp GGUF that carries config, tokenizer, and weights in one file. One runtime speaks ten architectures:

| family | models | what it adds |
|---|---|---|
| qwen2 | Qwen 1.5/2/2.5, Qwen2.5-Coder, the R1-Distill-Qwen line | attention biases |
| qwen3 | Qwen3 dense | per-head QK-norm, explicit head_dim, `-think` |
| qwen3_5 | Qwen3.5, Qwen3.6, Qwen3.8 | a gated delta rule on three layers in four, ordinary attention on the fourth; norms scale by 1 + w, RoPE turns a quarter of each head, and the queries carry a gate for the attention output. CPU only, and no `-draft` |

A `qwen3_5` prompt costs more to prefill than its size suggests: the delta
layers carry state token by token, so only the projections around the
recurrence batch, and a long prompt is closer to decoding it than to
prefilling one. Roughly ten milliseconds a token on an AVX2 machine for the
0.8B, against four for a qwen3 of the same size — enough that a couple of
thousand tokens of system prompt is a wait. The chunked formulation the
architecture allows would close most of that and is not implemented yet.
| llama | Llama 2/3, SmolLM2, Mistral, R1-Distill-Llama | the block everyone forked |
| smollm3 | SmolLM3-3B | RoPE skipped every fourth layer |
| gemma3 | Gemma 3 | sliding windows on 5/6 layers, sandwich norms, gelu-tanh gate, SentencePiece |
| gemma4 | Gemma 4 E2B/E4B | per-layer embeddings read from disk a token at a time, two head widths, the deeper layers attending against an earlier layer's cache, logits through a tanh cap |
| phi3 | Phi-3/3.5-mini | q/k/v and gate/up shipped pre-fused |
| qwen2moe / qwen3moe | Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B | top-k routed experts, a shared expert on qwen2moe |
| gpt-oss | gpt-oss-20b | MXFP4 experts, attention sinks, YaRN rope, harmony channels |

Gemma 4 keeps most of its parameters in a per-layer embedding table — 2.3
of E2B's 4.6 billion — that no step needs more than one row of, so the
table stays in the file and every token reads its own row on demand. What
that leaves resident is an ordinary transformer about the size of a 2B.
Its layers alternate a 256-wide local head with a 512-wide global one,
the deeper two thirds project queries only and attend against the last
layer of their own kind that kept a cache, and the logits leave through a
tanh cap. It decodes on the device too: on a Radeon 780M
about 15.7 tokens a second against 14.2 on the CPU, and the same in
prefill. The two are close enough that the machine's power state decides
it -- unplugged, that GPU falls to 3.1 tokens a second while the CPU
holds 13.6.

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

Each family gets a system prompt of its own when the caller does not choose one. `-system ""` sends no system turn at all rather than an empty one, which is what a model's own template writes when it is handed no system message: it is the flag to reach for when comparing against another runtime, since the system turn is otherwise the one thing in the prompt that differs.

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

Gated repositories -- Gemma and Llama among them -- serve 401 until their
licence is accepted and a token is sent. tensai looks for one in `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN` or `HUGGINGFACE_TOKEN`, and then in the file
`huggingface-cli login` writes (`$HF_HOME/token`, by default
`~/.cache/huggingface/token`), so a machine that has logged in needs nothing
further. A refused download says which of the two is missing rather than
retrying: no token at all, or a token whose account has not accepted that
repository's licence.

A download that dies partway is kept and resumed rather than started over,
which matters when the file is fifteen gigabytes: the partial sits beside the
model as `<name>.tmp` and the next attempt asks for the rest. The resume is
guarded against the ETag recorded with it, so a checkpoint that changed
upstream restarts instead of splicing two versions together. Transport errors
and a server's 5xx are retried a few times with a widening pause; a 404 is
not.

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

```
Qwen/Qwen3-4B-Instruct-2507                 7.5GB  qwen3    tools think 2026-08-27
Qwen2.5-1.5B-Instruct                       2.9GB  qwen2    tools       2026-08-23
SmolLM2-360M-Instruct                       692MB  llama    -           2026-08-24
qwen2.5-0.5b-instruct-q8_0.gguf             531MB  gguf     tools       2026-08-25
```

A model downloaded from a repo is named by that repo, organization included,
because the cache directory drops it — and without it the listing cannot say
what to type on a machine that does not have the model yet. Either form works
against a cache that already holds it, and `models rm` takes either as well.
Checkpoints cached before this, or placed by hand, keep their bare directory
name until something downloads them again.

The fourth column says what `serve` will do with the model — accept a request
offering `tools`, and give `-think` a block to reason in — not how well it will
do it: a 0.5B checkpoint is listed as taking tools because it will be offered
them, not because its calls are reliable. The answers follow the loader exactly,
family fallback included, so a checkpoint whose own template is not on disk is
listed the way it will be treated. Reading it costs a `.gguf` about 80ms of
metadata parsing; directories are free.

`serve` exposes `/v1/chat/completions` (messages array, SSE streaming, usage counts), so any OpenAI client pointed at it chats with a pure-Go model. A built-in chat demo page is served on `GET /`.

### Thinking

With `-think`, a model that reasons before it answers keeps the two apart on the wire: the block it writes first arrives as `reasoning_content` and only the reply is `content`, streaming as its own deltas, so a client can show the thinking as thinking or drop it. Reasoning never goes back into the prompt — replayed history keeps the answer and loses the thinking, as the model's own template does.

```json
"message": {
  "role": "assistant",
  "reasoning_content": "Okay, the user is asking for 17 multiplied by 3...",
  "content": "17 multiplied by 3 is 51."
}
```

Without `-think` the qwen3 and smollm3 families open the turn with an empty block, so there is nothing to separate. gpt-oss reasons in harmony channels instead, which this does not cover.

### Tool calling

Pass `tools` and the model can answer with `tool_calls` instead of prose, which is what an agent needs to drive a loop:

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}'
```

```json
"finish_reason": "tool_calls",
"message": {"role": "assistant", "content": "", "tool_calls": [
  {"index": 0, "id": "call_0", "type": "function",
   "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]}
```

Send the result back as a `tool` message naming the call it answers, alongside the assistant turn that made it, and the model writes the reply. Streaming works the same way: text streams as usual, the calls arrive as `tool_calls` deltas indexed from zero, and the turn ends with `finish_reason: "tool_calls"`.

The signatures are handed to the model in the convention its own family was trained on. For the ChatML families — qwen2, qwen3, their MoE variants, llama, and SmolLM — that is a `<tools>` block appended to the system turn and calls written as `<tool_call>` JSON. Qwen3.5 left that behind: its `<tools>` block leads the system turn instead, and a call is a `<function=name>` element with one `<parameter=key>` per argument, which carries no types, so the tool's own JSON Schema decides what each value means. Whether a particular checkpoint was prepared for it is not a guess: its own chat template either branches on the tool definitions it may be handed or it does not, and that answer overrides the family. A GGUF carries the template in its metadata; a downloaded checkpoint gets it from `tokenizer_config.json` (or `chat_template.jinja`, where newer ones keep it). A model named by a path or a cached name is read where it sits and never fetched from, so a checkpoint cached before this existed falls back to its family until the file is there. Gemma 4 speaks a fourth, which is neither JSON nor XML: the signatures go into the system turn as `<|tool>declaration:name{...}<tool|>` blocks written in a brace DSL where a string is wrapped in `<|"|>` rather than quoted, a call comes back as `<|tool_call>call:name{key:value}<tool_call|>`, and the result answering it goes back inside the same model turn as `<|tool_response>response:name{value:...}<tool_response|>` — the model then carries on in that turn rather than opening another. Families with no such convention (gemma3, phi3, mistral, deepseek, gpt-oss), and checkpoints whose template never mentions tools (SmolLM2, say), answer a request carrying `tools` with 400 rather than dropping them silently. Nothing constrains the sampler, so a call is the model's choice: `tool_choice: "none"` withholds the signatures, but `"required"` cannot force what only a grammar could, and it reads as `"auto"`. Bigger models call far more reliably than the 0.5B default.

- The default bind is loopback only (`127.0.0.1:8080`, or `$TENSAI_ADDR`); widen it explicitly if you mean to
- `-api-key` (or `$TENSAI_API_KEY`) requires a bearer token on the `/v1` routes; the demo page stays open
- The prompt an agent resends every turn — its system message and its tool
  definitions — is prefilled once and kept. A request that extends what came
  before continues from it, and one that only shares the opening restarts from
  the point the two parted, which the server checkpoints the first time it sees
  them diverge. On a 744-token prompt that is 12s for the first question and
  under 2s for the rest. The GPU path keeps its own resident cache and does not
  take part

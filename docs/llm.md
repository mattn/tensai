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

The `tensai` command runs modern instruction-tuned models: RMSNorm, rotary position embeddings, grouped-query attention, and a SwiGLU MLP, loaded from safetensors (config.json drives the dimensions, sharded checkpoints come through their index.json) or from a single llama.cpp GGUF that carries config, tokenizer, and weights in one file. One runtime speaks eleven architectures:

| family | models | what it adds |
|---|---|---|
| qwen2 | Qwen 1.5/2/2.5, Qwen2.5-Coder, the R1-Distill-Qwen line | attention biases |
| qwen3 | Qwen3 dense | per-head QK-norm, explicit head_dim, `-think` |
| qwen3_5 | Qwen3.5, Qwen3.6, Qwen3.8 | a gated delta rule on three layers in four, ordinary attention on the fourth; norms scale by 1 + w, RoPE turns a quarter of each head, and the queries carry a gate for the attention output. The larger ones share each key head among several value heads. CPU only, no `-draft` |

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
| gemma4 | Gemma 4 E2B/E4B/12b | per-layer embeddings read from disk a token at a time, two head widths, the deeper layers attending against an earlier layer's cache, logits through a tanh cap |
| phi3 | Phi-3/3.5-mini | q/k/v and gate/up shipped pre-fused |
| qwen2moe / qwen3moe | Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B | top-k routed experts, a shared expert on qwen2moe |
| gpt-oss | gpt-oss-20b | MXFP4 experts, attention sinks, YaRN rope, harmony channels |
| k2-horizon | K2-Horizon-7B | RMSNorm taken over four groups of the row, a word class that keeps combining marks and joiners together, a 512K context. GGUF only, and CPU only |

The dense 12b differs from the E-series again: no per-layer embeddings, a kv head count stated per layer (eight on the local layers, one on the global), and no value projection on the layers that narrow, which take their values from their keys. GPU decode sits that last one out.

Gemma 4 keeps most of its parameters in a per-layer embedding table — 2.3
of E2B's 4.6 billion — that no step needs more than one row of, so the
table stays in the file and every token reads its own row on demand. What
that leaves resident is an ordinary transformer about the size of a 2B.
Its layers alternate a 256-wide local head with a 512-wide global one,
the deeper two thirds project queries only and attend against the last
layer of their own kind that kept a cache, and the logits leave through a
tanh cap. It decodes on the device too, at about the same
speed the CPU manages: on a Radeon 780M 19.4 tokens a second against
19.6, prefill about even. The two are close enough that the machine's
power state decides it. Unplugged, that GPU falls to a fifth of itself
while the CPU barely moves.

The DeepSeek-R1 distills are stock qwen2/llama blocks wearing DeepSeek's turn markers, which the loader spots in the embedded chat template and switches automatically, `<think>` reasoning included.

```
$ tensai run -q8 "What is the capital of France?"
The capital of France is Paris.
43 tokens in 1.3s (33.1 tok/s)
```

## Quantized loading

Without `-q8` or `-q4` the loader chooses the width itself: what the file stores sets the ceiling (int4 blocks such as Q4_K repack into int4 exactly and gain nothing from int8; Q8_0, Q5_K, Q6_K and float checkpoints keep their precision only at int8), and when int8 would not fit the memory the machine has available (the weights plus a quarter for the tables, and half a gigabyte over) the model narrows to int4 rather than swap. `-v` says which was picked and why; `-f32` asks for float32 weights outright. With `-q8`/`-q4` each weight quantizes as it loads and its float32 copy dies immediately, so the full-precision model never has to fit in memory. Quantized GGUF checkpoints skip the float32 detour entirely: Q8_0, Q4_0, Q5_0, the Q4_K/Q5_K/Q6_K K-quant family, MXFP4, and PrismML's ternary PTQ1_0/PQ2_0 repack straight from the memory-mapped file, keeping llama.cpp's own quantization intact. A 1.5B Q4_K_M loads in about 3 seconds instead of 8; a 3B Q8_0 opens in 5 seconds instead of 32 (`-requant` restores the float detour, trading a much slower load for about 10% more decode speed).

The first `.gguf` load also writes the repacked weights to a cache file next to the model (`-nocache` opts out), and every later load just memory-maps it: the 1.5B Q4_K_M reopens in ~0.3 seconds, a Mistral 7B in well under a second, and gpt-oss-20b in under two. Mapped weights are clean file-backed pages the kernel can drop and re-read at will — on a machine where the model barely fits, that replaces swap thrashing with ordinary page cache behavior.

On a 15GB machine the ladder looks like: a 0.5B at ~40 tok/s with `-q8`, a 1.5B Q4_K_M at ~25 tok/s with `-q4` (tiled integer kernels, native Windows), and Qwen2.5-**7B**-Instruct — 15GB of BF16 shards, int4-quantized on the fly during a two-minute load into ~6GB resident — answering correctly at 3.5 tok/s.

### Ternary weights

PrismML's Bonsai checkpoints (`Ternary-Bonsai-2-27B`, a Qwen3.8-27B) keep
every weight at -1, 0 or +1 with one f16 scale per 128, in two encodings of
their own that stock llama.cpp does not read: `PTQ1_0` packs the trits five
to a byte (5.95 GB for the 27B), `PQ2_0` one to a two-bit slot (7.21 GB).
Both repack into a ternary layout of two bits a weight, so a 27B decodes in
under 8 GB, with no width to choose: `-q8` and `-q4` are accepted and
ignored, since there is nothing to quantize.

```bash
tensai run -model prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf "What is the capital of France?"
```

The weights sit in a rotated basis: each matrix was multiplied along its
input by a blockwise Walsh-Hadamard transform with fixed sign flips before
the rounding, which spreads an activation's energy evenly across a block
and is what makes three levels enough. The file declares it under
`prism.hadamard.*`, and the loader applies the matching transform to every
activation those matrices read, and the inverse to each embedding row it
looks up; a file that declares a transform the loader does not know is
refused rather than run into noise. The embedding table stays in the file
and is read a row at a time, since a quarter million rotated rows would be
gigabytes expanded.

The ternary kernel reads the codes as the unsigned operand of the
multiply-add and the activations as the signed one, so the correction it
needs is the sum of a group's activations, shared by every column, and no
per-column table streams beside the weights. On a Ryzen 7735HS the 27B
prefills at about 6 tokens/s and decodes at 3.4, which is the memory
bandwidth (about 28 GB/s of weights a token); its answers match the
PrismML llama.cpp build token for token on the prompts tried. The first
load repacks 27 billion weights, about a minute, and writes the repack
cache (8.5 GB beside the model); later loads map it in under a second.
The model runs on the CPU, as every qwen3_5 does.

## Prefill, speculative decoding, sampling

- **Batched prefill** — prompts feed through the model in blocks of eight token rows, streaming the weights once per block instead of once per token, cutting time-to-first-token by around 6x
- **Speculative decoding** — `-draft` points at a smaller same-family model (greedy only): the draft proposes a few tokens, one batched pass of the big model verifies them, and rejections roll the caches back, so the output is exactly what the big model alone would produce
- **Sampling** — `-temp` above 0 samples from the nucleus: `-topp 0.9` keeps the smallest probability-sorted set of tokens holding 90% of the mass, so the long tail where repetition loops live never gets a lottery ticket
- **Repetition penalties** — `-frequency` and `-presence` are OpenAI's: a token this completion has generated loses `-frequency` per occurrence and `-presence` once, which is what breaks a small model out of repeating a paragraph. `-repeat` is llama.cpp's: every token in the last `-repeat-last` positions, prompt included, has its logit divided by it (1.1 is mild, 1 is off). All three shape the logits before sampling, so they steer greedy decoding too. The prompt counting for `-repeat` is a known double edge: it penalizes the language of the prompt along with everything else, and a 7B answering Japanese can drift into Chinese under it where `-frequency` leaves the language alone. The API takes the same three as `frequency_penalty`, `presence_penalty` and `repetition_penalty`. None apply under `-draft`

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

All model commands share the same flags: `-model` (which model to run), `-q8`/`-q4`/`-f32` (the weight width, chosen from the file and the memory when none is given), `-gpu`, `-draft`, `-think`, `-tool`, `-system`, `-temp`, `-topp`, `-seed`, and more — run `tensai <command> -h` for the full list.

`-v` narrates what is otherwise a silent wait. It says what the file claims to be (architecture, layer and head counts, context, vocabulary), how the weights are being read (repacked or mapped from the cache, and how long each took), which template family and system prompt were chosen, and — the one thing a caller cannot otherwise see — the prompt as rendered, markers and all. Under `serve` each request announces itself on arrival with its message and tool counts, then reports how many tokens it prefilled and how fast. It adds lines and changes nothing else.

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
tensai ask -q8 -yesno "Is Paris in France?"      # a probability, no generation
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

### Asking without generating

`tensai ask` answers a question by measuring rather than generating. The
question goes through the chat template like `run`'s would, and each option
is scored as the log-likelihood the model assigns to writing it as the start
of its answer; the softmax over those is the answer. No token is sampled, so
the model cannot reply with anything outside the list, and what it does not
know shows up as probability spread across the options rather than as a
confident invention.

```bash
tensai ask -q8 -yesno "Is Paris the capital of France? Answer yes or no."
tensai ask -q8 -choice "positive,negative,neutral" "Sentiment of: 'cold food, rude waiter'. One word."
tensai ask -q8 -state "just finished work" -choice "coffee,beer,tea" "What to drink? One word."
tensai ask -q8 -json -choice "spam,ham" "Classify: 'You have won a prize'. One word."
```

```
 99.9%  yes
  0.1%  no
```

`-state` is the situation the question is asked about, rendered ahead of it
in the user turn; `-json` returns the chosen option and the probability of
each, for a caller that asked a typed question and wants a typed answer.
The cost is one prefill plus a decode step per option token, so a 0.5B
answers in a few tens of milliseconds and the prompt's cache is rolled back
between options rather than recomputed.

Two things to know. Options are scored in the form given: `yes` and `Yes`
are different tokens, and which one a model reaches for is a property of the
model, so a question that ends in "Answer yes or no." is worth the words. And
the numbers are the model's, calibration included: a 7B asked whether Paris
is the capital of Germany can put twenty percent on yes, so read a spread
between options as the signal and a single absolute value with the model's
biases in mind. Asked about something it does not know, the same 7B that
confidently invents a biography under `run` puts every candidate near fifty
percent here, which is the honest answer it cannot give in prose.

#### Typed questions about one state

A classifier asks the same situation several things: whether a message is
urgent, which team it belongs to, how angry its writer is. `-batch` takes
those as one request on stdin, in the shape of TypeSafe's Jev API, and
answers in the same shape:

```bash
tensai ask -q8 -batch -json <<'EOF'
{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
EOF
```

```json
{"model":"tensai","answers":{
  "is_urgent":   {"type":"noul","noul":0.93},
  "department":  {"type":"choice","choice":"technical","probabilities":{"billing":0.22,"sales":0.12,"technical":0.66},"confidence":0.21},
  "frustration": {"type":"score","score":0.93,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.07,"1":0.93,"2":0.00},"confidence":0.76}},
 "usage":{"input_tokens":174,"output_tokens":8}}
```

Three question types. A `noul` is yes or no and answers with the
probability of yes; its `criteria` may say what each means. A `choice`
names its options in `criteria`, each with a description, and answers with
the chosen name, a probability per option and a confidence. A `score` lists
its levels in `criteria` from low to high, up to ten, and answers with the
expected level as a number that can land between two, plus the legend and
the distribution. `confidence` is one minus the distribution's entropy as a
fraction of its maximum, which reproduces Jev's published numbers. `state`,
`instructions` and each criterion may be a string or any JSON value; what is
not a string is shown to the model as JSON.

Underneath, every question is rendered as a multiple choice lettered A, B,
C and the letter is scored, so each option costs one token however long its
description, and the answer is one read of the logits after the question.
The state is prefilled once and each question extends that cache, so N
questions cost one prefill of the state plus one of each question rather
than N of the state. The same rendering is available on a single question
as `-label`, where the option text would otherwise be scored token by
token. Small models lean on A whatever the question, so with a 0.5B compare
the options against each other rather than trusting one in isolation.

`serve` offers the same as `POST /v1/systemone`, so a client written
against Jev can be pointed at a local model instead.

### Serving an OpenAI-compatible API

```bash
tensai serve -q8 -addr 127.0.0.1:8080
```

`models` lists only what `run`, `chat`, and `serve` can load — a directory
with a `config.json`, or a `.gguf` file. It also lists the checkpoints
`tensai image` draws with (a directory holding `transformer`, `text_encoder`
and `vae`, or ComfyUI's single files): their kind is `diffusers` or `comfyui`, and the column that says `tools` or
`think` for a language model says `image`. One fetched under its `org/repo`
name sits a level down and lists under that name. The examples cache their datasets in the same place, and those are
counted separately rather than listed as models; `models rm` still removes
them by name.

```
Qwen-Image-2.1                             52.2GB  diffusers image       2026-09-22
Qwen/Qwen3-4B-Instruct-2507                 7.5GB  qwen3     tools think 2026-08-27
Qwen2.5-1.5B-Instruct                       2.9GB  qwen2     tools       2026-08-23
SmolLM2-360M-Instruct                       692MB  llama     -           2026-08-24
qwen2.5-0.5b-instruct-q8_0.gguf             531MB  gguf      tools       2026-08-25
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

`serve` exposes `/v1/chat/completions` (messages array, SSE streaming, usage counts), so any OpenAI client pointed at it chats with a pure-Go model, and `/v1/systemone`, the typed questions of `ask -batch` over HTTP. A built-in chat demo page is served on `GET /`.

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

The signatures are handed to the model in the convention its own family was trained on. For the ChatML families — qwen2, qwen3, their MoE variants, llama, and SmolLM — that is a `<tools>` block appended to the system turn and calls written as `<tool_call>` JSON. Qwen3.5 left that behind: its `<tools>` block leads the system turn instead, and a call is a `<function=name>` element with one `<parameter=key>` per argument, which carries no types, so the tool's own JSON Schema decides what each value means. Whether a particular checkpoint was prepared for it is not a guess: its own chat template either branches on the tool definitions it may be handed or it does not, and that answer overrides the family. A GGUF carries the template in its metadata; a downloaded checkpoint gets it from `tokenizer_config.json` (or `chat_template.jinja`, where newer ones keep it). A model named by a path or a cached name is read where it sits and never fetched from, so a checkpoint cached before this existed falls back to its family until the file is there. Gemma 4 speaks a fourth, which is neither JSON nor XML: the signatures go into the system turn as `<|tool>declaration:name{...}<tool|>` blocks written in a brace DSL where a string is wrapped in `<|"|>` rather than quoted, a call comes back as `<|tool_call>call:name{key:value}<tool_call|>`, and the result answering it goes back inside the same model turn as `<|tool_response>response:name{value:...}<tool_response|>` — the model then carries on in that turn rather than opening another. Carrying on needs one more thing: the thought channel a gemma4 opens for itself at the top of every turn, which it cannot open here because the turn is already running, so the prompt hands it the marker. Without it E4B answers a tool result by stopping immediately; with it the same prompt is answered. The thinking that follows is separated as usual, so `-think` and `reasoning_content` decide who sees it. A small model sampled at temperature sometimes writes the call body and forgets the marker in front of it, so a bare `name{...}` whose name the caller declared is read as the call it plainly is. Families with no such convention (gemma3, phi3, mistral, deepseek, gpt-oss), and checkpoints whose template never mentions tools (SmolLM2, say), answer a request carrying `tools` with 400 rather than dropping them silently. Nothing constrains the sampler, so a call is the model's choice: `tool_choice: "none"` withholds the signatures, but `"required"` cannot force what only a grammar could, and it reads as `"auto"`. Bigger models call far more reliably than the 0.5B default.

- The default bind is loopback only (`127.0.0.1:8080`, or `$TENSAI_ADDR`); widen it explicitly if you mean to
- `-api-key` (or `$TENSAI_API_KEY`) requires a bearer token on the `/v1` routes; the demo page stays open
- The prompt an agent resends every turn — its system message and its tool
  definitions — is prefilled once and kept. A request that extends what came
  before continues from it, and one that only shares the opening restarts from
  the point the two parted, which the server checkpoints the first time it sees
  them diverge. On a 744-token prompt that is 12s for the first question and
  under 2s for the rest. The GPU path keeps its own resident cache and does not
  take part

### Tools without a server

`serve` waits for a client to bring the tools. `run` and `chat` can carry one
themselves: `-tool wikipedia` offers the model a Wikipedia lookup, and when it
calls, the call runs here, its result goes back into the conversation, and the
model answers from what it read.

```bash
tensai run -q4 -model unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf \
  -tool wikipedia -n 400 "Who is Linus Torvalds?"
```

```
Linus Torvalds is a Finnish and American software engineer, best known as the
creator and lead developer of the Linux kernel since 1991.
```

Calls are kept off standard output because they are addressed to the tool, not
the reader; `-json` likewise reports only the answer. Every round re-renders the
whole conversation and prefills it again, since a tool result has to be written
into the turn that asked for it, and the model may go around at most four times
before it has to answer. Leave room in `-n`: a thinking model spends tokens on
the call, on reading the result, and only then on the reply.

`wikipedia` is the one tool that needs no key and no account. It searches, then
reads the opening of the best match, so a call costs one round trip rather than
two, and the alternative titles come back with it in case the model picked the
wrong one. A query written in Japanese is looked up on `ja.wikipedia.org`, and
anything else on `en.wikipedia.org`. It is an encyclopedia and not a search
engine: it answers who or what something is, and is weak on this week's news.
Wikimedia asks that a client name itself, which is what the `tensai` user agent
does; a run behind a proxy that strips it gets 403 back as the tool's answer,
which the model reads and can say so.

<p align="center"><img src="logo.svg" width="420" alt="tensai"></p>

# tensai - a tiny machine-learning framework in Go

[![release](https://img.shields.io/github/v/release/mattn/tensai)](https://github.com/mattn/tensai/releases/latest) [![Go Reference](https://pkg.go.dev/badge/github.com/mattn/tensai.svg)](https://pkg.go.dev/github.com/mattn/tensai)

`tensai` is a small machine-learning framework for learning and experiments. It implements forward passes, backpropagation, and optimization in pure Go; the default build has no external dependencies (the optional `wgpu` build tag adds exactly one, cgo-free: `ebitengine/purego`).

**Documentation: [mattn.github.io/tensai](https://mattn.github.io/tensai/)** — guides for every package, in English and [日本語](https://mattn.github.io/tensai/ja/).

## Features

- **Matrix operations** - `Matrix` plus basic operations such as `Dot`, `Add`, `T`, and `AddBias`. Tensors are float32 (`tensai.Float`)
- **N-dimensional tensors** - `Tensor` generalizes `Matrix` to any rank: element-wise `Add`/`Sub`/`Mul`/`Div` with NumPy-style broadcasting, batched `MatMul` (the leading axes broadcast, the per-matrix products run on the same kernel as `Dot`, parallelized across the batch) plus its transposed forms `MatMulTN` and `MatMulNT`, so a backward pass never materializes a transposed copy, axis-permuting `Transpose`, `Reshape` with `-1` inference, and zero-copy views to and from `Matrix`
- **SIMD acceleration** - AVX2 kernels written with Go's experimental `simd/archsimd` package: still pure Go, no cgo, no assembly files. Matmul, ReLU/LeakyReLU, Sigmoid/Tanh/Softmax (via a vectorized polynomial `exp`), GELU (via a vectorized `erf`), LayerNorm, and the Adam update are all 8-lane vectorized. Build with `GOEXPERIMENT=simd` on amd64 (Go 1.26 and 1.27 APIs both supported via build tags); every other build uses the portable fallbacks automatically
- **Low-allocation training** - layers reuse their forward/backward scratch buffers across training steps (a full MLP step runs in ~29 allocations), so GC stays out of the training loop; `Predict` always returns freshly allocated results
- **Layers** - `Embedding`, `Dense`, `Conv2D`, `MaxPool2D`, `BatchNorm`, `LayerNorm`, `Dropout`, plus `ReLU`, `LeakyReLU`, `GELU`, `Sigmoid`, `Tanh`, and `Softmax` activations
- **WebGPU backend (experimental)** - build with `-tags wgpu` (linux, macOS, Windows) and `gpu.Open()` runs batched `MatMul` as a WGSL compute shader on any GPU wgpu-native reaches (Vulkan, Metal, D3D12 — AMD, Intel, Apple, NVIDIA). The bindings go through `ebitengine/purego`, so there is still no cgo and no C compiler: the wgpu-native shared library is dlopen-ed at runtime. A `Device` also satisfies `tensai.Accelerator`: `tensai.UseAccelerator(dev)` moves every product above 4e8 multiply-accumulates — including both transposed products a backward pass needs — onto the GPU, so an autograd training step of a 2048-wide block runs 1.4x faster with nothing else changed. Resident tensors also carry the rest of a backward pass — element-wise `Binary`, `Activate`/`ActivateGrad` (ReLU, tanh, sigmoid, and the error-function GELU the CPU kernels use), `SumCols`, and an in-place `AdamStep` — so a training graph has the kernels to stay on the device. `tape.UseDevice(dev)` does exactly that — values, gradients and the Adam update all stay resident, and one step of a 2048-wide block runs 1787ms on the CPU, 930ms with products offloaded one at a time, and 521ms resident. LayerNorm, softmax, the permutes attention needs, and the embedding scatter-add have kernels too, so a whole transformer block trains without leaving the device — 217ms per step on the CPU against 89ms resident, at model width 512
- **int8 / int4 quantization** - `quant.Quantize` / `quant.Quantize4` build weight-only quantized twins: int4 group-wise with float32 accumulation, and int8 as a full integer path — weights in interleaved row quads, activations dynamically quantized to 7 bits, and the whole dot product running on the 256-bit u8 x s8 pairwise multiply-add plus a widening pair-add — two instructions per column, four rows deep — which reaches memory bandwidth (~31GB/s of weights on 16 cores). int4 halves the weights again — the difference between a 7B model fitting in RAM or not
- **Loss functions** - `MeanSquaredError` for regression, `SoftmaxCrossEntropy` for multi-class classification, and `BinaryCrossEntropy` for binary targets
- **Optimizers** - momentum `SGD`, `Adam`, and `AdamW` (decoupled weight decay)
- **k-NN baseline** - a `knn.Classifier` whose distance matrix runs on the same SIMD matmul kernel; useful as a no-training baseline next to the networks
- **Dataset utilities** - `Dataset` pairs inputs with targets and provides `Shuffle`, train/test `Split` (copy-free views), buffer-reusing mini-batch iteration with `Batches`, and `Standardize`/`StandardizeWith`
- **Sequential models** - stack layers and run `Compile` -> `Fit` / `FitStep` -> `Predict`
- **Automatic differentiation** - a micrograd-style reverse-mode autograd engine over n-dimensional tensors (`Param` / `Input` / `Backward`), for models that don't fit the Sequential mold. Values are `Tensor`s, so the same ops run on (batch, seq, model) activations: broadcasting element-wise arithmetic, batched `MatMul`, `Transpose`/`Reshape`, last-axis `Softmax`, axis reductions, `LayerNorm`, embedding `Embed`, and `CrossEntropy` all carry gradients, and a `Matrix` is still accepted anywhere a leaf is built. `ToDot` renders the computation graph for Graphviz
- **Recurrence and attention** - `rnn.Cell`, `rnn.LSTMCell`, and single-head `rnn.SelfAttention` built on the autograd engine, with backpropagation through time handled automatically
- **Serialization** - `Save`/`Load` (and `SaveFile`/`LoadFile`) round-trip trained Sequential parameters as JSON, including BatchNorm running statistics; `SaveParams`/`LoadParams` do the same for autograd parameters (RNN/LSTM/attention cells)
- **TFLite export** - the `encoding/tflite` package marshals Sequential models (FP32, NHWC) into `.tflite` flatbuffers that run on the TFLite/LiteRT runtimes and [go-tflite](https://github.com/mattn/go-tflite), with the FlatBuffers writer implemented in-tree — still no dependencies
- **safetensors** - `encoding/safetensors` reads the checkpoint format most published model weights ship in — lazily, one tensor at a time, with F16/BF16/F64 converted to float32 — and writes F32 checkpoints; interoperability is verified against the reference implementation in both directions. Also dependency-free
- **GGUF** - `encoding/gguf` reads llama.cpp's model container — typed metadata plus lazily-loaded tensors, with F16/BF16, the block-quantized Q8_0/Q4_0/Q4_1/Q5_0/Q5_1, the K-quants Q2_K through Q6_K, IQ4_NL, and gpt-oss's MXFP4 dequantized to float32 — verified block-exact against real llama.cpp conversions. Dependency-free as well
- **ONNX export** - `encoding/onnx` marshals Sequential models into ONNX (opset 13, FP32) with a hand-written protobuf encoder; onnxruntime reproduces `Predict` to ~1e-7 relative error. ONNX convolutions are NCHW, which is tensai's own row layout, so nothing is reordered
- **Tokenizers** - the `tokenizer` package loads Hugging Face `tokenizer.json` files and implements the byte-level BPE family (GPT-2, Llama 3, Qwen, ...), including the split patterns Go's regexp cannot express, as hand-written scanners — the GPT-2, cl100k, and o200k (gpt-4o/gpt-oss) families — plus SentencePiece (Gemma, the Llama-2 era) built from GGUF vocabularies via `NewSPM`; encodings match the reference `tokenizers` library and `llama-tokenize` exactly

## Layout

```
go.mod              Module definition (github.com/mattn/tensai)
.                   Core: Float, Matrix, N-d Tensor, Dot and the AVX2/portable matmul kernels
layer               Layer interface, Dense, Conv2D, MaxPool2D, BatchNorm, LayerNorm, Dropout, Embedding, activations
loss                Loss functions (MSE, SoftmaxCrossEntropy, BCE)
optim               Optimizers (SGD, Adam, AdamW)
model               Sequential model, training loop, and JSON save/load
autograd            Reverse-mode automatic differentiation (Node graph), Trainer, parameter save/load
rnn                 rnn.Cell / LSTMCell / SelfAttention on the autograd engine
knn                 k-NN baseline classifier
dataset             Shuffle, split, standardize, mini-batch iteration
quant               int8 / int4 / grouped-int8 / MXFP4 weight-only quantization
gpu                 WebGPU backend via purego + wgpu-native (build tags wgpu / wgpu24)
internal/kernels    Element-wise kernels: scalar bodies plus the AVX2 versions incl. vectorized exp
internal/simd       Load/store shims over both simd/archsimd API generations
internal/dims       Shape arithmetic shared between the core and the GPU backend
_example/helloworld Smallest possible program: add two values on the graph
_example/dataset    Dataset workflow: shuffle, split, standardize, batches
_example/xor        Runnable XOR training example
_example/fizzbuzz   Runnable FizzBuzz classification example
_example/spiral     Runnable 3-class spiral classification example
_example/iris       Runnable Iris classification example
_example/mnist      Runnable MNIST classifier (-model dense, cnn, or knn) with save/load
_example/charrnn    Character-level LSTM text generation on the autograd engine
_example/tinygpt    Character-level transformer (multi-head causal attention) trained from scratch
_example/plasma     Demoscene-style terminal plasma rendered by a neural network
_example/dot        Graphviz DOT export of the z = x + y graph
_example/tensor     Tour of the n-d Tensor: broadcasting, batched MatMul, attention
_example/wgpu       WebGPU MatMul: adapter info, CPU cross-check, GPU vs CPU sweep
_example/gpt2       The published GPT-2 (124M) checkpoint generating text in pure Go
cmd/tensai          The tensai command: run, chat, and serve subcommands over internal/llm
```

## Usage

### Regression: learn XOR with MSE

```go
net := model.NewSequential()
net.Add(layer.NewDense(8))
net.Add(&layer.Tanh{})
net.Add(layer.NewDense(1))
net.Add(&layer.Sigmoid{})

net.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05))
net.Fit(inputs, targets, 5000)

pred, _ := net.Predict(inputs)
```

### Classification: softmax + cross-entropy

```go
net := model.NewSequential()
net.Add(layer.NewDense(8))
net.Add(&layer.ReLU{})
net.Add(layer.NewDense(2)) // output width = number of classes

net.Compile(2, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.05))
```

`SoftmaxCrossEntropy` expects targets as an `Mx1` matrix of class indices. Softmax is applied inside the loss, so `Predict` returns raw logits. Use argmax for classification.

### Datasets

```go
ds, _ := dataset.New(inputs, targets)
ds.Shuffle(rng)
train, test, _ := ds.Split(0.2)          // views, no copying
mean, std := train.Standardize()         // fit on train...
test.StandardizeWith(mean, std)          // ...apply to test

for epoch := 0; epoch < epochs; epoch++ {
	train.Batches(32, rng, func(in, tgt *tensai.Matrix) error {
		_, err := net.FitStep(in, tgt)
		return err
	})
}
```

### Convolution, regularization, and saving

```go
net := model.NewSequential()
net.Add(layer.NewConv2D(8, 3, 1, 1)) // outC, kernel, stride, pad
net.Add(&layer.ReLU{})
net.Add(layer.NewMaxPool2D(2))
net.Add(layer.NewDense(64))
net.Add(layer.NewBatchNorm())
net.Add(layer.NewLeakyReLU(0.01))
net.Add(layer.NewDropout(0.3))
net.Add(layer.NewDense(10))

// The input geometry is stated once; the spatial shape threads through
// the stack, so the conv and pool layers pick their dimensions up from it.
net.CompileImage(layer.Image{H: 28, W: 28, C: 1}, loss.SoftmaxCrossEntropy{}, optim.NewAdamW(0.001, 0.01))
net.Fit(inputs, targets, 10)

net.SaveFile("model.json")
// Later: build + Compile the same architecture, then
net.LoadFile("model.json")
```

`Conv2D` and `MaxPool2D` treat each row as a channel-major image: `index = (channel*height + y)*width + x`. `Dropout` and `BatchNorm` switch automatically between training behavior (inside `Fit`/`FitStep`) and inference behavior (inside `Predict`).

`Embedding` keeps the current matrix-only API: each input row is a token-id sequence, and the layer concatenates the looked-up embedding vectors across columns. For example, `Compile(4, ...)` plus `NewEmbedding(vocab, 8)` turns an `Mx4` token-id matrix into an `Mx32` dense feature matrix that can feed `LayerNorm`, `GELU`, and `Dense`.

### Tokenizers

```go
import "github.com/mattn/tensai/tokenizer"

tok, err := tokenizer.Load("tokenizer.json") // the file models ship on Hugging Face
ids := tok.Encode("Hello, I'm a language model,")
text := tok.Decode(ids)
eos, _ := tok.ID("<|endoftext|>")
```

Byte-level BPE as GPT-2, Llama 3, and Qwen use it. The pre-tokenization regexes these models declare need lookahead and inline case-insensitive groups that `regexp` cannot express, so the two patterns that exist in the wild — the GPT-2 split and the cl100k-style split — are hand-written scanners, and anything else is rejected rather than silently mis-tokenized. Special tokens are matched verbatim during encode. Verified against the reference `tokenizers` library: an adversarial corpus and 2000 fuzzed strings encode and decode identically for both GPT-2 and Qwen2.5 (see `tokenizer/verify_hf.py`). An NFC normalizer passes through — input is assumed already NFC, which virtually all real-world text is.

### ONNX export

```go
import tensaionnx "github.com/mattn/tensai/encoding/onnx"

err := tensaionnx.MarshalFile("model.onnx", model)
```

Same layer support as the TFLite export (Dense, Conv2D, MaxPool2D, BatchNorm folded to Mul+Add, Dropout dropped, Softmax on dense features, and the ReLU/LeakyReLU/Sigmoid/Tanh activations), but no layout gotcha: ONNX convolutions are NCHW, which is exactly tensai's channel-major row layout, so the exported model consumes the same flattened rows tensai does, as a `[1, C, H, W]` tensor. Verified against onnxruntime to ~1e-7 relative error (see `encoding/onnx/verify_onnxruntime.py`).

### TFLite export

```go
import tensaitflite "github.com/mattn/tensai/encoding/tflite"

// after training:
err := tensaitflite.MarshalFile("model.tflite", model)
```

Supported layers: Dense, Conv2D (VALID/SAME padding), MaxPool2D, BatchNorm (folded into Mul+Add), Dropout (dropped), Softmax, and the ReLU/LeakyReLU/Sigmoid/Tanh activations. Exported convolutions follow TFLite's NHWC layout — feed the exported model NHWC input; weight reordering is handled during export. Outputs have been verified to match `Predict` to ~1e-7 relative error on the LiteRT interpreter (see `encoding/tflite/verify_litert.py`). Alias the import when combining with go-tflite, which also names its package `tflite`.

### safetensors checkpoints

`encoding/safetensors` opens the format published model weights usually ship in. Open parses only the header; each `Tensor` call reads just that tensor's bytes, so single tensors come out of multi-gigabyte checkpoints without loading the rest. F32 loads as-is and F16/BF16/F64 convert to tensai's float32:

```go
import "github.com/mattn/tensai/encoding/safetensors"

f, err := safetensors.Open("model.safetensors")
defer f.Close()
w, err := f.Tensor("model.layers.0.attention.wq.weight") // *tensai.Tensor
```

`encoding/gguf` does the same for llama.cpp's GGUF container: `Open` parses the typed metadata (`String`/`Int`/`Float`/`KV`) and the tensor directory, and each `Tensor` call reads and dequantizes just that tensor — F32/F16/BF16 plus the block-quantized Q8_0, Q4_0/Q4_1, Q5_0/Q5_1, the K-quants Q2_K through Q6_K, and the nonlinear IQ4_NL, so the whole ladder of checkpoints usually published for llama.cpp — q2_k up through q8_0 — opens directly. Dimensions come back row-major like every other reader here. One caveat inherited from the format: llama.cpp's converter permutes attention q/k projection rows into its interleaved RoPE order, which consumers pairing GGUF weights with half-split RoPE must undo.

`Names`, `Info`, and `Metadata` inspect a checkpoint without loading it; `Save`/`SaveFile` write F32 checkpoints that the reference implementation reads back bit-for-bit.

`_example/gpt2` puts the reader to work on a real model: it downloads the published GPT-2 small (124M) checkpoint from Hugging Face, loads the weights through this package, tokenizes with a from-scratch byte-level BPE, and decodes with a KV cache — every matvec running on the same `Dot` kernel as the rest of tensai, at ~30 tok/s with the AVX2 build:

```
$ GOEXPERIMENT=simd go run ./_example/gpt2 -n 20
Hello, I'm a language model, not a programming language. I'm a language model. ...
```

The greedy continuation matches GPT-2's well-known reference output token for token, which pins the whole pipeline — reader, tokenizer, and forward pass — in one check.

The prompt runs through the model as one batched pass; with `-gpu` (built with `-tags wgpu` or `wgpu24`) every block's causal multi-head attention becomes a single masked dispatch on the GPU. A 600-token prompt prefills about 2x faster even through dozen inside WSL2; native drivers gain more.

`-q8` quantizes the decode-path weights to int8 (weight-only, per-column scales) and doubles generation — 23 to 46 tok/s on the same machine — because decode streams the whole checkpoint per token and int8 pulls a quarter of the bytes. The text stays coherent but greedy decoding no longer reproduces the float32 reference tokens exactly; use the default float32 path for the reference check.

The `tensai` command does the same for modern instruction-tuned models: RMSNorm, rotary position embeddings, grouped-query attention, and a SwiGLU MLP, loaded from safetensors (config.json drives the dimensions, sharded checkpoints come through their index.json) or from a single llama.cpp GGUF that carries config, tokenizer, and weights in one file — `-model ./qwen2.5-0.5b-instruct-q8_0.gguf -q8` chats with nothing else on disk. One runtime speaks ten architectures, each contributing its own twist:

| family | models | what it adds |
|---|---|---|
| qwen2 | Qwen 1.5/2/2.5, Qwen2.5-Coder, the R1-Distill-Qwen line | attention biases |
| qwen3 | Qwen3 dense | per-head QK-norm, explicit head_dim, `-think` |
| qwen3_5 | Qwen3.5, Qwen3.6, Qwen3.8 | a gated delta rule on three layers in four, ordinary attention on the fourth; norms scale by 1 + w, RoPE turns a quarter of each head, and the queries carry a gate for the attention output. CPU only, and no `-draft` |
| llama | Llama 2/3, SmolLM2, Mistral, R1-Distill-Llama | the block everyone forked |
| smollm3 | SmolLM3-3B | RoPE skipped every fourth layer |
| gemma3 | Gemma 3 | sliding windows on 5/6 layers, sandwich norms, gelu-tanh gate, SentencePiece |
| phi3 | Phi-3/3.5-mini | q/k/v and gate/up shipped pre-fused |
| qwen2moe / qwen3moe | Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B | top-k routed experts, a shared expert on qwen2moe |
| gpt-oss | gpt-oss-20b | MXFP4 experts, attention sinks, YaRN rope, harmony channels |

The DeepSeek-R1 distills need no family of their own — they are stock qwen2/llama blocks wearing DeepSeek's turn markers, which the loader spots in the embedded chat template and switches automatically, `<think>` reasoning included. Mixture-of-experts blocks route each token through its top-k experts, repacked per expert straight from the GGUF's 3D tensors: Qwen1.5-MoE-A2.7B (14B total, 2.7B active) answers at ~9 tok/s from a 20-second load, and gpt-oss-20b — its experts kept in their native MXFP4 blocks, expanded through a one-shuffle table-lookup kernel — reasons in its harmony analysis channel and answers on the same 15GB machine.

With `-q8`/`-q4` each weight quantizes as it loads and its float32 copy dies immediately, so the full-precision model never has to fit in memory, and the layers load in parallel with the quantizer splitting columns across CPUs. Quantized GGUF checkpoints skip the float32 detour entirely: Q8_0, Q4_0, Q5_0, the whole Q4_K/Q5_K/Q6_K K-quant family, and MXFP4 repack straight from the memory-mapped file — nibbles copy raw where the grids line up, five- and six-bit spans renormalize with integer rounding under `-q4`, everything widens onto a finer int8 grid under `-q8` — keeping llama.cpp's own quantization intact. A 1.5B Q4_K_M loads in about 3 seconds instead of 8 (`-requant` restores the float detour, trading the much slower load for about 10% more decode speed from its coarser symmetric tables); a 3B Q8_0 opens in 5 seconds instead of 32.

The first `.gguf` load also writes the repacked weights to a cache file next to the model (`-nocache` opts out), and every later load just memory-maps it: the 1.5B Q4_K_M reopens in ~0.3 seconds, a Mistral 7B in well under a second, and gpt-oss-20b in under two. Beyond the instant reopen, mapped weights are clean file-backed pages the kernel can drop and re-read at will — on a machine where the model barely fits, that replaces swap thrashing with ordinary page cache behavior, which is the same structural advantage llama.cpp gets from decoding straight out of its mmap'd file.

On a 15GB machine the ladder looks like: 0.5B at ~40 tok/s with `-q8` and a 1.5B Q4_K_M at ~25 with `-q4` (the tiled integer kernels, measured on native Windows), and Qwen2.5-**7B**-Instruct — 15GB of BF16 shards, int4-quantized on the fly during a two-minute load into ~6GB resident — answering correctly at 3.5 tok/s. Prompts feed through a batched prefill: `QMatrix.MatMul` streams the weights once per block of eight token rows instead of once per token, cutting the wait before the first generated token by around 6x.

`-draft` points at a smaller same-family model for speculative decoding (greedy only): the draft proposes a few tokens, one batched pass of the big model verifies them, and rejections roll the caches back, so the output is exactly what the big model alone would produce — Qwen2.5-**7B** with the 0.5B drafting goes from 1.2 to 1.6 tok/s; the draft only pays off when the target is much larger than it. Sampling (`-temp` above 0) restricts itself to the nucleus: `-topp 0.9` keeps the smallest probability-sorted set of tokens holding 90% of the mass, so the long tail where repetition loops live never gets a lottery ticket. `-chat` turns it into a multi-turn conversation on stdin — the KV cache carries the whole dialogue, so each turn only processes its own tokens — and `-serve :8080` exposes the same model as an OpenAI-compatible `/v1/chat/completions` endpoint (messages array, SSE streaming, usage counts, and `tools` — the ChatML families answer with `tool_calls`, so an agent can drive a loop against a pure-Go model), so any OpenAI client pointed at it chats with a pure-Go model:

```
$ tensai run -q8 "What is the capital of France?"
The capital of France is Paris.
43 tokens in 1.3s (33.1 tok/s)
```

### Automatic differentiation

When a model doesn't fit the Sequential mold (weight sharing, custom losses, exotic architectures), build the computation directly and let reverse-mode autodiff derive the gradients:

```go
w1 := autograd.Param(tensai.RandomMatrix(2, 8, rng))
b1 := autograd.Param(tensai.NewMatrix(1, 8))
w2 := autograd.Param(tensai.RandomMatrix(8, 1, rng))
trainer := autograd.NewTrainer(optim.NewAdam(0.05), w1, b1, w2)

for step := 0; step < 2000; step++ {
	loss := autograd.Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).Sigmoid().MSELoss(y.Tensor())
	trainer.Step(loss) // backward + update + zero grads, returns the loss value
}
```

For manual control, the pieces are still public: `loss.Backward()`, `p.Grad()`, and `autograd.ZeroGrads(params...)`. A node's value and gradient are read through `Value()` and `Grad()` rather than fields, which is what lets a graph keep them on a GPU.

The graph a loss node holds can be visualized: `loss.ToDot()` returns Graphviz DOT (label leaves with `.Named("w1")`), so `go run ./_example/dot | dot -Tsvg > graph.svg` draws the network the same way Gorgonia's encoding/dot does.

A `Tape` recycles the buffers a step allocates — `tape.Bind(params...)` once, `tape.Reset()` after each step — which takes `_example/charrnn` from 22MB of allocation per training step to 0.75MB and about a quarter off its wall time; nothing from the finished step may be read after `Reset`. The same reuse is available one layer down through the `MatMulInto` / `AddInto` family, as `DotInto` has always offered for matrices.

Graphs are built dynamically per step (define-by-run) and are single-use. Available ops: `MatMul` (batched, with broadcast leading axes), `Add`, `Sub`, `Mul`/`MulElem`, `Div`, `Scale`, `Neg`, `AddRow`, `T`/`Transpose`, `Reshape`, `Softmax` (last axis), `Sum`, `Mean`, `SumAxis`/`MeanAxis`, `LayerNorm`, `Embed`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `Exp`, `Log`, `MSELoss`, `SoftmaxCELoss`, and `CrossEntropy`. Element-wise ops broadcast NumPy-style, and a gradient is summed back over whatever axes an operand was stretched along. Shape mismatches panic during graph construction. Every op's gradient is verified against finite differences in the test suite.

### Recurrent networks and attention

`rnn.Cell`, `rnn.LSTMCell`, and `rnn.SelfAttention` are built on the autograd engine, so unrolling a sequence is a plain Go loop and backpropagation through time comes for free:

```go
cell := rnn.NewLSTMCell(inSize, hidden, rng)
wOut := autograd.Param(tensai.RandomMatrix(hidden, numClasses, rng))
bOut := autograd.Param(tensai.NewMatrix(1, numClasses))
trainer := autograd.NewTrainer(optim.NewAdam(0.01), append(cell.Params(), wOut, bOut)...)

for step := 0; step < epochs; step++ {
	h, c := cell.InitState(batch)
	for _, x := range steps { // one (batch x inSize) matrix per time step
		h, c = cell.Step(autograd.Input(x), h, c)
	}
	logits := h.MatMul(wOut).AddRow(bOut)
	trainer.Step(logits.CrossEntropy(labels)) // labels is a []int of class indices
}
```

`rnn.SelfAttention` operates on one `(seqLen x inSize)` sequence node: `attn.Forward(x)` computes `softmax(Q*K^T/sqrt(d))*V` with learned projections; the raw `rnn.Attention(q, k, v)` form is also exposed.

Batches and heads are written directly on the n-dimensional engine instead: the per-head split is a `Reshape` plus a `Transpose`, and every score in the batch is one `MatMul`.

```go
heads := func(t *autograd.Node) *autograd.Node { // (batch, seq, model) -> (batch, head, seq, headDim)
	return t.Reshape(batch, seq, nHeads, headDim).Transpose(0, 2, 1, 3)
}
q, k, v := heads(x.MatMul(wq)), heads(x.MatMul(wk)), heads(x.MatMul(wv))
att := q.MatMul(k.T()).Scale(scale).Add(causalMask).Softmax() // (batch, head, seq, seq)
y := att.MatMul(v).Transpose(0, 2, 1, 3).Reshape(batch, seq, model).MatMul(wo)
```

`_example/tinygpt` is that block inside a working character-level transformer — token and position embeddings, two pre-norm blocks, a GELU feed-forward, and next-character cross-entropy. 106k parameters, about a minute of training with the AVX2 build, after which it writes the corpus back:

```
$ GOEXPERIMENT=simd go run ./_example/tinygpt
corpus: 1660 chars, vocab: 43, parameters: 106496
iter    1: loss=4.7166
iter 1000: loss=0.2258

generated:
Alice was beginning to get very tired of sitting by her sister on the bank, and look, aving nothing to do: once or twice coat-pocket, and looker a with ouble of of getting up and picgung to a daisies, when suddenly  a White Rabbit with pink eyes  ran close by her.
```

Autograd parameters are saved and restored positionally with `autograd.SaveParamsFile("cell.json", cell.Params()...)` / `autograd.LoadParamsFile("cell.json", cell.Params()...)` — build the same cell, then load.

### N-d tensors: broadcasting and batched MatMul

`Tensor` generalizes `Matrix` to any rank. Element-wise ops broadcast NumPy-style, and `MatMul` multiplies whole stacks of matrices at once — the leading batch axes broadcast too, so a shared 2-D weight applies to every sequence in a batch in one call:

```go
x := tensai.NewTensor(4, 6, 3)                    // (batch, position, channel)
mean, _ := tensai.NewTensorFromSlice([]float32{0.5, -1, 2}, 3)
centered, _ := x.Sub(mean)                        // (4,6,3) - (3)   -> (4,6,3)
h, _ := tensai.MatMul(centered, w)                // (4,6,3) @ (3,8) -> (4,6,8)

kt, _ := k.Transpose()                            // swap the last two axes
scores, _ := tensai.MatMul(q, kt)                 // (4,6,8) @ (4,8,6) -> (4,6,6)
scores.Scale(1 / float32(math.Sqrt(8)))
out, _ := tensai.MatMul(scores, v)                // attention for the whole batch
```

Tensors are contiguous and row-major; `Reshape` (with `-1` inference) and the `Matrix`/`Tensor` conversions are zero-copy views, while `Transpose` accepts an arbitrary axis permutation and materializes the result. See `_example/tensor` for the runnable version.

### GPU MatMul over WebGPU (experimental)

Building with `-tags wgpu` (linux/darwin/windows) enables a GPU backend for batched `MatMul` with the same shape and broadcasting semantics as the CPU version:

```go
dev, err := gpu.Open() // fails cleanly when no GPU / library is present
if err != nil { /* fall back to tensai.MatMul */ }
defer dev.Close()
fmt.Println(dev.Name()) // e.g. "AMD Radeon 780M (integrated)"
out, err := dev.MatMul(a, b)
```

On machines with both an integrated and a discrete GPU, pass a preference: `gpu.Open(gpu.LowPower)` steers to the iGPU, `gpu.HighPerformance` to the dGPU (it is a hint — with a single adapter you always get that one).

Buffers can also stay resident on the GPU, so a weight rides the bus once instead of on every call and intermediates never leave the device:

```go
gw, _ := dev.Upload(w)              // weight uploaded once
defer gw.Free()                     // GPU memory is not garbage collected
gx, _ := dev.Upload(x)
h, _ := gx.MatMul(gw)               // chain freely; nothing touches the host
out, _ := h.MatMul(gw2)
result, _ := out.Download()         // one readback at the end
```

`dev.MatMul(a, b)` is shorthand for Upload → MatMul → Download → Free. Residency matters most on discrete GPUs, where every transfer crosses PCIe; on shared-memory iGPUs the win is smaller and comes mainly from skipping intermediate readbacks.

Beyond MatMul, resident tensors support `MatMulT` (multiply by a transposed operand without materializing the transpose), an in-place `Scale`, and a row-parallel `Softmax` over the last axis — enough to run single-head attention entirely on the GPU:

```go
out, _ := gq.Attention(gk, gv)                 // softmax(q@k^T/sqrt(d))@v, no host round-trips
out, _ = gq.MultiHeadAttention(gk, gv, heads)  // packed (batch, seq, heads*dh) layout
```

Multi-head attention carves each head out of the packed layout with strided kernels — the matmul kernels take explicit row strides and per-batch offsets — so no permute is ever materialized. The causal variants (`CausalAttention`, `CausalMultiHeadAttention`) mask future positions inside the kernel, with k and v allowed to hold more positions than q — the prompt-prefill and chunked-decode patterns of autoregressive models — so no mask tensor is ever built either. `CausalMultiHeadAttention` runs as one fused flash-attention-style dispatch (an online softmax over kv tiles, for head dimensions up to 128): the scores matrix never exists, so memory stays at q+k+v+output regardless of sequence length, and shapes whose scores would blow past the device's storage-buffer limit — batch 8, heads 8, seq 1024 is a 256MiB scores matrix — just run.

There is no cgo involved: the bindings load the [wgpu-native](https://github.com/gfx-rs/wgpu-native) shared library at runtime via `ebitengine/purego` (`dlopen` on linux/macOS, `LoadLibrary` on Windows). Download a **v22.1.0.5** release binary (the C API these bindings target), then either install it where the loader finds it or point `TENSAI_WGPU_LIB` at it:

```bash
curl -sLO https://github.com/gfx-rs/wgpu-native/releases/download/v22.1.0.5/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip -d wgpu
TENSAI_WGPU_LIB=$PWD/wgpu/lib/libwgpu_native.so go test -tags wgpu ./...
```

On Windows, take `wgpu-windows-x86_64-msvc-release.zip` from the same release and point the variable at the `wgpu_native.dll` inside it (any `wgpu_native.dll` on `PATH` or next to the executable is found without the variable):

```powershell
$env:TENSAI_WGPU_LIB="$PWD\wgpu\lib\wgpu_native.dll"
go run -tags wgpu ./_example/wgpu
```

`_example/wgpu -sweep` walks a ladder of sizes and marks where the GPU overtakes the CPU kernel. It reports `gpu+xfer` for the convenient Upload → MatMul → Download call and `resident` for inputs uploaded once and reused (the final result is still downloaded each iteration). Because the CPU side is the same `dotRows` kernel the rest of the package uses, building the example twice compares portable Go, AVX2, and both GPU usage patterns:

```bash
GOEXPERIMENT=nosimd go build -tags wgpu -o wgpu-nosimd ./_example/wgpu
GOEXPERIMENT=simd   go build -tags wgpu -o wgpu-simd   ./_example/wgpu
./wgpu-nosimd -sweep && ./wgpu-simd -sweep
```

The crossover moves with the CPU kernel, GPU driver, and transfer pattern. The `res/cpu` column and crossover marker use the resident-input timing, since that is the normal pattern for repeated inference. On a Ryzen iGPU (AMD Radeon 780M, native Windows, AVX2 CPU kernel) the register-tiled kernels put every rung of the ladder on the GPU side:

```
             shape                   MFLOP   gpu+xfer   resident        cpu   res/cpu
mnist dense  1x100x784@784x128        20.1     1.51ms      597µs      652µs     1.09x
mnist conv2  1x19600x72@72x16         45.2    1.388ms      763µs    2.354ms     3.09x
tiny         1x128x128@128x128         4.2      432µs      302µs      410µs     1.36x
small        1x512x512@512x512       268.4    1.331ms    1.216ms    6.865ms     5.65x
medium       8x512x512@512x512      2147.5    8.053ms    6.297ms    71.56ms    11.36x
large        32x512x512@512x512     8589.9    86.71ms   28.856ms  266.277ms     9.23x
huge         64x512x512@512x512    17179.9  116.374ms   62.128ms  566.726ms     9.12x
```

Arithmetic no longer dominates the convenient path — `gpu+xfer` at `large` spends two thirds of its time on the bus — which is exactly what keeping inputs resident is for. Through a translation layer like dozen inside WSL2 the ratios shrink to roughly parity-to-3x, and on CPU Vulkan implementations the GPU path loses outright; measure on the driver you will ship on.

Quantized weights stay quantized on the device too: `UploadQ8` packs a `QMatrix` four int8 weights per u32, and `gpu.QMatrix.MatMul` dequantizes them in registers, so a decode matvec — whose cost is streaming the weights — moves a quarter of the f32 bytes. On the same iGPU through dozen it runs the matvec 2.2x faster than the resident f32 kernel.

`UploadQ4` does the same for the int4 twin — nibbles packed four row-pair bytes per u32, group scales folded at group boundaries in registers — so `-q4 -gpu` runs models whose int8 weights would not fit. The rest of a transformer decode step is there as well — `RMSNorm`, in-place `RoPE`, `Add`, `SiluMul`, `GroupedCausalAttention` (a KV cache packing fewer heads than the queries, read up to a valid length), and `CopyRowsInto` to append fresh k/v rows to a resident cache — so `tensai run -q8 -gpu` runs every block on the device and only the hidden state comes back per token. `BeginBatch`/`Flush` record a whole token's dispatches into one submission, and freed intermediates recycle through a buffer pool, which together took a dozen-translated decode from 1.2 to ~17 tok/s steady state on the machine above. The vec4-staged integer GEMM and the scalar-state attention kernel then lifted the same path's prefill from ~840 to ~1800 t/s on a 625-token prompt — about 87% of llama.cpp's Vulkan backend on the same GPU — with decode at ~20 tok/s. On native Windows the same iGPU speaks D3D12 directly, and `-q8 -gpu` held the 0.5B decode crown for a while — 29.7 tok/s against 23.2 on the AVX2 path — until the tiled integer kernels took it back: the CPU now decodes the same model at ~42 tok/s against ~30 on the GPU, which stays useful for keeping the cores free.

wgpu-native picks Vulkan on Linux, Vulkan or D3D12 on Windows, and Metal on macOS, so AMD, Intel, Apple, and NVIDIA GPUs all work — as do CPU Vulkan implementations like lavapipe, which is how the tests run on machines without a GPU. `gpu.MatMul` uploads the operands and reads the product back on every call; `Upload` plus `gpu.Tensor.MatMul` keeps inputs and intermediates resident, so only the final result needs to cross the bus. Without the build tag `gpu.Open` returns an error and nothing else changes.

#### `-tags wgpu24`: the new wgpu-native API, and the real GPU inside WSL2

`-tags wgpu24` (linux/darwin/windows) builds the same `gpu.Open` API against the reworked wgpu-native C API instead — pair it with a **v29-series** release binary. The new API's payoff is `WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter`, which un-hides non-conformant Vulkan drivers. Concretely: Mesa's dozen (Vulkan-on-D3D12, shipped in the kisak-mesa PPA) exposes the real host GPU inside WSL2, but the v22 API hides it as non-conformant and falls back to lavapipe; the wgpu24 build reaches it:

```bash
VK_DRIVER_FILES=/path/to/dzn_icd.json \
TENSAI_WGPU_LIB=$PWD/wgpu29/lib/libwgpu_native.so \
    go run -tags wgpu24 ./_example/wgpu   # adapter: Microsoft Direct3D12 (AMD Radeon(TM) Graphics)
```

The new API passes structs by value. Every one of them is reached through a pointer field except the three callback-info arguments, and those are the only per-OS code in the binding: `wgpu24_callinfo.go` hands the 40-byte struct to SysV/AAPCS in registers, while `wgpu24_callinfo_windows.go` passes its address, because the Windows x64 convention already defines any aggregate that is not 1, 2, 4, or 8 bytes wide as passed by reference. `WGPUFuture` results come back in RAX either way. When both tags are set, `wgpu24` wins.

On Windows, pair it with `wgpu-windows-x86_64-msvc-release.zip` from the same v29 release:

```powershell
$env:TENSAI_WGPU_LIB="$PWD\wgpu29\lib\wgpu_native.dll"
go run -tags wgpu24 ./_example/wgpu
```

Note that new does not mean faster: on a Radeon 780M at `32x512x512@512x512` the v22 library runs the same shader in 85ms and the v29 one in 165ms (D3D12 190ms, Vulkan 438ms when forced with `WGPU_BACKEND`). Use `wgpu24` for the adapters it reaches, not for speed.

## Run

```bash
go run ./_example/helloworld
go run ./_example/dataset
go run ./_example/xor
go run ./_example/fizzbuzz
go run ./_example/spiral
go run ./_example/iris
go run ./_example/charrnn
GOEXPERIMENT=simd go run ./_example/tinygpt      # trains a small transformer, ~1 minute
go run ./_example/plasma
go run ./_example/tensor
GOEXPERIMENT=simd go run ./_example/gpt2          # downloads the GPT-2 checkpoint (~550MB) on first run

# The tensai command runs instruction-tuned models (~1GB downloaded on first run):
GOEXPERIMENT=simd go install ./cmd/tensai
tensai run -q8 "What is the capital of France?"
tensai chat -q8 -model ./model.gguf
tensai serve -q8 -addr :8080                      # OpenAI-compatible API
GOEXPERIMENT=simd go run -tags wgpu24 ./cmd/tensai bench -q8   # CPU vs GPU
go run -tags wgpu ./_example/wgpu          # needs wgpu-native, see above
go run -tags wgpu ./_example/wgpu -sweep  # GPU vs CPU across sizes
go test ./...

# With the AVX2 SIMD kernel (Go 1.26+ / 1.27, amd64):
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -bench=Dot .
```

The MNIST example downloads the standard IDX gzip files into `_example/mnist/data` when they are missing. Set `MNIST_DIR` to use another cache directory, and pass `-model cnn` for the convolutional variant (Conv2D/MaxPool2D/Dropout + AdamW); both trained variants finish by saving the model and re-scoring it after a reload. `-model knn` runs the no-training k-NN baseline instead — on the 5000-sample subset it scores ~91% against ~92% for the MLP and ~95% for the CNN:

```bash
go run ./_example/mnist
go run ./_example/mnist -model cnn
go run ./_example/mnist -model knn
go run ./_example/mnist -model cnn -export mnist.tflite
MNIST_DIR=/path/to/mnist go run ./_example/mnist
```

`-export` writes the trained model as a TFLite flatbuffer (the exported CNN scores identically on the LiteRT interpreter). MNIST is single-channel, so images feed the exported model unchanged; consume it from Go with [go-tflite](https://github.com/mattn/go-tflite):

```go
model := tflite.NewModelFromFile("mnist.tflite")
interpreter := tflite.NewInterpreter(model, nil)
interpreter.AllocateTensors()
copy(interpreter.GetInputTensor(0).Float32s(), image) // 28*28 floats, NHWC
interpreter.Invoke()
scores := interpreter.GetOutputTensor(0).Float32s() // 10 logits
```

The charrnn example trains a character-level LSTM on an embedded public-domain text, saves the parameters with `SaveParamsFile`, restores them into a fresh model, and generates a sample from the reloaded parameters.

The plasma example animates a demoscene-style plasma in the terminal where the plasma function is a randomly weighted network (a CPPN) evaluated for every pixel of every frame as one batch. The status line shows the per-frame network time, which makes it a live SIMD benchmark: 120x90 pixels runs at ~32 fps on the portable build and ~100 fps with `GOEXPERIMENT=simd` on the same machine. Try different `-seed` values for different effects.

Both raw IDX files and `.gz` variants are accepted.

## SIMD Coverage

Where the AVX2 kernels apply today, and where they still could:

- [x] Matmul (`Dot`/`DotInto`) — used by `Dense`, `Conv2D` (im2col product), `knn.Classifier` distances, and autograd `MatMul`
- [x] ReLU / LeakyReLU forward & backward
- [x] Sigmoid / Tanh forward & backward (vectorized polynomial `exp`)
- [x] GELU forward & backward (vectorized `erf`)
- [x] LayerNorm forward & backward (vector row reductions)
- [x] Softmax / SoftmaxCrossEntropy exponentials and scaling
- [x] Adam / AdamW parameter update
- [x] SGD update (momentum form, same fused multiply-add loop as Adam)
- [x] Slice add & scale primitives (bias add, `Embedding` gradient scatter-add)
- [x] Transpose-free gradient matmul (`DotTAInto`) — `Dense`/`Conv2D` weight gradients no longer materialize `input^T` / `im2col^T`
- [x] Remaining transposes (`T`/`TInto`, now only weight matrices and autograd) — cache-blocked 32x32 tiles
- [x] Softmax backward row dot products (autograd) — fused AVX2 dot and Jacobian-vector accumulation
- [ ] MSE / BinaryCrossEntropy losses (BCE needs a vectorized `log`)
- [ ] Autograd element-wise backward passes (gradients accumulate with `+=`, so they need dedicated fused kernels)
- [ ] BatchNorm statistics (column-strided access needs a restructure)
- [ ] MaxPool2D window scan
- [ ] im2col / col2im gather-scatter (contiguous runs could use bulk copies)

The unchecked items are ordered roughly by expected impact; none of them show up prominently in training profiles today.

## Design Notes

- All operations are batched. Inputs are `MxN` matrices, where `M` is the batch size and `N` is the feature dimension.
- `Embedding` inputs are also matrices: values must be exact integer token ids stored in `Float`, and the embedding vectors are flattened across the row.
- The `Layer` interface standardizes `Forward`, `Backward`, `Params`, and `Grads`, which keeps new layers such as convolution or dropout straightforward to add.
- `Dense` weights use Glorot/He-style initialization to keep early training stable.
- `SoftmaxCrossEntropy` subtracts the row maximum before softmax for numerical stability.

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)

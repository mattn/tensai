# tensai

<p align="center"><img src="assets/logo.svg" width="420" alt="tensai"></p>

<p align="center"><a href="https://github.com/mattn/tensai/releases/latest"><img src="https://img.shields.io/github/v/release/mattn/tensai" alt="latest release"></a> <a href="https://pkg.go.dev/github.com/mattn/tensai"><img src="https://pkg.go.dev/badge/github.com/mattn/tensai.svg" alt="Go Reference"></a></p>

**tensai** is a small machine-learning framework for learning and experiments, written in pure Go. It implements forward passes, backpropagation, and optimization with no external dependencies in the default build — no cgo, no assembly files, no C compiler.

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

Despite its size, tensai reaches surprisingly far: the same kernels that train a XOR network run the published GPT-2 checkpoint, chat with Qwen2.5 and Gemma 3, decode llama.cpp GGUF quantizations block-exactly, and serve an OpenAI-compatible API — all in pure Go.

## Highlights

- **Matrices and N-d tensors** — float32 `Matrix` and rank-N `Tensor` with NumPy-style broadcasting, batched `MatMul`, zero-copy `Reshape` and views
- **Layers** — `Embedding`, `Dense`, `Conv2D`, `MaxPool2D`, `BatchNorm`, `LayerNorm`, `Dropout`, plus `ReLU`, `LeakyReLU`, `GELU`, `Sigmoid`, `Tanh`, and `Softmax`
- **Training** — `Sequential` models with `Compile` → `Fit` / `FitStep` → `Predict`, three loss functions, momentum `SGD` / `Adam` / `AdamW`, and dataset utilities; a full MLP step runs in ~29 allocations
- **Autograd** — a micrograd-style reverse-mode engine over matrices, with `rnn.Cell`, `rnn.LSTMCell`, and `rnn.SelfAttention` built on top; backpropagation through time is a plain Go loop
- **SIMD acceleration** — AVX2 kernels written with Go's experimental `simd/archsimd` package; build with `GOEXPERIMENT=simd`, and every other build uses the portable fallbacks automatically
- **WebGPU backend** — `-tags wgpu` runs batched `MatMul`, attention, and a full quantized transformer decode step on any GPU wgpu-native reaches, through `purego` with no cgo
- **int8 / int4 quantization** — weight-only quantized matmuls that reach memory bandwidth, plus MXFP4 for gpt-oss
- **Model formats** — TFLite and ONNX export, safetensors read/write, and a GGUF reader covering the K-quants — all with in-tree encoders, still no dependencies
- **Tokenizers** — Hugging Face `tokenizer.json` byte-level BPE (GPT-2, cl100k, o200k families) and SentencePiece, verified to match the reference implementations exactly
- **LLM inference** — `_example/gpt2`, `_example/qwen` (nine model families), and the `tensai` command with `run`, `chat`, and `serve` subcommands

## Where to go next

- [Getting Started](getting-started.md) — install and train your first model
- [Guide](guide/tensors.md) — tensors, layers, training, autograd, quantization, SIMD, GPU
- [Model Formats](formats.md) — TFLite, ONNX, safetensors, GGUF
- [LLM Inference](llm.md) — run real language models in pure Go
- [Examples](examples.md) — fifteen runnable examples, from hello-world to a 7B chat model

## Design notes

- All operations are batched: inputs are `MxN` matrices, where `M` is the batch size and `N` is the feature dimension
- The `Layer` interface standardizes `Forward`, `Backward`, `Params`, and `Grads`, which keeps new layers straightforward to add
- `Dense` weights use Glorot/He-style initialization to keep early training stable
- `SoftmaxCrossEntropy` subtracts the row maximum before softmax for numerical stability

## License

MIT — [Yasuhiro Matsumoto (a.k.a. mattn)](https://github.com/mattn)

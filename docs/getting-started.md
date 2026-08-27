# Getting Started

## Install

```bash
go get github.com/mattn/tensai
```

The default build has no external dependencies. Go 1.26 or later is recommended: on amd64 it unlocks the AVX2 SIMD kernels via `GOEXPERIMENT=simd` (both the Go 1.26 and 1.27 `simd` APIs are supported through build tags). Every other platform and build uses the portable fallbacks automatically — same results, just slower.

```bash
# portable build, any platform
go build ./...

# AVX2-accelerated build (amd64, Go 1.26+)
GOEXPERIMENT=simd go build ./...
```

## Your first model: XOR

Learning XOR is the "hello world" of neural networks — it is the smallest problem a linear model cannot solve. Inputs are `MxN` matrices where `M` is the batch size, so the four XOR cases are one `4x2` matrix:

```go
package main

import (
	"fmt"

	"github.com/mattn/tensai"
)

func main() {
	model := tensai.NewSequential()
	model.Add(tensai.NewDense(8))
	model.Add(&tensai.Tanh{})
	model.Add(tensai.NewDense(1))
	model.Add(&tensai.Sigmoid{})

	// Compile(inputSize, loss, optimizer)
	if err := model.Compile(2, tensai.MeanSquaredError{}, tensai.NewAdam(0.05)); err != nil {
		panic(err)
	}

	// XOR truth table: 4 samples, 2 features each.
	inputs, _ := tensai.NewMatrixFromSlice(4, 2, []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	targets, _ := tensai.NewMatrixFromSlice(4, 1, []float32{0, 1, 1, 0})

	if err := model.Fit(inputs, targets, 5000); err != nil {
		panic(err)
	}

	pred, _ := model.Predict(inputs)
	for r := 0; r < inputs.Rows; r++ {
		fmt.Printf("[%g %g] -> %.4f\n", inputs.At(r, 0), inputs.At(r, 1), pred.At(r, 0))
	}
}
```

## Classification

For multi-class problems, make the last `Dense` as wide as the number of classes and switch the loss to `SoftmaxCrossEntropy`:

```go
model := tensai.NewSequential()
model.Add(tensai.NewDense(8))
model.Add(&tensai.ReLU{})
model.Add(tensai.NewDense(2)) // output width = number of classes

model.Compile(2, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.05))
```

`SoftmaxCrossEntropy` expects targets as an `Mx1` matrix of class indices. Softmax is applied inside the loss, so `Predict` returns raw logits — use argmax for classification.

## Run the examples

The repository ships fifteen runnable examples, from a two-line autograd hello-world up to real LLM inference:

```bash
go run ./_example/xor
go run ./_example/mnist -model cnn
GOEXPERIMENT=simd go run ./_example/gpt2      # downloads GPT-2 (~550MB) on first run
GOEXPERIMENT=simd go run ./_example/qwen -q8  # downloads Qwen2.5-0.5B (~1GB) on first run
```

See [Examples](examples.md) for the full list.

## Build flags at a glance

| Flag | Effect |
|---|---|
| *(none)* | Portable pure-Go build, zero dependencies |
| `GOEXPERIMENT=simd` | AVX2 kernels on amd64 (Go 1.26/1.27) |
| `-tags wgpu` | WebGPU backend via wgpu-native **v22.1.0.5** (adds one cgo-free dependency: `ebitengine/purego`) |
| `-tags wgpu24` | Same API against the reworked wgpu-native C API (**v29** series); reaches non-conformant adapters such as Mesa dozen inside WSL2 |

See [SIMD Acceleration](guide/simd.md) and [GPU (WebGPU)](guide/gpu.md) for details.

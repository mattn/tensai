<p align="center"><img src="logo.svg" width="420" alt="tensai"></p>

# tensai - a tiny machine-learning framework in Go

`tensai` is a small machine-learning framework for learning and experiments. It implements forward passes, backpropagation, and optimization in pure Go; the default build has no external dependencies (the optional `wgpu` build tag adds exactly one, cgo-free: `ebitengine/purego`).

## Features

- **Matrix operations** - `Matrix` plus basic operations such as `Dot`, `Add`, `T`, and `AddBias`. Tensors are float32 (`tensai.Float`)
- **N-dimensional tensors** - `Tensor` generalizes `Matrix` to any rank: element-wise `Add`/`Sub`/`Mul`/`Div` with NumPy-style broadcasting, batched `MatMul` (the leading axes broadcast, the per-matrix products run on the same kernel as `Dot`, parallelized across the batch), axis-permuting `Transpose`, `Reshape` with `-1` inference, and zero-copy views to and from `Matrix`
- **SIMD acceleration** - AVX2 kernels written with Go's experimental `simd/archsimd` package: still pure Go, no cgo, no assembly files. Matmul, ReLU/LeakyReLU, Sigmoid/Tanh/Softmax (via a vectorized polynomial `exp`), GELU (via a vectorized `erf`), LayerNorm, and the Adam update are all 8-lane vectorized. Build with `GOEXPERIMENT=simd` on amd64 (Go 1.26 and 1.27 APIs both supported via build tags); every other build uses the portable fallbacks automatically
- **Low-allocation training** - layers reuse their forward/backward scratch buffers across training steps (a full MLP step runs in ~29 allocations), so GC stays out of the training loop; `Predict` always returns freshly allocated results
- **Layers** - `Embedding`, `Dense`, `Conv2D`, `MaxPool2D`, `BatchNorm`, `LayerNorm`, `Dropout`, plus `ReLU`, `LeakyReLU`, `GELU`, `Sigmoid`, `Tanh`, and `Softmax` activations
- **WebGPU backend (experimental)** - build with `-tags wgpu` (linux, macOS, Windows) and `OpenGPU()` runs batched `MatMul` as a WGSL compute shader on any GPU wgpu-native reaches (Vulkan, Metal, D3D12 — AMD, Intel, Apple, NVIDIA). The bindings go through `ebitengine/purego`, so there is still no cgo and no C compiler: the wgpu-native shared library is dlopen-ed at runtime
- **Loss functions** - `MeanSquaredError` for regression, `SoftmaxCrossEntropy` for multi-class classification, and `BinaryCrossEntropy` for binary targets
- **Optimizers** - momentum `SGD`, `Adam`, and `AdamW` (decoupled weight decay)
- **k-NN baseline** - a `KNN` classifier whose distance matrix runs on the same SIMD matmul kernel; useful as a no-training baseline next to the networks
- **Dataset utilities** - `Dataset` pairs inputs with targets and provides `Shuffle`, train/test `Split` (copy-free views), buffer-reusing mini-batch iteration with `Batches`, and `Standardize`/`StandardizeWith`
- **Sequential models** - stack layers and run `Compile` -> `Fit` / `FitStep` -> `Predict`
- **Automatic differentiation** - a micrograd-style reverse-mode autograd engine over matrices (`Param` / `Input` / `Backward`), for models that don't fit the Sequential mold; `ToDot` renders the computation graph for Graphviz
- **Recurrence and attention** - `RNNCell`, `LSTMCell`, and single-head `SelfAttention` built on the autograd engine, with backpropagation through time handled automatically
- **Serialization** - `Save`/`Load` (and `SaveFile`/`LoadFile`) round-trip trained Sequential parameters as JSON, including BatchNorm running statistics; `SaveParams`/`LoadParams` do the same for autograd parameters (RNN/LSTM/attention cells)
- **TFLite export** - the `encoding/tflite` package marshals Sequential models (FP32, NHWC) into `.tflite` flatbuffers that run on the TFLite/LiteRT runtimes and [go-tflite](https://github.com/mattn/go-tflite), with the FlatBuffers writer implemented in-tree — still no dependencies

## Layout

```
go.mod              Module definition (github.com/mattn/tensai)
tensor.go           Matrix and vector operations
ndtensor.go         N-d Tensor: broadcasting element-wise ops, batched MatMul
wgpu.go             WebGPU MatMul backend via purego + wgpu-native (build tag wgpu)
dot_simd.go         AVX2 matmul kernel (GOEXPERIMENT=simd, amd64)
dot_generic.go      Portable matmul kernel (all other builds)
kernels.go          Scalar bodies of the element-wise kernels
mathvec_simd.go     AVX2 element-wise kernels incl. vectorized exp
mathvec_generic.go  Portable element-wise kernel dispatchers
layer.go            Layer interface plus Dense and activations
conv.go             Conv2D and MaxPool2D layers
batchnorm.go        BatchNorm layer
dropout.go          Dropout layer
loss.go             Loss functions (MSE, SoftmaxCrossEntropy, BCE)
optimizer.go        Optimizers (SGD, Adam, AdamW)
model.go            Sequential model and training loop
autograd.go         Reverse-mode automatic differentiation (Node graph)
rnn.go              RNNCell / LSTMCell / SelfAttention on the autograd engine
serialize.go        Model parameter save/load (JSON)
tensai_test.go      Unit tests plus XOR convergence test
features_test.go    Gradient checks and tests for the newer layers
_example/helloworld Smallest possible program: add two values on the graph
_example/dataset    Dataset workflow: shuffle, split, standardize, batches
_example/xor        Runnable XOR training example
_example/fizzbuzz   Runnable FizzBuzz classification example
_example/spiral     Runnable 3-class spiral classification example
_example/iris       Runnable Iris classification example
_example/mnist      Runnable MNIST classifier (-model dense, cnn, or knn) with save/load
_example/charrnn    Character-level LSTM text generation on the autograd engine
_example/plasma     Demoscene-style terminal plasma rendered by a neural network
_example/dot        Graphviz DOT export of the z = x + y graph
_example/tensor     Tour of the n-d Tensor: broadcasting, batched MatMul, attention
_example/wgpu       WebGPU MatMul: adapter info, CPU cross-check, GPU vs CPU sweep
```

## Usage

### Regression: learn XOR with MSE

```go
model := tensai.NewSequential()
model.Add(tensai.NewDense(8))
model.Add(&tensai.Tanh{})
model.Add(tensai.NewDense(1))
model.Add(&tensai.Sigmoid{})

model.Compile(2, tensai.MeanSquaredError{}, tensai.NewAdam(0.05))
model.Fit(inputs, targets, 5000)

pred, _ := model.Predict(inputs)
```

### Classification: softmax + cross-entropy

```go
model := tensai.NewSequential()
model.Add(tensai.NewDense(8))
model.Add(&tensai.ReLU{})
model.Add(tensai.NewDense(2)) // output width = number of classes

model.Compile(2, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.05))
```

`SoftmaxCrossEntropy` expects targets as an `Mx1` matrix of class indices. Softmax is applied inside the loss, so `Predict` returns raw logits. Use argmax for classification.

### Datasets

```go
ds, _ := tensai.NewDataset(inputs, targets)
ds.Shuffle(rng)
train, test, _ := ds.Split(0.2)          // views, no copying
mean, std := train.Standardize()         // fit on train...
test.StandardizeWith(mean, std)          // ...apply to test

for epoch := 0; epoch < epochs; epoch++ {
	train.Batches(32, rng, func(in, tgt *tensai.Matrix) error {
		_, err := model.FitStep(in, tgt)
		return err
	})
}
```

### Convolution, regularization, and saving

```go
model := tensai.NewSequential()
model.Add(tensai.NewConv2D(28, 28, 1, 8, 3, 1, 1)) // inH, inW, inC, outC, kernel, stride, pad
model.Add(&tensai.ReLU{})
model.Add(tensai.NewMaxPool2D(28, 28, 8, 2))
model.Add(tensai.NewDense(64))
model.Add(tensai.NewBatchNorm())
model.Add(tensai.NewLeakyReLU(0.01))
model.Add(tensai.NewDropout(0.3))
model.Add(tensai.NewDense(10))

model.Compile(28*28, tensai.SoftmaxCrossEntropy{}, tensai.NewAdamW(0.001, 0.01))
model.Fit(inputs, targets, 10)

model.SaveFile("model.json")
// Later: build + Compile the same architecture, then
model.LoadFile("model.json")
```

`Conv2D` and `MaxPool2D` treat each row as a channel-major image: `index = (channel*height + y)*width + x`. `Dropout` and `BatchNorm` switch automatically between training behavior (inside `Fit`/`FitStep`) and inference behavior (inside `Predict`).

`Embedding` keeps the current matrix-only API: each input row is a token-id sequence, and the layer concatenates the looked-up embedding vectors across columns. For example, `Compile(4, ...)` plus `NewEmbedding(vocab, 8)` turns an `Mx4` token-id matrix into an `Mx32` dense feature matrix that can feed `LayerNorm`, `GELU`, and `Dense`.

### TFLite export

```go
import tensaitflite "github.com/mattn/tensai/encoding/tflite"

// after training:
err := tensaitflite.MarshalFile("model.tflite", model)
```

Supported layers: Dense, Conv2D (VALID/SAME padding), MaxPool2D, BatchNorm (folded into Mul+Add), Dropout (dropped), Softmax, and the ReLU/LeakyReLU/Sigmoid/Tanh activations. Exported convolutions follow TFLite's NHWC layout — feed the exported model NHWC input; weight reordering is handled during export. Outputs have been verified to match `Predict` to ~1e-7 relative error on the LiteRT interpreter (see `encoding/tflite/verify_litert.py`). Alias the import when combining with go-tflite, which also names its package `tflite`.

### Automatic differentiation

When a model doesn't fit the Sequential mold (weight sharing, custom losses, exotic architectures), build the computation directly and let reverse-mode autodiff derive the gradients:

```go
w1 := tensai.Param(tensai.RandomMatrix(2, 8, rng))
b1 := tensai.Param(tensai.NewMatrix(1, 8))
w2 := tensai.Param(tensai.RandomMatrix(8, 1, rng))
trainer := tensai.NewTrainer(tensai.NewAdam(0.05), w1, b1, w2)

for step := 0; step < 2000; step++ {
	loss := tensai.Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).Sigmoid().MSELoss(y)
	trainer.Step(loss) // backward + update + zero grads, returns the loss value
}
```

For manual control, the pieces are still public: `loss.Backward()`, `p.Grad`, and `tensai.ZeroGrads(params...)`.

The graph a loss node holds can be visualized: `loss.ToDot()` returns Graphviz DOT (label leaves with `.Named("w1")`), so `go run ./_example/dot | dot -Tsvg > graph.svg` draws the network the same way Gorgonia's encoding/dot does.

Graphs are built dynamically per step (define-by-run) and are single-use. Available ops: `MatMul`, `Add`, `Sub`, `MulElem`, `Scale`, `AddRow`, `T`, `Softmax`, `ReLU`, `Sigmoid`, `Tanh`, `Sum`, `Mean`, `MSELoss`, and `SoftmaxCELoss`. Shape mismatches panic during graph construction. Every op's gradient is verified against finite differences in the test suite.

### Recurrent networks and attention

`RNNCell`, `LSTMCell`, and `SelfAttention` are built on the autograd engine, so unrolling a sequence is a plain Go loop and backpropagation through time comes for free:

```go
cell := tensai.NewLSTMCell(inSize, hidden, rng)
wOut := tensai.Param(tensai.RandomMatrix(hidden, numClasses, rng))
bOut := tensai.Param(tensai.NewMatrix(1, numClasses))
trainer := tensai.NewTrainer(tensai.NewAdam(0.01), append(cell.Params(), wOut, bOut)...)

for step := 0; step < epochs; step++ {
	h, c := cell.InitState(batch)
	for _, x := range steps { // one (batch x inSize) matrix per time step
		h, c = cell.Step(tensai.Input(x), h, c)
	}
	logits := h.MatMul(wOut).AddRow(bOut)
	trainer.Step(logits.SoftmaxCELoss(labels))
}
```

`SelfAttention` operates on one `(seqLen x inSize)` sequence node: `attn.Forward(x)` computes `softmax(Q*K^T/sqrt(d))*V` with learned projections; the raw `tensai.Attention(q, k, v)` form is also exposed.

Autograd parameters are saved and restored positionally with `tensai.SaveParamsFile("cell.json", cell.Params()...)` / `tensai.LoadParamsFile("cell.json", cell.Params()...)` — build the same cell, then load.

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
gpu, err := tensai.OpenGPU() // fails cleanly when no GPU / library is present
if err != nil { /* fall back to tensai.MatMul */ }
defer gpu.Close()
fmt.Println(gpu.Name()) // e.g. "AMD Radeon 780M (integrated)"
out, err := gpu.MatMul(a, b)
```

On machines with both an integrated and a discrete GPU, pass a preference: `tensai.OpenGPU(tensai.GPULowPower)` steers to the iGPU, `tensai.GPUHighPerformance` to the dGPU (it is a hint — with a single adapter you always get that one).

Buffers can also stay resident on the GPU, so a weight rides the bus once instead of on every call and intermediates never leave the device:

```go
gw, _ := gpu.Upload(w)              // weight uploaded once
defer gw.Free()                     // GPU memory is not garbage collected
gx, _ := gpu.Upload(x)
h, _ := gx.MatMul(gw)               // chain freely; nothing touches the host
out, _ := h.MatMul(gw2)
result, _ := out.Download()         // one readback at the end
```

`gpu.MatMul(a, b)` is shorthand for Upload → MatMul → Download → Free. Residency matters most on discrete GPUs, where every transfer crosses PCIe; on shared-memory iGPUs the win is smaller and comes mainly from skipping intermediate readbacks.

Beyond MatMul, resident tensors support `MatMulT` (multiply by a transposed operand without materializing the transpose), an in-place `Scale`, and a row-parallel `Softmax` over the last axis — enough to run single-head attention entirely on the GPU:

```go
out, _ := gq.Attention(gk, gv)                 // softmax(q@k^T/sqrt(d))@v, no host round-trips
out, _ = gq.MultiHeadAttention(gk, gv, heads)  // packed (batch, seq, heads*dh) layout
```

Multi-head attention carves each head out of the packed layout with strided kernels — the matmul kernels take explicit row strides and per-batch offsets — so no permute is ever materialized.

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

`_example/wgpu -sweep` walks a ladder of sizes and marks where the GPU overtakes the CPU kernel. Because the CPU side is the same `dotRows` kernel the rest of the package uses, building the example twice compares all three implementations — portable Go, AVX2, and the GPU:

```bash
GOEXPERIMENT=nosimd go build -tags wgpu -o wgpu-nosimd ./_example/wgpu
GOEXPERIMENT=simd   go build -tags wgpu -o wgpu-simd   ./_example/wgpu
./wgpu-nosimd -sweep && ./wgpu-simd -sweep
```

On a Ryzen iGPU (AMD Radeon 780M, Vulkan) the crossover moves with the CPU kernel — the faster the CPU, the bigger the product has to be before the upload and readback pay for themselves:

```
             shape                   MFLOP        gpu        cpu   gpu/cpu
mnist dense  1x100x784@784x128        20.1     2.52ms      491µs     0.19x
tiny         1x128x128@128x128         4.2    3.356ms      353µs     0.11x
small        1x512x512@512x512       268.4    4.523ms    4.499ms     0.99x
medium       8x512x512@512x512      2147.5   15.894ms   49.388ms     3.11x   <- crossover (AVX2)
huge         64x512x512@512x512    17179.9   121.01ms  392.146ms     3.24x
```

With the portable kernel instead of AVX2 the crossover arrives one rung earlier, at `small`. Either way a single MNIST layer is two orders of magnitude too small to bother.

wgpu-native picks Vulkan on Linux, Vulkan or D3D12 on Windows, and Metal on macOS, so AMD, Intel, Apple, and NVIDIA GPUs all work — as do CPU Vulkan implementations like lavapipe, which is how the tests run on machines without a GPU. Every call uploads the operands and reads the product back over the bus, so the GPU only pays off for large products; without the build tag `OpenGPU` returns an error and nothing else changes.

#### `-tags wgpu24`: the new wgpu-native API, and the real GPU inside WSL2

`-tags wgpu24` (linux/darwin/windows) builds the same `OpenGPU` API against the reworked wgpu-native C API instead — pair it with a **v29-series** release binary. The new API's payoff is `WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter`, which un-hides non-conformant Vulkan drivers. Concretely: Mesa's dozen (Vulkan-on-D3D12, shipped in the kisak-mesa PPA) exposes the real host GPU inside WSL2, but the v22 API hides it as non-conformant and falls back to lavapipe; the wgpu24 build reaches it:

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
go run ./_example/plasma
go run ./_example/tensor
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

- [x] Matmul (`Dot`/`DotInto`) — used by `Dense`, `Conv2D` (im2col product), `KNN` distances, and autograd `MatMul`
- [x] ReLU / LeakyReLU forward & backward
- [x] Sigmoid / Tanh forward & backward (vectorized polynomial `exp`)
- [x] GELU forward & backward (vectorized `erf`)
- [x] LayerNorm forward & backward (vector row reductions)
- [x] Softmax / SoftmaxCrossEntropy exponentials and scaling
- [x] Adam / AdamW parameter update
- [x] Slice add & scale primitives (bias add, `Embedding` gradient scatter-add)
- [x] Transpose-free gradient matmul (`DotTAInto`) — `Dense`/`Conv2D` weight gradients no longer materialize `input^T` / `im2col^T`
- [ ] Remaining transposes (`T`/`TInto`, now only weight matrices and autograd) — needs 8x8 block-and-shuffle
- [ ] Softmax backward row dot products (layer and autograd)
- [ ] MSE / BinaryCrossEntropy losses (BCE needs a vectorized `log`)
- [ ] Autograd element-wise backward passes (gradients accumulate with `+=`, so they need dedicated fused kernels)
- [ ] BatchNorm statistics (column-strided access needs a restructure)
- [ ] MaxPool2D window scan
- [ ] im2col / col2im gather-scatter (contiguous runs could use bulk copies)
- [ ] SGD update

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

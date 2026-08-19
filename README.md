<p align="center"><img src="logo.svg" width="420" alt="tensai"></p>

# tensai - a tiny machine-learning framework in Go

`tensai` is a small machine-learning framework for learning and experiments. It implements forward passes, backpropagation, and optimization in pure Go with no external dependencies.

## Features

- **Matrix operations** - `Matrix` plus basic operations such as `Dot`, `Add`, `T`, and `AddBias`. Tensors are float32 (`tensai.Float`)
- **SIMD acceleration** - AVX2 kernels written with Go's experimental `simd/archsimd` package: still pure Go, no cgo, no assembly files. Matmul, ReLU/LeakyReLU, Sigmoid/Tanh/Softmax (via a vectorized polynomial `exp`), and the Adam update are all 8-lane vectorized. Build with `GOEXPERIMENT=simd` on amd64 (Go 1.26+); every other build uses the portable fallbacks automatically
- **Low-allocation training** - layers reuse their forward/backward scratch buffers across training steps (a full MLP step runs in ~29 allocations), so GC stays out of the training loop; `Predict` always returns freshly allocated results
- **Layers** - `Dense`, `Conv2D`, `MaxPool2D`, `BatchNorm`, `Dropout`, plus `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, and `Softmax` activations
- **Loss functions** - `MeanSquaredError` for regression, `SoftmaxCrossEntropy` for multi-class classification, and `BinaryCrossEntropy` for binary targets
- **Optimizers** - momentum `SGD`, `Adam`, and `AdamW` (decoupled weight decay)
- **k-NN baseline** - a `KNN` classifier whose distance matrix runs on the same SIMD matmul kernel; useful as a no-training baseline next to the networks
- **Sequential models** - stack layers and run `Compile` -> `Fit` / `FitStep` -> `Predict`
- **Automatic differentiation** - a micrograd-style reverse-mode autograd engine over matrices (`Param` / `Input` / `Backward`), for models that don't fit the Sequential mold; `ToDot` renders the computation graph for Graphviz
- **Recurrence and attention** - `RNNCell`, `LSTMCell`, and single-head `SelfAttention` built on the autograd engine, with backpropagation through time handled automatically
- **Serialization** - `Save`/`Load` (and `SaveFile`/`LoadFile`) round-trip trained Sequential parameters as JSON, including BatchNorm running statistics; `SaveParams`/`LoadParams` do the same for autograd parameters (RNN/LSTM/attention cells)

## Layout

```
go.mod              Module definition (github.com/mattn/tensai)
tensor.go           Matrix and vector operations
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
_example/xor        Runnable XOR training example
_example/fizzbuzz   Runnable FizzBuzz classification example
_example/spiral     Runnable 3-class spiral classification example
_example/iris       Runnable Iris classification example
_example/mnist      Runnable MNIST classifier (-model dense, cnn, or knn) with save/load
_example/charrnn    Character-level LSTM text generation on the autograd engine
_example/plasma     Demoscene-style terminal plasma rendered by a neural network
_example/dot        Graphviz DOT export of the z = x + y graph
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

## Run

```bash
go run ./_example/helloworld
go run ./_example/xor
go run ./_example/fizzbuzz
go run ./_example/spiral
go run ./_example/iris
go run ./_example/charrnn
go run ./_example/plasma
go test ./...

# With the AVX2 SIMD kernel (Go 1.26+, amd64):
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -bench=Dot .
```

The MNIST example downloads the standard IDX gzip files into `_example/mnist/data` when they are missing. Set `MNIST_DIR` to use another cache directory, and pass `-model cnn` for the convolutional variant (Conv2D/MaxPool2D/Dropout + AdamW); both trained variants finish by saving the model and re-scoring it after a reload. `-model knn` runs the no-training k-NN baseline instead — on the 5000-sample subset it scores ~91% against ~92% for the MLP and ~95% for the CNN:

```bash
go run ./_example/mnist
go run ./_example/mnist -model cnn
go run ./_example/mnist -model knn
MNIST_DIR=/path/to/mnist go run ./_example/mnist
```

The charrnn example trains a character-level LSTM on an embedded public-domain text, saves the parameters with `SaveParamsFile`, restores them into a fresh model, and generates a sample from the reloaded parameters.

The plasma example animates a demoscene-style plasma in the terminal where the plasma function is a randomly weighted network (a CPPN) evaluated for every pixel of every frame as one batch. The status line shows the per-frame network time, which makes it a live SIMD benchmark: 120x90 pixels runs at ~32 fps on the portable build and ~100 fps with `GOEXPERIMENT=simd` on the same machine. Try different `-seed` values for different effects.

Both raw IDX files and `.gz` variants are accepted.

## Design Notes

- All operations are batched. Inputs are `MxN` matrices, where `M` is the batch size and `N` is the feature dimension.
- The `Layer` interface standardizes `Forward`, `Backward`, `Params`, and `Grads`, which keeps new layers such as convolution or dropout straightforward to add.
- `Dense` weights use Glorot/He-style initialization to keep early training stable.
- `SoftmaxCrossEntropy` subtracts the row maximum before softmax for numerical stability.

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)

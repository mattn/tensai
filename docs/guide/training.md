# Training Models

## The Sequential workflow

Stack layers, then `Compile` → `Fit` (or `FitStep`) → `Predict`:

```go
net := model.NewSequential()
net.Add(layer.NewDense(8))
net.Add(&layer.Tanh{})
net.Add(layer.NewDense(1))
net.Add(&layer.Sigmoid{})

net.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05))
net.Fit(inputs, targets, 5000)   // 5000 epochs over the full batch

pred, _ := net.Predict(inputs)
```

`Compile(inputCols, loss, optimizer)` initializes every layer, threading the column count through the stack. `Fit` runs full-batch epochs; `FitStep(input, target)` runs exactly one forward/backward/update step and returns the loss, which is the building block for mini-batch training.

## Datasets

`Dataset` pairs inputs with targets and provides shuffling, splitting, standardization, and buffer-reusing mini-batch iteration:

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

`Split` returns copy-free views; `Batches` reuses its batch buffers across iterations, keeping allocation out of the inner loop.

## A convolutional model

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
```

Each input row is a flattened channel-major image (`index = (channel*height + y)*width + x`). `Dropout` and `BatchNorm` are automatically in training mode inside `Fit`/`FitStep` and in inference mode inside `Predict`.

## Saving and loading

`Save`/`Load` (and the `SaveFile`/`LoadFile` convenience wrappers) round-trip trained Sequential parameters as JSON, including BatchNorm running statistics:

```go
net.SaveFile("model.json")

// Later: build + Compile the same architecture, then
net.LoadFile("model.json")
```

The architecture itself is not serialized — reconstruct the same layer stack and `Compile` before loading. Autograd parameters (RNN/LSTM/attention cells) are saved positionally with `SaveParams`/`LoadParams` — see [Automatic Differentiation](autograd.md).

For deployment beyond Go, trained models export to TFLite and ONNX — see [Model Formats](../formats.md).

## Low-allocation training

Layers reuse their forward/backward scratch buffers across training steps, so a full MLP step runs in ~29 allocations and GC stays out of the training loop. `Predict` always returns freshly allocated results, so predictions are safe to keep.

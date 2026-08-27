# Training Models

## The Sequential workflow

Stack layers, then `Compile` → `Fit` (or `FitStep`) → `Predict`:

```go
model := tensai.NewSequential()
model.Add(tensai.NewDense(8))
model.Add(&tensai.Tanh{})
model.Add(tensai.NewDense(1))
model.Add(&tensai.Sigmoid{})

model.Compile(2, tensai.MeanSquaredError{}, tensai.NewAdam(0.05))
model.Fit(inputs, targets, 5000)   // 5000 epochs over the full batch

pred, _ := model.Predict(inputs)
```

`Compile(inputCols, loss, optimizer)` initializes every layer, threading the column count through the stack. `Fit` runs full-batch epochs; `FitStep(input, target)` runs exactly one forward/backward/update step and returns the loss, which is the building block for mini-batch training.

## Datasets

`Dataset` pairs inputs with targets and provides shuffling, splitting, standardization, and buffer-reusing mini-batch iteration:

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

`Split` returns copy-free views; `Batches` reuses its batch buffers across iterations, keeping allocation out of the inner loop.

## A convolutional model

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
```

Each input row is a flattened channel-major image (`index = (channel*height + y)*width + x`). `Dropout` and `BatchNorm` are automatically in training mode inside `Fit`/`FitStep` and in inference mode inside `Predict`.

## Saving and loading

`Save`/`Load` (and the `SaveFile`/`LoadFile` convenience wrappers) round-trip trained Sequential parameters as JSON, including BatchNorm running statistics:

```go
model.SaveFile("model.json")

// Later: build + Compile the same architecture, then
model.LoadFile("model.json")
```

The architecture itself is not serialized — reconstruct the same layer stack and `Compile` before loading. Autograd parameters (RNN/LSTM/attention cells) are saved positionally with `SaveParams`/`LoadParams` — see [Automatic Differentiation](autograd.md).

For deployment beyond Go, trained models export to TFLite and ONNX — see [Model Formats](../formats.md).

## Low-allocation training

Layers reuse their forward/backward scratch buffers across training steps, so a full MLP step runs in ~29 allocations and GC stays out of the training loop. `Predict` always returns freshly allocated results, so predictions are safe to keep.

# Layers, Losses, Optimizers

## The Layer interface

Every layer implements the same interface:

```go
type Layer interface {
	Init(inputCols int, rng *rand.Rand) (outputCols int, err error)
	Forward(input *Matrix) (*Matrix, error)
	Backward(gradOutput *Matrix) (*Matrix, error)
	Grads() (*Matrix, []Float)
	Params() (*Matrix, []Float)
	SetParams(weights *Matrix, bias []Float) error
}
```

`Init` is called by `Sequential.Compile`, which threads the column count through the stack — that is why `NewDense(8)` only names its output width. Layers reuse their forward/backward scratch buffers across training steps, so GC stays out of the training loop (a full MLP step runs in ~29 allocations); `Predict` always returns freshly allocated results.

## Layers

| Layer | Constructor | Notes |
|---|---|---|
| Dense | `layer.NewDense(outCols)` | Fully connected; Glorot/He-style initialization |
| Embedding | `layer.NewEmbedding(vocabSize, dim)` | Input rows are integer token ids stored in `Float`; looked-up vectors concatenate across the row |
| Conv2D | `layer.NewConv2D(inH, inW, inC, outC, kernel, stride, pad)` | im2col + the `Dot` kernel |
| MaxPool2D | `layer.NewMaxPool2D(inH, inW, channels, size)` | |
| BatchNorm | `layer.NewBatchNorm()` | Running statistics are saved with the model |
| LayerNorm | `layer.NewLayerNorm()` | Per-row normalization |
| Dropout | `layer.NewDropout(rate)` | Active only during training |

`Conv2D` and `MaxPool2D` treat each row as a channel-major image: `index = (channel*height + y)*width + x`. `Dropout` and `BatchNorm` switch automatically between training behavior (inside `Fit`/`FitStep`) and inference behavior (inside `Predict`).

`Embedding` keeps the matrix-only API: each input row is a token-id sequence, and the layer concatenates the looked-up embedding vectors across columns. For example, `Compile(4, ...)` plus `NewEmbedding(vocab, 8)` turns an `Mx4` token-id matrix into an `Mx32` dense feature matrix that can feed `LayerNorm`, `GELU`, and `Dense`.

## Activations

Activations are layers too — add them like any other:

| Activation | Usage |
|---|---|
| ReLU | `&layer.ReLU{}` |
| LeakyReLU | `layer.NewLeakyReLU(0.01)` |
| GELU | `&layer.GELU{}` (vectorized `erf` in the SIMD build) |
| Sigmoid | `&layer.Sigmoid{}` |
| Tanh | `&layer.Tanh{}` |
| Softmax | `&layer.Softmax{}` (usually you want `SoftmaxCrossEntropy` instead) |

## Loss functions

| Loss | For | Targets |
|---|---|---|
| `loss.MeanSquaredError{}` | Regression | Same shape as the prediction |
| `loss.SoftmaxCrossEntropy{}` | Multi-class classification | `Mx1` matrix of class indices |
| `loss.BinaryCrossEntropy{}` | Binary targets | Same shape as the prediction |

Softmax is applied *inside* `SoftmaxCrossEntropy` (with the row maximum subtracted for numerical stability), so the model ends with a plain `Dense` and `Predict` returns raw logits — use argmax for the class.

## Optimizers

| Optimizer | Constructor |
|---|---|
| Momentum SGD | `optim.NewSGD(lr, momentum)` |
| Adam | `optim.NewAdam(lr)` |
| AdamW | `optim.NewAdamW(lr, weightDecay)` — decoupled weight decay |

The Adam/AdamW parameter update is one of the AVX2-vectorized kernels in the SIMD build.

## k-NN baseline

`knn.New(k)` is a no-training baseline classifier — `Fit` just stores the data, and `Predict` builds the distance matrix on the same SIMD matmul kernel. On the MNIST 5000-sample subset it scores ~91% against ~92% for the MLP and ~95% for the CNN — a useful sanity floor next to the networks.

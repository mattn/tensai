# Automatic Differentiation

When a model doesn't fit the Sequential mold (weight sharing, custom losses, exotic architectures), build the computation directly and let reverse-mode autodiff derive the gradients. The engine is micrograd-style: a dynamically built graph of `Node`s over matrices, define-by-run, single-use.

## Params, Inputs, and the Trainer

```go
w1 := autograd.Param(tensai.RandomMatrix(2, 8, rng))
b1 := autograd.Param(tensai.NewMatrix(1, 8))
w2 := autograd.Param(tensai.RandomMatrix(8, 1, rng))
trainer := autograd.NewTrainer(optim.NewAdam(0.05), w1, b1, w2)

for step := 0; step < 2000; step++ {
	loss := autograd.Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).Sigmoid().MSELoss(y)
	trainer.Step(loss) // backward + update + zero grads, returns the loss value
}
```

`Param` wraps a matrix whose gradient should be tracked and updated; `Input` wraps data. For manual control, the pieces are still public: `loss.Backward()`, `p.Grad`, and `autograd.ZeroGrads(params...)`.

## Available ops

`MatMul`, `Add`, `Sub`, `MulElem`, `Scale`, `AddRow`, `T`, `Softmax`, `ReLU`, `Sigmoid`, `Tanh`, `Sum`, `Mean`, `MSELoss`, and `SoftmaxCELoss`.

Graphs are built dynamically per step and are single-use. Shape mismatches panic during graph construction. Every op's gradient is verified against finite differences in the test suite.

## Visualizing the graph

`loss.ToDot()` returns Graphviz DOT (label leaves with `.Named("w1")`):

```bash
go run ./_example/dot | dot -Tsvg > graph.svg
```

## Recurrent networks

`rnn.Cell` and `rnn.LSTMCell` are built on the autograd engine, so unrolling a sequence is a plain Go loop and backpropagation through time comes for free:

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
	trainer.Step(logits.SoftmaxCELoss(labels))
}
```

`_example/charrnn` trains a character-level LSTM on an embedded public-domain text and generates samples from the reloaded parameters.

## Attention

`rnn.SelfAttention` operates on one `(seqLen x inSize)` sequence node: `attn.Forward(x)` computes `softmax(Q*K^T/sqrt(d))*V` with learned projections. The raw `rnn.Attention(q, k, v)` form is also exposed.

## Saving parameters

Autograd parameters are saved and restored positionally:

```go
autograd.SaveParamsFile("cell.json", cell.Params()...)
// build the same cell, then
autograd.LoadParamsFile("cell.json", cell.Params()...)
```

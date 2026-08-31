# Automatic Differentiation

When a model doesn't fit the Sequential mold (weight sharing, custom losses, exotic architectures), build the computation directly and let reverse-mode autodiff derive the gradients. The engine is micrograd-style: a dynamically built graph of `Node`s over n-dimensional tensors, define-by-run, single-use. A `Node` holds a `Tensor`, so element-wise ops broadcast and `MatMul` multiplies stacks of matrices; a `Matrix` is taken as a zero-copy 2-D view wherever a leaf is built, so the two-dimensional code below reads the same as it always did.

## Params, Inputs, and the Trainer

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

`Param` wraps a matrix whose gradient should be tracked and updated; `Input` wraps data. For manual control, the pieces are still public: `loss.Backward()`, `p.Grad`, and `autograd.ZeroGrads(params...)`.

## Available ops

`MatMul` (batched, with broadcast leading axes), `Add`, `Sub`, `Mul`/`MulElem`, `Div`, `Scale`, `Neg`, `AddRow`, `T`/`Transpose`, `Reshape`, `Softmax` (last axis), `Sum`, `Mean`, `SumAxis`/`MeanAxis`, `LayerNorm`, `Embed`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `Exp`, `Log`, `MSELoss`, `SoftmaxCELoss`, and `CrossEntropy`. Element-wise ops broadcast NumPy-style, and a gradient is summed back over whatever axes an operand was stretched along.

Graphs are built dynamically per step and are single-use. Shape mismatches panic during graph construction. Every op's gradient is verified against finite differences in the test suite.

## Reusing buffers with a tape

A graph is built and thrown away every step, so by default every step asks the allocator for every intermediate value and gradient it touches. A `Tape` recycles them. Bind the parameters once, then reset it at the end of each step:

```go
tape := autograd.NewTape()
tape.Bind(w1, b1, w2, b2) // ops inherit the tape from their parents

for step := 0; step < steps; step++ {
	trainer.Step(forward(x).MSELoss(y))
	tape.Reset() // hands this step's buffers back to the pool
}
```

The rule is the one a training loop already follows: **after `Reset`, nothing from the finished step may be read** — no node's `Value`, no node's `Grad`. Parameter values are never recycled (the tape only owns what operations produce), so trained weights are always safe to keep; copy anything else before resetting. A `Tape` is not safe for concurrent use, so give each training goroutine its own.

On `_example/charrnn`, which unrolls 32 time steps per iteration, the tape takes a training step from 22 MB of allocation to 0.75 MB and cuts about a quarter off its wall time.

The same reuse is available one layer down: `MatMulInto`, `MatMulTNInto`, `MatMulNTInto`, `AddInto`, `SubInto`, `MulInto` and `DivInto` write into a tensor you already own, the way `DotInto` does for matrices.

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
	trainer.Step(logits.CrossEntropy(labels)) // labels is a []int of class indices
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

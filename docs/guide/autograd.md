# Automatic Differentiation

When a model doesn't fit the Sequential mold (weight sharing, custom losses, exotic architectures), build the computation directly and let reverse-mode autodiff derive the gradients. The engine is micrograd-style: a dynamically built graph of `Node`s, define-by-run, single-use.

A `Node` holds an n-dimensional `Tensor`, so the same ops that run on a `(rows, cols)` matrix run on a `(batch, sequence, model)` activation: element-wise arithmetic broadcasts NumPy-style and `MatMul` multiplies whole stacks of matrices. A `Matrix` is taken as a zero-copy 2-D view wherever a leaf is built, so two-dimensional code reads the way it always did.

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

`Param` wraps a value whose gradient should be tracked and updated; `Input` wraps data. Both accept a `*tensai.Matrix` or a `*tensai.Tensor` — a matrix becomes a 2-D tensor view sharing the same backing array, so a parameter built from a matrix keeps updating that matrix.

For manual control, the pieces are public: `loss.Backward()`, `p.Grad`, and `autograd.ZeroGrads(params...)`. `Backward` starts from a single-element node and **accumulates** into `Grad`, which is why a training step clears the gradients afterwards (`Trainer.Step` does it for you).

On a node, `Value` and `Grad` are `*tensai.Tensor`. `Shape()` returns the shape, `Matrix()` returns a matrix view of a 2-D node, `Scalar()` reads a single-element node, and `Named("w1")` labels a leaf for `ToDot`.

## Ops

| Op | Shapes |
| --- | --- |
| `MatMul(o)` | `(…, m, k) * (…, k, n)` → `(…, m, n)`; the leading axes broadcast, so one 2-D weight applies to a whole batch |
| `Add`, `Sub`, `Mul` (`MulElem`), `Div` | element-wise, NumPy broadcasting |
| `Scale(s)`, `Neg()` | element-wise by a scalar |
| `AddRow(row)` | `(m, n) + (1, n)`; `Add` already broadcasts, this only states the intent |
| `T()`, `Transpose(perm...)` | swap the last two axes / permute every axis |
| `Reshape(shape...)` | one dimension may be `-1`; shares the buffer |
| `Softmax()` | over the last axis |
| `LayerNorm(gain, bias, eps)` | over the last axis; `gain` and `bias` hold one element per feature and may be `nil` or trainable |
| `Embed(ids, shape...)` | `(vocab, d)` table plus `len(ids)` indices → `shape…, d`; repeated ids accumulate on backward |
| `Sum()`, `Mean()` | reduce everything to one element |
| `SumAxis(axis, keepDims)`, `MeanAxis(axis, keepDims)` | reduce one axis; a negative axis counts from the end |
| `ReLU()`, `LeakyReLU(a)`, `Sigmoid()`, `Tanh()`, `GELU()`, `Exp()`, `Log()` | element-wise |
| `MSELoss(target)`, `SoftmaxCELoss(target)`, `CrossEntropy(labels []int)` | scalar losses; the cross-entropies read the last axis as classes |

Graphs are built dynamically per step and are single-use. Shape mismatches panic during construction rather than returning an error: chaining would be unusable otherwise, and a wrong shape is a programming mistake. Every op's gradient is verified against finite differences in the test suite.

## Broadcasting

An element-wise op aligns shapes at their trailing axes and stretches any axis of length 1, so a `(1, 1, d)` bias adds to a `(batch, seq, d)` activation. Gradients follow the same rule in reverse: whatever axes an operand was stretched along, its gradient is summed back over. That is why a bias, a per-feature `LayerNorm` gain, and a weight shared across a batch all collect the contributions of every position that used them without any special case in the op.

## Building attention

Multi-head attention is a reshape, a transpose, and two batched products:

```go
// x is (batch, seq, model); wq, wk, wv, wo are (model, model).
heads := func(t *autograd.Node) *autograd.Node {
	// (batch, seq, model) -> (batch, head, seq, headDim)
	return t.Reshape(batch, seq, nHeads, headDim).Transpose(0, 2, 1, 3)
}
q, k, v := heads(x.MatMul(wq)), heads(x.MatMul(wk)), heads(x.MatMul(wv))

// (batch, head, seq, seq) scores for every head of every sequence at once.
att := q.MatMul(k.T()).Scale(1 / float32(math.Sqrt(headDim))).Add(mask).Softmax()

y := att.MatMul(v).Transpose(0, 2, 1, 3).Reshape(batch, seq, model).MatMul(wo)
```

`mask` is a constant `Input` of shape `(1, 1, seq, seq)` holding `0` on and below the diagonal and `math.Inf(-1)` above it, broadcast over batch and head; softmax then gives future positions no weight. `_example/tinygpt` puts exactly this into a working character-level transformer — pre-norm blocks, a GELU feed-forward, and next-character cross-entropy — that memorizes a page of text in about a minute.

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

The rule is the one a training loop already follows: **after `Reset`, nothing from the finished step may be read** — no node's `Value`, no node's `Grad`. Parameter values are never recycled (the tape only owns what operations produce), so trained weights are always safe to keep; copy anything else with `Clone` before resetting. A `Tape` is not safe for concurrent use, so give each training goroutine its own.

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

`rnn.SelfAttention` is the single-head, one-sequence form: `attn.Forward(x)` computes `softmax(Q*K^T/sqrt(d))*V` on a `(seqLen, inSize)` node with learned projections, and the raw `rnn.Attention(q, k, v)` is also exposed. For batches and heads, write the block above.

## Saving parameters

Autograd parameters are saved and restored positionally:

```go
autograd.SaveParamsFile("cell.json", cell.Params()...)
// build the same cell, then
autograd.LoadParamsFile("cell.json", cell.Params()...)
```

Parameters of any rank round-trip; two-dimensional ones keep the encoding earlier checkpoints used, so files written before the engine went n-dimensional still load.

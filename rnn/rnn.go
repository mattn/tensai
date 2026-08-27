package rnn

import (
	"math/rand"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/internal/kernels"
)

// The recurrent cells and attention below are built on the autograd autograd.Node
// graph rather than the Layer interface: unrolling through time (BPTT) and
// attention's data-dependent weights fall out of reverse-mode autodiff for
// free, where a hand-written Layer Backward would have to re-derive them.
//
// Convention: one time step is a (batch x features) matrix node; a sequence
// is a Go slice of steps. Hidden state starts as a zero Input node.

// Cell is a simple (Elman) recurrent cell:
// h' = tanh(x*Wx + h*Wh + b).
type Cell struct {
	Wx *autograd.Node // in x hidden
	Wh *autograd.Node // hidden x hidden
	B  *autograd.Node // 1 x hidden
}

// NewCell returns a randomly initialized RNN cell.
func NewCell(inSize, hidden int, rng *rand.Rand) *Cell {
	return &Cell{
		Wx: autograd.Param(tensai.RandomMatrix(inSize, hidden, rng)),
		Wh: autograd.Param(tensai.RandomMatrix(hidden, hidden, rng)),
		B:  autograd.Param(tensai.NewMatrix(1, hidden)),
	}
}

// Step consumes one time step and the previous hidden state, returning the
// next hidden state.
func (c *Cell) Step(x, h *autograd.Node) *autograd.Node {
	return x.MatMul(c.Wx).Add(h.MatMul(c.Wh)).AddRow(c.B).Tanh()
}

// InitState returns a zero hidden state for the given batch size.
func (c *Cell) InitState(batch int) *autograd.Node {
	return autograd.Input(tensai.NewMatrix(batch, c.Wh.Value.Rows))
}

// Params returns the cell's trainable parameters, for NewTrainer.
func (c *Cell) Params() []*autograd.Node {
	return []*autograd.Node{c.Wx, c.Wh, c.B}
}

// LSTMCell is a long short-term memory cell with forget/input/output gates.
type LSTMCell struct {
	// One (Wx, Wh, B) triple per gate: forget, input, output, candidate.
	Wxf, Whf, Bf *autograd.Node
	Wxi, Whi, Bi *autograd.Node
	Wxo, Who, Bo *autograd.Node
	Wxg, Whg, Bg *autograd.Node
}

// NewLSTMCell returns a randomly initialized LSTM cell. Forget-gate biases
// start at 1 so early training defaults to remembering.
func NewLSTMCell(inSize, hidden int, rng *rand.Rand) *LSTMCell {
	newGate := func() (*autograd.Node, *autograd.Node, *autograd.Node) {
		return autograd.Param(tensai.RandomMatrix(inSize, hidden, rng)),
			autograd.Param(tensai.RandomMatrix(hidden, hidden, rng)),
			autograd.Param(tensai.NewMatrix(1, hidden))
	}
	c := &LSTMCell{}
	c.Wxf, c.Whf, c.Bf = newGate()
	c.Wxi, c.Whi, c.Bi = newGate()
	c.Wxo, c.Who, c.Bo = newGate()
	c.Wxg, c.Whg, c.Bg = newGate()
	for i := range c.Bf.Value.Data {
		c.Bf.Value.Data[i] = 1
	}
	return c
}

// Step consumes one time step with the previous (hidden, cell) state and
// returns the next (hidden, cell) state.
func (c *LSTMCell) Step(x, h, cell *autograd.Node) (*autograd.Node, *autograd.Node) {
	gate := func(wx, wh, b *autograd.Node) *autograd.Node {
		return x.MatMul(wx).Add(h.MatMul(wh)).AddRow(b)
	}
	f := gate(c.Wxf, c.Whf, c.Bf).Sigmoid()
	i := gate(c.Wxi, c.Whi, c.Bi).Sigmoid()
	o := gate(c.Wxo, c.Who, c.Bo).Sigmoid()
	g := gate(c.Wxg, c.Whg, c.Bg).Tanh()
	cellNext := f.MulElem(cell).Add(i.MulElem(g))
	hNext := o.MulElem(cellNext.Tanh())
	return hNext, cellNext
}

// InitState returns zero (hidden, cell) states for the given batch size.
func (c *LSTMCell) InitState(batch int) (*autograd.Node, *autograd.Node) {
	hidden := c.Whf.Value.Rows
	return autograd.Input(tensai.NewMatrix(batch, hidden)), autograd.Input(tensai.NewMatrix(batch, hidden))
}

// Params returns the cell's trainable parameters, for NewTrainer.
func (c *LSTMCell) Params() []*autograd.Node {
	return []*autograd.Node{
		c.Wxf, c.Whf, c.Bf,
		c.Wxi, c.Whi, c.Bi,
		c.Wxo, c.Who, c.Bo,
		c.Wxg, c.Whg, c.Bg,
	}
}

// Attention computes scaled dot-product attention softmax(q*k^T/sqrt(d))*v
// for a single sequence, where q, k, v are (seqLen x d) nodes.
func Attention(q, k, v *autograd.Node) *autograd.Node {
	scale := 1 / kernels.SqrtF(tensai.Float(k.Value.Cols))
	return q.MatMul(k.T()).Scale(scale).Softmax().MatMul(v)
}

// SelfAttention is a single-head self-attention block with learned query,
// key, and value projections. It operates on one sequence at a time: the
// input is a (seqLen x inSize) node.
type SelfAttention struct {
	Wq, Wk, Wv *autograd.Node // inSize x dModel
}

// NewSelfAttention returns a randomly initialized self-attention block.
func NewSelfAttention(inSize, dModel int, rng *rand.Rand) *SelfAttention {
	return &SelfAttention{
		Wq: autograd.Param(tensai.RandomMatrix(inSize, dModel, rng)),
		Wk: autograd.Param(tensai.RandomMatrix(inSize, dModel, rng)),
		Wv: autograd.Param(tensai.RandomMatrix(inSize, dModel, rng)),
	}
}

// Forward applies self-attention to a (seqLen x inSize) sequence, returning
// a (seqLen x dModel) sequence.
func (a *SelfAttention) Forward(x *autograd.Node) *autograd.Node {
	return Attention(x.MatMul(a.Wq), x.MatMul(a.Wk), x.MatMul(a.Wv))
}

// Params returns the block's trainable parameters, for NewTrainer.
func (a *SelfAttention) Params() []*autograd.Node {
	return []*autograd.Node{a.Wq, a.Wk, a.Wv}
}

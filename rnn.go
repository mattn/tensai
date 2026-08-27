package tensai

import (
	"math/rand"

	"github.com/mattn/tensai/internal/kernels"
)

// The recurrent cells and attention below are built on the autograd Node
// graph rather than the Layer interface: unrolling through time (BPTT) and
// attention's data-dependent weights fall out of reverse-mode autodiff for
// free, where a hand-written Layer Backward would have to re-derive them.
//
// Convention: one time step is a (batch x features) matrix node; a sequence
// is a Go slice of steps. Hidden state starts as a zero Input node.

// RNNCell is a simple (Elman) recurrent cell:
// h' = tanh(x*Wx + h*Wh + b).
type RNNCell struct {
	Wx *Node // in x hidden
	Wh *Node // hidden x hidden
	B  *Node // 1 x hidden
}

// NewRNNCell returns a randomly initialized RNN cell.
func NewRNNCell(inSize, hidden int, rng *rand.Rand) *RNNCell {
	return &RNNCell{
		Wx: Param(RandomMatrix(inSize, hidden, rng)),
		Wh: Param(RandomMatrix(hidden, hidden, rng)),
		B:  Param(NewMatrix(1, hidden)),
	}
}

// Step consumes one time step and the previous hidden state, returning the
// next hidden state.
func (c *RNNCell) Step(x, h *Node) *Node {
	return x.MatMul(c.Wx).Add(h.MatMul(c.Wh)).AddRow(c.B).Tanh()
}

// InitState returns a zero hidden state for the given batch size.
func (c *RNNCell) InitState(batch int) *Node {
	return Input(NewMatrix(batch, c.Wh.Value.Rows))
}

// Params returns the cell's trainable parameters, for NewTrainer.
func (c *RNNCell) Params() []*Node {
	return []*Node{c.Wx, c.Wh, c.B}
}

// LSTMCell is a long short-term memory cell with forget/input/output gates.
type LSTMCell struct {
	// One (Wx, Wh, B) triple per gate: forget, input, output, candidate.
	Wxf, Whf, Bf *Node
	Wxi, Whi, Bi *Node
	Wxo, Who, Bo *Node
	Wxg, Whg, Bg *Node
}

// NewLSTMCell returns a randomly initialized LSTM cell. Forget-gate biases
// start at 1 so early training defaults to remembering.
func NewLSTMCell(inSize, hidden int, rng *rand.Rand) *LSTMCell {
	newGate := func() (*Node, *Node, *Node) {
		return Param(RandomMatrix(inSize, hidden, rng)),
			Param(RandomMatrix(hidden, hidden, rng)),
			Param(NewMatrix(1, hidden))
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
func (c *LSTMCell) Step(x, h, cell *Node) (*Node, *Node) {
	gate := func(wx, wh, b *Node) *Node {
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
func (c *LSTMCell) InitState(batch int) (*Node, *Node) {
	hidden := c.Whf.Value.Rows
	return Input(NewMatrix(batch, hidden)), Input(NewMatrix(batch, hidden))
}

// Params returns the cell's trainable parameters, for NewTrainer.
func (c *LSTMCell) Params() []*Node {
	return []*Node{
		c.Wxf, c.Whf, c.Bf,
		c.Wxi, c.Whi, c.Bi,
		c.Wxo, c.Who, c.Bo,
		c.Wxg, c.Whg, c.Bg,
	}
}

// Attention computes scaled dot-product attention softmax(q*k^T/sqrt(d))*v
// for a single sequence, where q, k, v are (seqLen x d) nodes.
func Attention(q, k, v *Node) *Node {
	scale := 1 / kernels.SqrtF(Float(k.Value.Cols))
	return q.MatMul(k.T()).Scale(scale).Softmax().MatMul(v)
}

// SelfAttention is a single-head self-attention block with learned query,
// key, and value projections. It operates on one sequence at a time: the
// input is a (seqLen x inSize) node.
type SelfAttention struct {
	Wq, Wk, Wv *Node // inSize x dModel
}

// NewSelfAttention returns a randomly initialized self-attention block.
func NewSelfAttention(inSize, dModel int, rng *rand.Rand) *SelfAttention {
	return &SelfAttention{
		Wq: Param(RandomMatrix(inSize, dModel, rng)),
		Wk: Param(RandomMatrix(inSize, dModel, rng)),
		Wv: Param(RandomMatrix(inSize, dModel, rng)),
	}
}

// Forward applies self-attention to a (seqLen x inSize) sequence, returning
// a (seqLen x dModel) sequence.
func (a *SelfAttention) Forward(x *Node) *Node {
	return Attention(x.MatMul(a.Wq), x.MatMul(a.Wk), x.MatMul(a.Wv))
}

// Params returns the block's trainable parameters, for NewTrainer.
func (a *SelfAttention) Params() []*Node {
	return []*Node{a.Wq, a.Wk, a.Wv}
}

package tensai

import (
	"math/rand"
	"testing"
)

func TestAutogradTransposeAndSoftmax(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	w := RandomMatrix(3, 4, rng)
	x := RandomMatrix(5, 3, rng)

	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		return Input(x).MatMul(p).T().MatMul(Input(x)).Sum(), p
	}, "transpose")

	// Weighted sum keeps the softmax gradient non-trivial (a plain Sum of
	// each row is constant 1 and would zero it out).
	weights := RandomMatrix(5, 4, rng)
	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		return Input(x).MatMul(p).Softmax().MulElem(Input(weights)).Sum(), p
	}, "softmax")
}

func TestRNNCellGradientThroughTime(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	cell := NewRNNCell(2, 3, rng)
	steps := []*Matrix{
		RandomMatrix(4, 2, rng),
		RandomMatrix(4, 2, rng),
		RandomMatrix(4, 2, rng),
	}
	// Wh is used at every time step: this validates gradient accumulation
	// through the unrolled graph (BPTT).
	checkParamGrad(t, cell.Wh.Value, func() (*Node, *Node) {
		h := cell.InitState(4)
		for _, x := range steps {
			h = cell.Step(Input(x), h)
		}
		return h.Sum(), cell.Wh
	}, "rnn-Wh")
	checkParamGrad(t, cell.Wx.Value, func() (*Node, *Node) {
		h := cell.InitState(4)
		for _, x := range steps {
			h = cell.Step(Input(x), h)
		}
		return h.Sum(), cell.Wx
	}, "rnn-Wx")
}

func TestLSTMCellGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	cell := NewLSTMCell(2, 3, rng)
	steps := []*Matrix{
		RandomMatrix(4, 2, rng),
		RandomMatrix(4, 2, rng),
	}
	for _, param := range []struct {
		name string
		node *Node
	}{
		{"lstm-Whf", cell.Whf},
		{"lstm-Wxi", cell.Wxi},
		{"lstm-Bg", cell.Bg},
	} {
		checkParamGrad(t, param.node.Value, func() (*Node, *Node) {
			h, c := cell.InitState(4)
			for _, x := range steps {
				h, c = cell.Step(Input(x), h, c)
			}
			return h.Sum(), param.node
		}, param.name)
	}
}

func TestSelfAttentionGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	attn := NewSelfAttention(4, 3, rng)
	seq := RandomMatrix(5, 4, rng) // seqLen=5, inSize=4
	weights := RandomMatrix(5, 3, rng)
	for _, param := range []struct {
		name string
		node *Node
	}{
		{"attn-Wq", attn.Wq},
		{"attn-Wk", attn.Wk},
		{"attn-Wv", attn.Wv},
	} {
		checkParamGrad(t, param.node.Value, func() (*Node, *Node) {
			out := attn.Forward(Input(seq))
			return out.MulElem(Input(weights)).Sum(), param.node
		}, param.name)
	}
}

func TestRNNLearnsParity(t *testing.T) {
	// All 16 binary sequences of length 4; the label is the XOR of the bits.
	// Solving it requires the hidden state to carry information across
	// steps, so convergence exercises backpropagation through time.
	const seqLen = 4
	const batch = 1 << seqLen
	stepInputs := make([]*Matrix, seqLen)
	for t := range stepInputs {
		stepInputs[t] = NewMatrix(batch, 1)
	}
	targets := NewMatrix(batch, 1)
	for s := 0; s < batch; s++ {
		parity := 0
		for t := 0; t < seqLen; t++ {
			bit := s >> t & 1
			stepInputs[t].Data[s] = Float(bit)
			parity ^= bit
		}
		targets.Data[s] = Float(parity)
	}

	rng := rand.New(rand.NewSource(1))
	cell := NewRNNCell(1, 8, rng)
	wOut := Param(RandomMatrix(8, 2, rng))
	bOut := Param(NewMatrix(1, 2))
	trainer := NewTrainer(NewAdam(0.05), append(cell.Params(), wOut, bOut)...)

	forward := func() *Node {
		h := cell.InitState(batch)
		for _, x := range stepInputs {
			h = cell.Step(Input(x), h)
		}
		return h.MatMul(wOut).AddRow(bOut)
	}

	var lossVal Float
	for step := 0; step < 800; step++ {
		lossVal = trainer.Step(forward().SoftmaxCELoss(targets))
	}
	if lossVal > 0.1 {
		t.Fatalf("RNN failed to learn parity: loss=%g", lossVal)
	}
	logits := forward()
	for s := 0; s < batch; s++ {
		best := 0
		if logits.Value.At(s, 1) > logits.Value.At(s, 0) {
			best = 1
		}
		if best != int(targets.Data[s]) {
			t.Errorf("sequence %04b: predicted %d, want %g", s, best, targets.Data[s])
		}
	}
}

package rnn

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/optim"
)

// checkParamGrad compares the analytic gradient of loss w.r.t. param against
// finite differences. build must construct the graph fresh from the current
// matrix contents and return (loss node, param node).
func checkParamGrad(t *testing.T, param *tensai.Matrix, build func() (*autograd.Node, *autograd.Node), name string) {
	t.Helper()
	loss, p := build()
	// Param nodes may be shared across successive checks; Backward
	// accumulates, so start from a clean gradient.
	autograd.ZeroGrads(p)
	loss.Backward()
	if p.Grad == nil {
		t.Fatalf("%s: no gradient computed", name)
	}
	analytic := make([]float64, len(p.Grad.Data))
	for i, g := range p.Grad.Data {
		analytic[i] = float64(g)
	}

	// float32 forward passes cap the precision, so the step and tolerance
	// are coarser than a float64 check would use.
	const h = 1e-2
	for i := range param.Data {
		orig := param.Data[i]
		param.Data[i] = orig + h
		lp, _ := build()
		param.Data[i] = orig - h
		lm, _ := build()
		param.Data[i] = orig
		num := float64(lp.Value.Data[0]-lm.Value.Data[0]) / (2 * h)
		if math.Abs(num-analytic[i]) > 2e-2*(1+math.Abs(num)) {
			t.Errorf("%s grad %d: numeric=%.8f analytic=%.8f", name, i, num, analytic[i])
		}
	}
}

func TestRNNCellGradientThroughTime(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	cell := NewCell(2, 3, rng)
	steps := []*tensai.Matrix{
		tensai.RandomMatrix(4, 2, rng),
		tensai.RandomMatrix(4, 2, rng),
		tensai.RandomMatrix(4, 2, rng),
	}
	// Wh is used at every time step: this validates gradient accumulation
	// through the unrolled graph (BPTT).
	checkParamGrad(t, cell.Wh.Value, func() (*autograd.Node, *autograd.Node) {
		h := cell.InitState(4)
		for _, x := range steps {
			h = cell.Step(autograd.Input(x), h)
		}
		return h.Sum(), cell.Wh
	}, "rnn-Wh")
	checkParamGrad(t, cell.Wx.Value, func() (*autograd.Node, *autograd.Node) {
		h := cell.InitState(4)
		for _, x := range steps {
			h = cell.Step(autograd.Input(x), h)
		}
		return h.Sum(), cell.Wx
	}, "rnn-Wx")
}

func TestLSTMCellGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	cell := NewLSTMCell(2, 3, rng)
	steps := []*tensai.Matrix{
		tensai.RandomMatrix(4, 2, rng),
		tensai.RandomMatrix(4, 2, rng),
	}
	for _, param := range []struct {
		name string
		node *autograd.Node
	}{
		{"lstm-Whf", cell.Whf},
		{"lstm-Wxi", cell.Wxi},
		{"lstm-Bg", cell.Bg},
	} {
		checkParamGrad(t, param.node.Value, func() (*autograd.Node, *autograd.Node) {
			h, c := cell.InitState(4)
			for _, x := range steps {
				h, c = cell.Step(autograd.Input(x), h, c)
			}
			return h.Sum(), param.node
		}, param.name)
	}
}

func TestSelfAttentionGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	attn := NewSelfAttention(4, 3, rng)
	seq := tensai.RandomMatrix(5, 4, rng) // seqLen=5, inSize=4
	weights := tensai.RandomMatrix(5, 3, rng)
	for _, param := range []struct {
		name string
		node *autograd.Node
	}{
		{"attn-Wq", attn.Wq},
		{"attn-Wk", attn.Wk},
		{"attn-Wv", attn.Wv},
	} {
		checkParamGrad(t, param.node.Value, func() (*autograd.Node, *autograd.Node) {
			out := attn.Forward(autograd.Input(seq))
			return out.MulElem(autograd.Input(weights)).Sum(), param.node
		}, param.name)
	}
}

func TestRNNLearnsParity(t *testing.T) {
	// All 16 binary sequences of length 4; the label is the XOR of the bits.
	// Solving it requires the hidden state to carry information across
	// steps, so convergence exercises backpropagation through time.
	const seqLen = 4
	const batch = 1 << seqLen
	stepInputs := make([]*tensai.Matrix, seqLen)
	for t := range stepInputs {
		stepInputs[t] = tensai.NewMatrix(batch, 1)
	}
	targets := tensai.NewMatrix(batch, 1)
	for s := 0; s < batch; s++ {
		parity := 0
		for t := 0; t < seqLen; t++ {
			bit := s >> t & 1
			stepInputs[t].Data[s] = tensai.Float(bit)
			parity ^= bit
		}
		targets.Data[s] = tensai.Float(parity)
	}

	rng := rand.New(rand.NewSource(1))
	cell := NewCell(1, 8, rng)
	wOut := autograd.Param(tensai.RandomMatrix(8, 2, rng))
	bOut := autograd.Param(tensai.NewMatrix(1, 2))
	trainer := autograd.NewTrainer(optim.NewAdam(0.05), append(cell.Params(), wOut, bOut)...)

	forward := func() *autograd.Node {
		h := cell.InitState(batch)
		for _, x := range stepInputs {
			h = cell.Step(autograd.Input(x), h)
		}
		return h.MatMul(wOut).AddRow(bOut)
	}

	var lossVal tensai.Float
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

func TestSaveLoadParams(t *testing.T) {
	rng := rand.New(rand.NewSource(59))
	cell := NewLSTMCell(3, 5, rng)
	x := tensai.RandomMatrix(4, 3, rng)

	forward := func(c *LSTMCell) *tensai.Matrix {
		h, s := c.InitState(4)
		h, _ = c.Step(autograd.Input(x), h, s)
		return h.Value
	}
	want := forward(cell)

	var buf bytes.Buffer
	if err := autograd.SaveParams(&buf, cell.Params()...); err != nil {
		t.Fatal(err)
	}

	// A differently initialized cell must reproduce the original's output
	// after loading.
	cell2 := NewLSTMCell(3, 5, rand.New(rand.NewSource(61)))
	if err := autograd.LoadParams(&buf, cell2.Params()...); err != nil {
		t.Fatal(err)
	}
	got := forward(cell2)
	for i := range want.Data {
		if want.Data[i] != got.Data[i] {
			t.Fatalf("output %d differs after load: %g vs %g", i, want.Data[i], got.Data[i])
		}
	}

	// Shape mismatches must be rejected.
	if err := autograd.SaveParams(&buf, cell.Params()...); err != nil {
		t.Fatal(err)
	}
	other := NewLSTMCell(3, 6, rand.New(rand.NewSource(67)))
	if err := autograd.LoadParams(&buf, other.Params()...); err == nil {
		t.Error("loading into a different architecture should fail")
	}
}

package autograd

import (
	"bytes"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/optim"
)

// randTensor returns a tensor of the given shape with small normal values.
func randTensor(rng *rand.Rand, shape ...int) *tensai.Tensor {
	t := tensai.NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = tensai.Float(rng.NormFloat64()) * 0.5
	}
	return t
}

// posTensor returns a tensor whose values sit in [1, 2), so division and
// logarithms stay well conditioned for a finite-difference check.
func posTensor(rng *rand.Rand, shape ...int) *tensai.Tensor {
	t := tensai.NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = 1 + tensai.Float(rng.Float64())
	}
	return t
}

// TestTensorOpGradients differentiates every n-dimensional op against
// finite differences. Each case makes one tensor the parameter and reduces
// the graph to a scalar; a plain Sum would be blind to any op whose row
// gradient is constant, so the reductions are weighted where that matters.
func TestTensorOpGradients(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	x := randTensor(rng, 2, 3, 4)       // (batch, seq, model)
	w := randTensor(rng, 4, 5)          // a shared projection
	weights := randTensor(rng, 2, 3, 5) // weighting for the final reduction
	bias := randTensor(rng, 1, 1, 5)    // broadcast over batch and sequence
	denom := posTensor(rng, 1, 1, 5)
	k := randTensor(rng, 2, 6, 4)
	attnW := randTensor(rng, 2, 3, 6)
	gain := randTensor(rng, 5)
	shift := randTensor(rng, 5)
	table := randTensor(rng, 7, 4)
	xt := randTensor(rng, 3, 2, 4)    // weighting in the permuted layout
	pooled := randTensor(rng, 2, 4)   // weighting for a reduced axis
	numer := randTensor(rng, 1, 1, 5) // a dividend to differentiate
	ids := []int{3, 0, 6, 0, 1, 5}    // a repeat, so the scatter must accumulate

	cases := []struct {
		name  string
		param *tensai.Tensor
		build func() (*Node, *Node)
	}{
		{"matmul-broadcast", w, func() (*Node, *Node) {
			// (2,3,4) x (4,5): the weight is broadcast across the batch, so
			// its gradient has to be summed back over that axis.
			p := Param(w)
			return Input(x).MatMul(p).Mul(Input(weights)).Sum(), p
		}},
		{"matmul-stacked", k, func() (*Node, *Node) {
			// (2,3,4) x (2,4,6): a real batch of distinct matrices.
			p := Param(k)
			return Input(x).MatMul(p.T()).Mul(Input(attnW)).Sum(), p
		}},
		{"add-broadcast", bias, func() (*Node, *Node) {
			p := Param(bias)
			return Input(x).MatMul(Input(w)).Add(p).Tanh().Mul(Input(weights)).Sum(), p
		}},
		{"sub-broadcast", bias, func() (*Node, *Node) {
			p := Param(bias)
			return Input(weights).Sub(p).Mul(Input(weights)).Sum(), p
		}},
		{"mul-broadcast", bias, func() (*Node, *Node) {
			p := Param(bias)
			return Input(weights).Mul(p).Tanh().Sum(), p
		}},
		{"div-numerator", numer, func() (*Node, *Node) {
			p := Param(numer)
			return p.Div(Input(denom)).Mul(Input(weights)).Sum(), p
		}},
		{"div-denominator", denom, func() (*Node, *Node) {
			p := Param(denom)
			return Input(weights).Div(p).Mul(Input(weights)).Sum(), p
		}},
		{"scale-neg", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).Scale(0.75).Neg().Mul(Input(weights)).Sum(), p
		}},
		{"transpose-perm", x, func() (*Node, *Node) {
			p := Param(x)
			return p.Transpose(1, 0, 2).Mul(Input(xt)).Sum(), p
		}},
		{"reshape", x, func() (*Node, *Node) {
			p := Param(x)
			return p.Reshape(6, 4).MatMul(Input(w)).Reshape(2, 3, 5).Mul(Input(weights)).Sum(), p
		}},
		{"sumaxis", x, func() (*Node, *Node) {
			p := Param(x)
			return p.SumAxis(1, false).Mul(Input(pooled)).Sum(), p
		}},
		{"sumaxis-keepdims", x, func() (*Node, *Node) {
			p := Param(x)
			return p.SumAxis(-1, true).Tanh().Sum(), p
		}},
		{"meanaxis", x, func() (*Node, *Node) {
			p := Param(x)
			return p.MeanAxis(0, false).Tanh().Sum(), p
		}},
		{"softmax-lastaxis", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).Softmax().Mul(Input(weights)).Sum(), p
		}},
		{"gelu", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).GELU().Mul(Input(weights)).Sum(), p
		}},
		{"leakyrelu", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).LeakyReLU(0.1).Mul(Input(weights)).Sum(), p
		}},
		{"exp", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).Exp().Mul(Input(weights)).Sum(), p
		}},
		{"log", denom, func() (*Node, *Node) {
			p := Param(denom)
			return p.Log().Mul(Input(bias)).Sum(), p
		}},
		{"layernorm-input", x, func() (*Node, *Node) {
			p := Param(x)
			return p.MatMul(Input(w)).LayerNorm(nil, nil, 1e-5).Mul(Input(weights)).Sum(), p
		}},
		{"layernorm-gain", gain, func() (*Node, *Node) {
			p := Param(gain)
			return Input(x).MatMul(Input(w)).LayerNorm(p, nil, 1e-5).Mul(Input(weights)).Sum(), p
		}},
		{"layernorm-bias", shift, func() (*Node, *Node) {
			p := Param(shift)
			return Input(x).MatMul(Input(w)).LayerNorm(Input(gain), p, 1e-5).Mul(Input(weights)).Sum(), p
		}},
		{"embed", table, func() (*Node, *Node) {
			p := Param(table)
			return p.Embed(ids, 2, 3).Mul(Input(x)).Sum(), p
		}},
	}
	for _, tc := range cases {
		checkParamGrad(t, tc.param, tc.build, tc.name)
	}
}

// TestTensorCrossEntropy checks the loss over a (batch, seq, vocab) logit
// stack, which is the shape a language model produces.
func TestTensorCrossEntropy(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	x := randTensor(rng, 2, 3, 4)
	w := randTensor(rng, 4, 5)
	labels := []int{0, 4, 2, 1, 3, 3}
	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		return Input(x).MatMul(p).CrossEntropy(labels), p
	}, "cross-entropy-3d")
}

// TestBroadcastGradientReduction pins the reduction itself: a bias
// broadcast over both leading axes must collect every element that used it.
func TestBroadcastGradientReduction(t *testing.T) {
	b := tensai.NewTensor(1, 1, 3)
	p := Param(b)
	x := tensai.NewTensor(2, 4, 3)
	Input(x).Add(p).Sum().Backward()
	for i, g := range p.Grad.Data {
		if g != 8 { // 2 x 4 positions share each bias element
			t.Errorf("bias grad %d = %g, want 8", i, g)
		}
	}
}

// TestEmbedAccumulatesRepeats checks that a token appearing twice gets both
// gradients scattered into the same table row.
func TestEmbedAccumulatesRepeats(t *testing.T) {
	table := tensai.NewTensor(3, 2)
	p := Param(table)
	p.Embed([]int{1, 1, 2}).Sum().Backward()
	want := []tensai.Float{0, 0, 2, 2, 1, 1}
	for i, g := range p.Grad.Data {
		if g != want[i] {
			t.Errorf("table grad %d = %g, want %g", i, g, want[i])
		}
	}
}

// TestLayerNormMatchesDefinition checks the forward pass against the plain
// definition of the statistics it normalizes by.
func TestLayerNormMatchesDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	x := randTensor(rng, 2, 3, 4)
	out := Input(x).LayerNorm(nil, nil, 1e-5)
	for r := 0; r < 6; r++ {
		row := out.Value.Data[r*4 : (r+1)*4]
		var mean, sq tensai.Float
		for _, v := range row {
			mean += v
		}
		mean /= 4
		for _, v := range row {
			sq += v * v
		}
		if math.Abs(float64(mean)) > 1e-4 {
			t.Errorf("row %d mean = %g, want 0", r, mean)
		}
		if math.Abs(float64(sq/4-1)) > 1e-3 {
			t.Errorf("row %d variance = %g, want 1", r, sq/4)
		}
	}
}

// TestTensorParamsRoundTrip saves and reloads parameters of mixed rank.
func TestTensorParamsRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	params := []*Node{
		Param(randTensor(rng, 2, 3, 4)),
		Param(randTensor(rng, 5)),
		Param(tensai.RandomMatrix(3, 2, rng)),
	}
	var buf bytes.Buffer
	if err := SaveParams(&buf, params...); err != nil {
		t.Fatal(err)
	}
	saved := buf.String()
	// Rank-2 parameters keep the Rows/Cols encoding older checkpoints use.
	if !strings.Contains(saved, `"Rows":3,"Cols":2`) {
		t.Errorf("2-D parameter lost its Rows/Cols encoding:\n%s", saved)
	}
	if !strings.Contains(saved, `"Shape":[2,3,4]`) {
		t.Errorf("3-D parameter lost its shape:\n%s", saved)
	}

	restored := []*Node{
		Param(tensai.NewTensor(2, 3, 4)),
		Param(tensai.NewTensor(5)),
		Param(tensai.NewMatrix(3, 2)),
	}
	if err := LoadParams(strings.NewReader(saved), restored...); err != nil {
		t.Fatal(err)
	}
	for i, p := range params {
		for j, v := range p.Value.Data {
			if restored[i].Value.Data[j] != v {
				t.Fatalf("param %d element %d: got %g, want %g", i, j, restored[i].Value.Data[j], v)
			}
		}
	}

	// A rank mismatch must be rejected rather than silently reinterpreted.
	if err := LoadParams(strings.NewReader(saved), Param(tensai.NewTensor(24))); err == nil {
		t.Error("loading a 3-D parameter into a 1-D one should fail")
	}
}

// TestLegacyParamsLoad reads the Rows/Cols encoding written before the
// engine went n-dimensional.
func TestLegacyParamsLoad(t *testing.T) {
	const old = `{"params":[{"Rows":2,"Cols":2,"Data":[1,2,3,4]}]}`
	p := Param(tensai.NewMatrix(2, 2))
	if err := LoadParams(strings.NewReader(old), p); err != nil {
		t.Fatal(err)
	}
	for i, want := range []tensai.Float{1, 2, 3, 4} {
		if p.Value.Data[i] != want {
			t.Errorf("element %d = %g, want %g", i, p.Value.Data[i], want)
		}
	}
}

// TestAttentionTrainsBatched trains a single-head attention block written
// directly in the n-dimensional API -- batched matmul, causal mask,
// per-token cross-entropy -- on a copy task: every position must predict
// the token at the first position. Getting there requires the gradient to
// flow through the attention weights, not just the projections.
func TestAttentionTrainsBatched(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	const (
		batch, seq, model, vocab = 4, 4, 16, 5
	)
	tokens := make([]int, batch*seq)
	labels := make([]int, batch*seq)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			tokens[b*seq+s] = rng.Intn(vocab)
		}
		for s := 0; s < seq; s++ {
			labels[b*seq+s] = tokens[b*seq] // always the first token
		}
	}

	embed := Param(randTensor(rng, vocab, model))
	wq := Param(randTensor(rng, model, model))
	wk := Param(randTensor(rng, model, model))
	wv := Param(randTensor(rng, model, model))
	wOut := Param(randTensor(rng, model, vocab))
	scale := 1 / kernels.SqrtF(tensai.Float(model))

	// Causal mask: -inf above the diagonal, broadcast over the batch.
	mask := tensai.NewTensor(1, seq, seq)
	for i := 0; i < seq; i++ {
		for j := i + 1; j < seq; j++ {
			mask.Data[i*seq+j] = tensai.Float(math.Inf(-1))
		}
	}

	forward := func() *Node {
		x := embed.Embed(tokens, batch, seq) // (batch, seq, model)
		q, kk, v := x.MatMul(wq), x.MatMul(wk), x.MatMul(wv)
		att := q.MatMul(kk.T()).Scale(scale).Add(Input(mask)).Softmax()
		return att.MatMul(v).MatMul(wOut)
	}

	trainer := NewTrainer(optim.NewAdam(0.05), embed, wq, wk, wv, wOut)
	var lossVal tensai.Float
	for step := 0; step < 400; step++ {
		lossVal = trainer.Step(forward().CrossEntropy(labels))
	}
	if lossVal > 0.1 {
		t.Fatalf("batched attention failed to learn the copy task: loss=%g", lossVal)
	}

	logits := forward()
	for i, want := range labels {
		row := logits.Value.Data[i*vocab : (i+1)*vocab]
		best := 0
		for j, v := range row {
			if v > row[best] {
				best = j
			}
		}
		if best != want {
			t.Errorf("position %d: predicted %d, want %d", i, best, want)
		}
	}
}

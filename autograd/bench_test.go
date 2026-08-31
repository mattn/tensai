package autograd

import (
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/optim"
)

// benchModel is one transformer-ish block over a (batch, seq, model)
// activation: embed, project, attend, project out, cross-entropy. It is the
// shape the n-dimensional engine exists for, and big enough that its
// buffers -- not its bookkeeping -- dominate what a step allocates.
func benchModel(rng *rand.Rand) (params []*Node, loss func() *Node) {
	const (
		batch, seq, model, vocab = 8, 16, 64, 32
	)
	embed := Param(tensai.RandomMatrix(vocab, model, rng).Tensor())
	wq := Param(tensai.RandomMatrix(model, model, rng).Tensor())
	wk := Param(tensai.RandomMatrix(model, model, rng).Tensor())
	wv := Param(tensai.RandomMatrix(model, model, rng).Tensor())
	wOut := Param(tensai.RandomMatrix(model, vocab, rng).Tensor())
	params = []*Node{embed, wq, wk, wv, wOut}

	tokens := make([]int, batch*seq)
	labels := make([]int, batch*seq)
	for i := range tokens {
		tokens[i] = rng.Intn(vocab)
		labels[i] = rng.Intn(vocab)
	}
	loss = func() *Node {
		x := embed.Embed(tokens, batch, seq).LayerNorm(nil, nil, 1e-5)
		q, k, v := x.MatMul(wq), x.MatMul(wk), x.MatMul(wv)
		att := q.MatMul(k.T()).Scale(0.125).Softmax()
		return att.MatMul(v).MatMul(wOut).CrossEntropy(labels)
	}
	return params, loss
}

func BenchmarkStep(b *testing.B) {
	params, loss := benchModel(rand.New(rand.NewSource(1)))
	trainer := NewTrainer(optim.NewAdam(0.01), params...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer.Step(loss())
	}
}

// BenchmarkStepTaped is the same step with the buffers recycled.
func BenchmarkStepTaped(b *testing.B) {
	params, loss := benchModel(rand.New(rand.NewSource(1)))
	trainer := NewTrainer(optim.NewAdam(0.01), params...)
	tape := NewTape()
	tape.Bind(params...)
	trainer.Step(loss()) // fill the pool
	tape.Reset()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer.Step(loss())
		tape.Reset()
	}
}

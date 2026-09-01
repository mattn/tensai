package llm

import (
	"math/rand"
	"testing"

	tensai "github.com/mattn/tensai"
)

// A model assembled the way the safetensors loader assembles one -- the
// blocks made and filled in place, with none of the gguf loader's
// per-layer shaping -- has to decode. It did not: a layer's "whose cache
// do I attend against" started at zero, which named layer 0 rather than
// itself, so no layer wrote a cache and the first attention read an
// empty one.
func TestZeroValueBlockKeepsItsOwnCache(t *testing.T) {
	const layers, hidden, heads, kvHeads, ff = 2, 8, 2, 1, 16
	head := hidden / heads
	rng := rand.New(rand.NewSource(1))
	mat := func(r, c int) *tensai.Matrix {
		m := tensai.NewMatrix(r, c)
		for i := range m.Data {
			m.Data[i] = float32(rng.NormFloat64()) * 0.1
		}
		return m
	}
	vec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	m := &qwen{
		cfg: config{
			ModelType: "qwen2", Layers: layers, HiddenSize: hidden, Heads: heads,
			KVHeads: kvHeads, Intermediate: ff, RMSEps: 1e-6, RopeTheta: 10000,
			Vocab: 4, MaxPos: 64,
		},
		headSz: head,
	}
	m.embed = tensai.NewTensor(4, hidden)
	for i := range m.embed.Data {
		m.embed.Data[i] = float32(rng.NormFloat64())
	}
	m.normW = vec(hidden)
	m.lmT = mat(hidden, 4)
	m.blocks = make([]qblock, layers) // exactly what loadQwen does
	for i := range m.blocks {
		b := &m.blocks[i]
		b.ln1, b.ln2 = vec(hidden), vec(hidden)
		b.wQKV = mat(hidden, heads*head+2*kvHeads*head)
		b.wo = mat(heads*head, hidden)
		b.wGU = mat(hidden, 2*ff)
		b.wDown = mat(ff, hidden)
	}
	m.initRopeFreqs()

	// Prefill is where the empty cache was read, and decode has to agree
	// with it: feeding the same tokens one at a time lands in the same
	// place.
	batch := m.prefill([]int{0, 1, 2}, 0)
	m.reset()
	var one []float32
	for i, tok := range []int{0, 1, 2} {
		one = m.step(tok, i)
	}
	for i := range batch {
		if diff := batch[i] - one[i]; diff > 1e-4 || diff < -1e-4 {
			t.Fatalf("logit %d: %v prefilled, %v decoded", i, batch[i], one[i])
		}
	}
	for i := range m.blocks {
		if got := len(m.blocks[i].kc); got != 3 {
			t.Errorf("layer %d holds %d cache rows, want 3", i, got)
		}
	}
}

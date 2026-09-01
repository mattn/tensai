// Command tinygpt trains a small character-level transformer -- token and
// position embeddings, pre-norm blocks with multi-head causal attention and
// a GELU feed-forward, a final norm and an output projection -- and then
// generates text from it.
//
// Everything below is written directly against the n-dimensional autograd
// engine: activations are (batch, sequence, model) tensors, the per-head
// split is a Reshape plus a Transpose, and every attention score in the
// batch is one batched MatMul. The gradients come from the engine, so the
// model is only its forward pass.
//
// With -gpu (and a wgpu build) the whole block trains on the device --
// values, gradients and the Adam update all stay there, and only the loss
// comes back each step. Whether that is faster depends on the shape: at the
// default size the tensors are too small to keep a GPU busy and the AVX2
// kernels win, while a wider model crosses over. On an AMD 780M:
//
//	                                        CPU        -gpu
//	default (model 64, batch 8, seq 32)     24ms/step  72ms/step
//	-model 256 -heads 8 -batch 16 -seq 64  317ms/step 145ms/step
//
// The losses are identical either way, which is the point of the check.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/optim"
)

// The shape of the model. They are variables rather than constants so the
// flags can scale it: the CPU wins at the default size, where the tensors
// are too small to keep a GPU busy, and the device pulls ahead as the model
// grows -- see the -gpu flag below.
var (
	seqLen    = 32 // tokens the model sees at once
	dModel    = 64
	nHeads    = 4
	headDim   = dModel / nHeads
	dFF       = 4 * dModel
	nBlocks   = 2
	batchSize = 8
)

const eps = 1e-5

// block is one pre-norm transformer block.
type block struct {
	ln1g, ln1b     *autograd.Node // (dModel)
	wq, wk, wv, wo *autograd.Node // (dModel, dModel)
	ln2g, ln2b     *autograd.Node // (dModel)
	w1             *autograd.Node // (dModel, dFF)
	w2             *autograd.Node // (dFF, dModel)
}

func newBlock(rng *rand.Rand) *block {
	return &block{
		ln1g: autograd.Param(ones(dModel)),
		ln1b: autograd.Param(tensai.NewTensor(dModel)),
		wq:   autograd.Param(tensai.RandomMatrix(dModel, dModel, rng)),
		wk:   autograd.Param(tensai.RandomMatrix(dModel, dModel, rng)),
		wv:   autograd.Param(tensai.RandomMatrix(dModel, dModel, rng)),
		wo:   autograd.Param(tensai.RandomMatrix(dModel, dModel, rng)),
		ln2g: autograd.Param(ones(dModel)),
		ln2b: autograd.Param(tensai.NewTensor(dModel)),
		w1:   autograd.Param(tensai.RandomMatrix(dModel, dFF, rng)),
		w2:   autograd.Param(tensai.RandomMatrix(dFF, dModel, rng)),
	}
}

func (b *block) params() []*autograd.Node {
	return []*autograd.Node{b.ln1g, b.ln1b, b.wq, b.wk, b.wv, b.wo, b.ln2g, b.ln2b, b.w1, b.w2}
}

// heads splits the model axis into heads and moves them in front of the
// sequence, turning (batch, seq, model) into (batch, head, seq, headDim) so
// that one MatMul attends every head of every sequence at once.
func heads(x *autograd.Node, batch int) *autograd.Node {
	return x.Reshape(batch, seqLen, nHeads, headDim).Transpose(0, 2, 1, 3)
}

// forward runs one block over a (batch, seq, model) activation. mask is the
// causal (1, 1, seq, seq) constant.
func (b *block) forward(x, mask *autograd.Node, batch int) *autograd.Node {
	h := x.LayerNorm(b.ln1g, b.ln1b, eps)
	q, k, v := heads(h.MatMul(b.wq), batch), heads(h.MatMul(b.wk), batch), heads(h.MatMul(b.wv), batch)

	// (batch, head, seq, seq) scores, masked so a position only sees what
	// came before it, then the weighted sum of the values.
	att := q.MatMul(k.T()).Scale(1 / float32(math.Sqrt(float64(headDim)))).Add(mask).Softmax()
	merged := att.MatMul(v).Transpose(0, 2, 1, 3).Reshape(batch, seqLen, dModel)
	x = x.Add(merged.MatMul(b.wo))

	h = x.LayerNorm(b.ln2g, b.ln2b, eps)
	return x.Add(h.MatMul(b.w1).GELU().MatMul(b.w2))
}

// model is the whole network.
type model struct {
	tok        *autograd.Node // (vocab, dModel)
	pos        *autograd.Node // (1, seq, dModel)
	blocks     []*block
	lnfg, lnfb *autograd.Node // (dModel)
	wOut       *autograd.Node // (dModel, vocab)
	mask       *autograd.Node // (1, 1, seq, seq)
	tape       *autograd.Tape
	vocab      []rune
	index      map[rune]int
}

func newModel(vocab []rune, rng *rand.Rand) *model {
	m := &model{
		tok:   autograd.Param(tensai.RandomMatrix(len(vocab), dModel, rng)),
		pos:   autograd.Param(reshape(tensai.RandomMatrix(seqLen, dModel, rng).Tensor(), 1, seqLen, dModel)),
		lnfg:  autograd.Param(ones(dModel)),
		lnfb:  autograd.Param(tensai.NewTensor(dModel)),
		wOut:  autograd.Param(tensai.RandomMatrix(dModel, len(vocab), rng)),
		mask:  autograd.Input(causalMask(seqLen)),
		vocab: vocab,
		index: make(map[rune]int, len(vocab)),
	}
	for i, r := range vocab {
		m.index[r] = i
	}
	for i := 0; i < nBlocks; i++ {
		m.blocks = append(m.blocks, newBlock(rng))
	}
	return m
}

func (m *model) params() []*autograd.Node {
	ps := []*autograd.Node{m.tok, m.pos, m.lnfg, m.lnfb, m.wOut}
	for _, b := range m.blocks {
		ps = append(ps, b.params()...)
	}
	return ps
}

// forward maps batch*seqLen token ids to (batch, seq, vocab) logits.
func (m *model) forward(tokens []int, batch int) *autograd.Node {
	x := m.tok.Embed(tokens, batch, seqLen).Add(m.pos) // the position row broadcasts over the batch
	for _, b := range m.blocks {
		x = b.forward(x, m.mask, batch)
	}
	return x.LayerNorm(m.lnfg, m.lnfb, eps).MatMul(m.wOut)
}

// batchAt draws random windows out of the text: each row predicts the next
// character at every position.
func (m *model) batchAt(text []rune, rng *rand.Rand) (tokens, labels []int) {
	tokens = make([]int, 0, batchSize*seqLen)
	labels = make([]int, 0, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		p := rng.Intn(len(text) - seqLen - 1)
		for t := 0; t < seqLen; t++ {
			tokens = append(tokens, m.index[text[p+t]])
			labels = append(labels, m.index[text[p+t+1]])
		}
	}
	return tokens, labels
}

// generate continues the seed text one character at a time. The window is
// always seqLen tokens wide, so the position embedding always lines up.
func (m *model) generate(seed []rune, n int, temperature float32, rng *rand.Rand) string {
	window := make([]int, seqLen)
	for i, r := range seed[len(seed)-seqLen:] {
		window[i] = m.index[r]
	}
	var sb strings.Builder
	vocab := len(m.vocab)
	for i := 0; i < n; i++ {
		logits := m.forward(window, 1).Scale(1 / temperature).Softmax().Value()
		// The next character is the distribution at the last position. It
		// is read before the tape recycles the buffer it lives in.
		next := sample(logits.Data[(seqLen-1)*vocab:], rng)
		m.tape.Reset()
		sb.WriteRune(m.vocab[next])
		copy(window, window[1:])
		window[seqLen-1] = next
	}
	return sb.String()
}

func sample(probs []tensai.Float, rng *rand.Rand) int {
	x := float32(rng.Float64())
	for i, p := range probs {
		x -= p
		if x <= 0 {
			return i
		}
	}
	return len(probs) - 1
}

// causalMask is zero on and below the diagonal and -Inf above it, so
// softmax gives future positions no weight at all.
func causalMask(n int) *tensai.Tensor {
	mask := tensai.NewTensor(1, 1, n, n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			mask.Data[i*n+j] = tensai.Float(math.Inf(-1))
		}
	}
	return mask
}

func ones(n int) *tensai.Tensor {
	t := tensai.NewTensor(n)
	for i := range t.Data {
		t.Data[i] = 1
	}
	return t
}

func reshape(t *tensai.Tensor, shape ...int) *tensai.Tensor {
	out, err := t.Reshape(shape...)
	if err != nil {
		panic(err)
	}
	return out
}

func vocabOf(text []rune) []rune {
	seen := map[rune]bool{}
	var vocab []rune
	for _, r := range text {
		if !seen[r] {
			seen[r] = true
			vocab = append(vocab, r)
		}
	}
	return vocab
}

func main() {
	iters := flag.Int("iters", 1000, "training steps")
	lr := flag.Float64("lr", 0.003, "Adam learning rate")
	temp := flag.Float64("temp", 0.8, "sampling temperature")
	n := flag.Int("n", 400, "characters to generate")
	seed := flag.Int64("seed", 7, "random seed")
	useGPU := flag.Bool("gpu", false, "train on the GPU (needs -tags wgpu24 or wgpu)")
	flag.IntVar(&dModel, "model", dModel, "model width")
	flag.IntVar(&nHeads, "heads", nHeads, "attention heads")
	flag.IntVar(&nBlocks, "blocks", nBlocks, "transformer blocks")
	flag.IntVar(&batchSize, "batch", batchSize, "sequences per step")
	flag.IntVar(&seqLen, "seq", seqLen, "tokens the model sees at once")
	flag.Parse()
	if dModel%nHeads != 0 {
		fmt.Fprintf(os.Stderr, "tinygpt: model width %d is not divisible by %d heads\n", dModel, nHeads)
		os.Exit(1)
	}
	headDim, dFF = dModel/nHeads, 4*dModel

	text := []rune(corpus)
	vocab := vocabOf(text)
	rng := rand.New(rand.NewSource(*seed))
	m := newModel(vocab, rng)

	params := m.params()
	var count int
	for _, p := range params {
		count += len(p.Value().Data)
	}
	fmt.Printf("corpus: %d chars, vocab: %d, parameters: %d\n", len(text), len(vocab), count)

	trainer := autograd.NewTrainer(optim.NewAdam(tensai.Float(*lr)), params...)
	// Every step builds a fresh graph and drops it; the tape hands the last
	// step's buffers back instead of allocating them again.
	tape := autograd.NewTape()
	if *useGPU {
		// With a device the whole block stays there: values, gradients and
		// the Adam update. Only the loss comes back each step.
		dev, err := gpu.Open(gpu.HighPerformance)
		if err != nil {
			fmt.Fprintf(os.Stderr, "tinygpt: %v\n", err)
			os.Exit(1)
		}
		defer dev.Close()
		tape.UseDevice(dev)
		fmt.Printf("training on %s\n", dev.Name())
	}
	tape.Bind(params...)
	m.tape = tape

	start := time.Now()
	for it := 1; it <= *iters; it++ {
		tokens, labels := m.batchAt(text, rng)
		lossVal := trainer.Step(m.forward(tokens, batchSize).CrossEntropy(labels))
		tape.Reset()
		if it == 1 || it%50 == 0 {
			fmt.Printf("iter %4d: loss=%.4f\n", it, lossVal)
		}
	}
	took := time.Since(start)
	fmt.Printf("trained %d steps in %v (%.1fms/step)\n",
		*iters, took.Round(time.Millisecond), float64(took.Milliseconds())/float64(*iters))

	if len(text) < seqLen {
		fmt.Fprintln(os.Stderr, "tinygpt: corpus shorter than the context window")
		os.Exit(1)
	}
	prompt := text[:seqLen]
	fmt.Printf("\nprompt: %q\n\ngenerated:\n%s\n", string(prompt),
		string(prompt)+m.generate(prompt, *n, float32(*temp), rand.New(rand.NewSource(*seed+1))))
}

// The opening of "Alice's Adventures in Wonderland" by Lewis Carroll
// (public domain), the same text _example/charrnn learns.
const corpus = `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.`

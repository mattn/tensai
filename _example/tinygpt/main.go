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
//	-model 256 -heads 8 -batch 16 -seq 64  282ms/step 129ms/step
//
// The losses are identical either way, which is the point of the check.
//
// It trains on its own page of Alice unless -data names a text file, and
// splits the text into characters unless -tokenizer names a tokenizer.json,
// in which case it learns BPE tokens (only the ids the corpus uses, so the
// embedding stays small). -save writes one checkpoint holding the shape,
// the vocabulary, the tokenizer and the weights; -load starts from it,
// generating from -prompt on its own or training on when given -data:
//
//	go run ./_example/tinygpt -data notes.txt -tokenizer tokenizer.json -save notes.json
//	go run ./_example/tinygpt -load notes.json -prompt "The tape" -n 60
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"time"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/optim"
	"github.com/mattn/tensai/tokenizer"
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

// model is the whole network. Its vocabulary is the token ids that occur
// in the text it learns, packed down to 0..len(vocab)-1: a rune per entry
// in character mode, a tokenizer id per entry with -tokenizer, so a 50k
// entry BPE vocabulary costs only the ids the corpus actually uses.
type model struct {
	tok        *autograd.Node // (vocab, dModel)
	pos        *autograd.Node // (1, seq, dModel)
	blocks     []*block
	lnfg, lnfb *autograd.Node // (dModel)
	wOut       *autograd.Node // (dModel, vocab)
	mask       *autograd.Node // (1, 1, seq, seq)
	tape       *autograd.Tape
	vocab      []int       // model index -> token id
	index      map[int]int // token id -> model index
}

func newModel(vocab []int, rng *rand.Rand) *model {
	m := &model{
		tok:   autograd.Param(tensai.RandomMatrix(len(vocab), dModel, rng)),
		pos:   autograd.Param(reshape(tensai.RandomMatrix(seqLen, dModel, rng).Tensor(), 1, seqLen, dModel)),
		lnfg:  autograd.Param(ones(dModel)),
		lnfb:  autograd.Param(tensai.NewTensor(dModel)),
		wOut:  autograd.Param(tensai.RandomMatrix(dModel, len(vocab), rng)),
		mask:  autograd.Input(causalMask(seqLen)),
		vocab: vocab,
		index: make(map[int]int, len(vocab)),
	}
	for i, id := range vocab {
		m.index[id] = i
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

// indexes maps token ids to model indexes, failing on an id the
// vocabulary does not have (text outside what a loaded checkpoint saw).
func (m *model) indexes(ids []int, dec func([]int) string) ([]int, error) {
	out := make([]int, len(ids))
	for i, id := range ids {
		x, ok := m.index[id]
		if !ok {
			return nil, fmt.Errorf("token %q is not in the model's vocabulary", dec([]int{id}))
		}
		out[i] = x
	}
	return out, nil
}

// forward maps batch*seqLen token indexes to (batch, seq, vocab) logits.
func (m *model) forward(tokens []int, batch int) *autograd.Node {
	x := m.tok.Embed(tokens, batch, seqLen).Add(m.pos) // the position row broadcasts over the batch
	for _, b := range m.blocks {
		x = b.forward(x, m.mask, batch)
	}
	return x.LayerNorm(m.lnfg, m.lnfb, eps).MatMul(m.wOut)
}

// batchAt draws random windows out of the text: each row predicts the next
// token at every position.
func (m *model) batchAt(text []int, rng *rand.Rand) (tokens, labels []int) {
	tokens = make([]int, 0, batchSize*seqLen)
	labels = make([]int, 0, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		p := rng.IntN(len(text) - seqLen - 1)
		tokens = append(tokens, text[p:p+seqLen]...)
		labels = append(labels, text[p+1:p+seqLen+1]...)
	}
	return tokens, labels
}

// generate continues the prompt one token at a time and returns the new
// token indexes. A prompt shorter than the context window sits at its
// start: attention is causal, so the next token read at the prompt's last
// position never sees the unused slots after it. Once the window is full
// it slides, and the position embedding still lines up.
func (m *model) generate(prompt []int, n int, temperature float32, rng *rand.Rand) []int {
	window := make([]int, seqLen)
	cur := copy(window, prompt[max(0, len(prompt)-seqLen):])
	out := make([]int, 0, n)
	vocab := len(m.vocab)
	for i := 0; i < n; i++ {
		logits := m.forward(window, 1).Scale(1 / temperature).Softmax().Value()
		// The next token is the distribution at the last filled position.
		// It is read before the tape recycles the buffer it lives in.
		next := sample(logits.Data[(cur-1)*vocab:cur*vocab], rng)
		m.tape.Reset()
		out = append(out, next)
		if cur < seqLen {
			window[cur] = next
			cur++
		} else {
			copy(window, window[1:])
			window[seqLen-1] = next
		}
	}
	return out
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

// vocabOf lists the distinct token ids in the order they first appear.
func vocabOf(ids []int) []int {
	seen := map[int]bool{}
	var vocab []int
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			vocab = append(vocab, id)
		}
	}
	return vocab
}

// codec turns text into token ids and back: runes by default, or a
// tokenizer.json's BPE with -tokenizer.
type codec struct {
	tok *tokenizer.Tokenizer
	raw json.RawMessage // the tokenizer.json, kept for the checkpoint
}

func newCodec(raw []byte) (*codec, error) {
	if len(raw) == 0 {
		return &codec{}, nil
	}
	tok, err := tokenizer.Parse(raw)
	if err != nil {
		return nil, err
	}
	return &codec{tok: tok, raw: raw}, nil
}

func (c *codec) encode(s string) []int {
	if c.tok != nil {
		return c.tok.Encode(s)
	}
	ids := make([]int, 0, len(s))
	for _, r := range s {
		ids = append(ids, int(r))
	}
	return ids
}

func (c *codec) decode(ids []int) string {
	if c.tok != nil {
		return c.tok.Decode(ids)
	}
	rs := make([]rune, len(ids))
	for i, id := range ids {
		rs[i] = rune(id)
	}
	return string(rs)
}

// checkpoint is everything -load needs to rebuild the model without the
// flags or files it was trained with: the shape, the vocabulary, the
// tokenizer itself and the parameters. The Adam moments are not kept, so
// training on from a checkpoint starts them over.
type checkpoint struct {
	Model     int             `json:"model"`
	Heads     int             `json:"heads"`
	Blocks    int             `json:"blocks"`
	Seq       int             `json:"seq"`
	Vocab     []int           `json:"vocab"`
	Tokenizer json.RawMessage `json:"tokenizer,omitempty"`
	Params    json.RawMessage `json:"params"`
}

func save(path string, m *model, c *codec) error {
	var params bytes.Buffer
	if err := autograd.SaveParams(&params, m.params()...); err != nil {
		return err
	}
	b, err := json.Marshal(&checkpoint{
		Model: dModel, Heads: nHeads, Blocks: nBlocks, Seq: seqLen,
		Vocab: m.vocab, Tokenizer: c.raw, Params: params.Bytes(),
	})
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func readCheckpoint(path string) (*checkpoint, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ck checkpoint
	if err := json.Unmarshal(b, &ck); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &ck, nil
}

func main() {
	iters := flag.Int("iters", 1000, "training steps")
	lr := flag.Float64("lr", 0.003, "Adam learning rate")
	temp := flag.Float64("temp", 0.8, "sampling temperature")
	n := flag.Int("n", 400, "tokens to generate")
	seed := flag.Int64("seed", 7, "random seed")
	useGPU := flag.Bool("gpu", false, "train on the GPU (needs -tags wgpu24 or wgpu)")
	dataPath := flag.String("data", "", "train on this text file instead of the built-in corpus")
	tokPath := flag.String("tokenizer", "", "tokenize with this tokenizer.json (BPE) instead of by character")
	savePath := flag.String("save", "", "write a checkpoint here after training")
	loadPath := flag.String("load", "", "start from this checkpoint; without -data it only generates")
	prompt := flag.String("prompt", "", "text to continue (default: the start of the corpus)")
	flag.IntVar(&dModel, "model", dModel, "model width")
	flag.IntVar(&nHeads, "heads", nHeads, "attention heads")
	flag.IntVar(&nBlocks, "blocks", nBlocks, "transformer blocks")
	flag.IntVar(&batchSize, "batch", batchSize, "sequences per step")
	flag.IntVar(&seqLen, "seq", seqLen, "tokens the model sees at once")
	flag.Parse()
	if err := run(*iters, *lr, *temp, *n, *seed, *useGPU, *dataPath, *tokPath, *savePath, *loadPath, *prompt); err != nil {
		fmt.Fprintf(os.Stderr, "tinygpt: %v\n", err)
		os.Exit(1)
	}
}

func run(iters int, lr, temp float64, n int, seed int64, useGPU bool, dataPath, tokPath, savePath, loadPath, prompt string) error {
	// The checkpoint, when there is one, decides the shape and the
	// tokenizer; the flags that would change them are ignored.
	var ck *checkpoint
	var tokJSON []byte
	if loadPath != "" {
		var err error
		if ck, err = readCheckpoint(loadPath); err != nil {
			return err
		}
		if tokPath != "" {
			return fmt.Errorf("-tokenizer cannot change the tokenizer of a checkpoint")
		}
		dModel, nHeads, nBlocks, seqLen = ck.Model, ck.Heads, ck.Blocks, ck.Seq
		tokJSON = ck.Tokenizer
	} else if tokPath != "" {
		var err error
		if tokJSON, err = os.ReadFile(tokPath); err != nil {
			return err
		}
	}
	if dModel%nHeads != 0 {
		return fmt.Errorf("model width %d is not divisible by %d heads", dModel, nHeads)
	}
	headDim, dFF = dModel/nHeads, 4*dModel
	c, err := newCodec(tokJSON)
	if err != nil {
		return err
	}

	// A checkpoint alone only generates; anything else trains, on the
	// built-in corpus unless -data names a file.
	var text []int
	switch {
	case dataPath != "":
		b, err := os.ReadFile(dataPath)
		if err != nil {
			return err
		}
		text = c.encode(string(b))
	case ck == nil:
		text = c.encode(corpus)
	}

	vocab := vocabOf(text)
	if ck != nil {
		vocab = ck.Vocab
	}
	rng := rand.New(rand.NewPCG(uint64(seed), 0))
	m := newModel(vocab, rng)
	params := m.params()
	if ck != nil {
		if err := autograd.LoadParams(bytes.NewReader(ck.Params), params...); err != nil {
			return fmt.Errorf("%s: %w", loadPath, err)
		}
	}
	var count int
	for _, p := range params {
		count += len(p.Value().Data)
	}
	if text != nil {
		fmt.Printf("corpus: %d tokens, ", len(text))
	}
	fmt.Printf("vocab: %d, parameters: %d\n", len(vocab), count)

	var data []int // the corpus as model indexes; nil when only generating
	if text != nil {
		if data, err = m.indexes(text, c.decode); err != nil {
			return err
		}
	}
	if data != nil && len(data) <= seqLen {
		return fmt.Errorf("corpus of %d tokens is not longer than the %d token context window", len(data), seqLen)
	}

	// Every step builds a fresh graph and drops it; the tape hands the last
	// step's buffers back instead of allocating them again.
	tape := autograd.NewTape()
	if useGPU {
		// With a device the whole block stays there: values, gradients and
		// the Adam update. Only the loss comes back each step.
		dev, err := gpu.Open(gpu.HighPerformance)
		if err != nil {
			return err
		}
		defer dev.Close()
		tape.UseDevice(dev)
		fmt.Printf("training on %s\n", dev.Name())
	}
	tape.Bind(params...)
	m.tape = tape

	if data != nil && iters > 0 {
		trainer := autograd.NewTrainer(optim.NewAdam(tensai.Float(lr)), params...)
		start := time.Now()
		for it := 1; it <= iters; it++ {
			tokens, labels := m.batchAt(data, rng)
			lossVal := trainer.Step(m.forward(tokens, batchSize).CrossEntropy(labels))
			tape.Reset()
			if it == 1 || it%50 == 0 {
				fmt.Printf("iter %4d: loss=%.4f\n", it, lossVal)
			}
		}
		took := time.Since(start)
		fmt.Printf("trained %d steps in %v (%.1fms/step)\n",
			iters, took.Round(time.Millisecond), float64(took.Milliseconds())/float64(iters))
	}
	if savePath != "" {
		if err := save(savePath, m, c); err != nil {
			return err
		}
		fmt.Printf("saved %s\n", savePath)
	}
	if n <= 0 {
		return nil
	}

	var seedIdx []int
	if prompt != "" {
		if seedIdx, err = m.indexes(c.encode(prompt), c.decode); err != nil {
			return fmt.Errorf("prompt: %w", err)
		}
	} else if data != nil {
		seedIdx = data[:seqLen]
	}
	if len(seedIdx) == 0 {
		return fmt.Errorf("nothing to continue: give -prompt")
	}
	gen := m.generate(seedIdx, n, float32(temp), rand.New(rand.NewPCG(uint64(seed+1), 0)))
	ids := make([]int, 0, len(seedIdx)+len(gen))
	for _, x := range append(seedIdx, gen...) {
		ids = append(ids, m.vocab[x])
	}
	fmt.Printf("\nprompt: %q\n\ngenerated:\n%s\n", c.decode(ids[:len(seedIdx)]), c.decode(ids))
	return nil
}

// The opening of "Alice's Adventures in Wonderland" by Lewis Carroll
// (public domain), the same text _example/charrnn learns.
const corpus = `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.`

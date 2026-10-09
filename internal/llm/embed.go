package llm

// Text embeddings: a ModernBERT encoder (Ruri v3 and its kin) read from
// a GGUF file turns a text into one vector, the mean of its tokens'
// final hidden states, scaled to unit length so that a dot product is
// the cosine similarity. Every token sees every other in both
// directions; two layers in three see only the 64 tokens to either
// side and rotate with their own RoPE base, the third sees the whole
// text. Nothing is cached between calls, so one Embedder serves any
// number of goroutines at once.

import (
	"fmt"
	"io"
	"math"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/gguf"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/internal/workpool"
	"github.com/mattn/tensai/quant"
	"github.com/mattn/tensai/tokenizer"
)

// Embedder turns texts into embedding vectors.
type Embedder struct {
	g      *gguf.File
	tok    *tokenizer.Tokenizer
	emb    *embedTable
	hs     int // hidden width, and the vector's
	heads  int
	headSz int
	ff     int
	eps    float64
	window int // tokens to either side a local layer sees
	maxPos int
	vocab  int
	bos    int // -1 when the model does not frame its input
	eos    int
	embNrm []float32
	outNrm []float32
	blocks []embedBlock
}

// embedBlock is one encoder layer.
type embedBlock struct {
	attnNorm []float32 // nil on the first layer, which has none
	ffnNorm  []float32
	wQKV     *tensai.Matrix // columns [q | k | v]
	qQKV     *qmat
	wOut     *tensai.Matrix
	qOut     *qmat
	wUp      *tensai.Matrix // columns [input | gate]
	qUp      *qmat
	wDown    *tensai.Matrix
	qDown    *qmat
	local    bool
	freq     []float64 // RoPE frequency per pair of a head
}

// IsEmbeddingModel reports whether the GGUF file at path holds an
// encoder that LoadEmbedder reads, rather than a model that generates.
func IsEmbeddingModel(path string) bool {
	g, err := gguf.Open(path)
	if err != nil {
		return false
	}
	defer g.Close()
	arch, _ := g.String("general.architecture")
	return arch == "modern-bert"
}

// LoadEmbedder reads a ModernBERT GGUF file. bits picks the width the
// projections run at: 0 (and BitsAuto) keeps float32, 8 or 4 quantize.
// float32 is the default because the quantized matmuls round each
// activation row to 7 bits against one scale, and an encoder's rows
// carry outliers that make that cost more than the weights' own
// quantization: on Ruri v3 the cosine to the original model drops from
// 0.998 to 0.993 at int8, which is enough to reorder close search hits.
func LoadEmbedder(path string, bits int, vlog io.Writer) (*Embedder, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	e, err := loadEmbedder(g, bits, vlog)
	if err != nil {
		g.Close()
		return nil, err
	}
	return e, nil
}

func loadEmbedder(g *gguf.File, bits int, vlog io.Writer) (*Embedder, error) {
	arch, _ := g.String("general.architecture")
	if arch != "modern-bert" {
		return nil, fmt.Errorf("unsupported embedding architecture %q (this speaks modern-bert)", arch)
	}
	if bits == BitsAuto {
		bits = 0
	}
	meta := func(key string) int {
		n, _ := g.Int(arch + "." + key)
		return int(n)
	}
	e := &Embedder{
		g:      g,
		hs:     meta("embedding_length"),
		heads:  meta("attention.head_count"),
		ff:     meta("feed_forward_length"),
		window: meta("attention.sliding_window") / 2,
		maxPos: meta("context_length"),
	}
	layers := meta("block_count")
	if e.hs == 0 || e.heads == 0 || e.ff == 0 || layers == 0 {
		return nil, fmt.Errorf("gguf is missing %s.* dimensions", arch)
	}
	e.headSz = e.hs / e.heads
	e.eps, _ = g.Float(arch + ".attention.layer_norm_epsilon")
	if e.eps == 0 {
		e.eps = 1e-5
	}
	if act, ok := g.String(arch + ".hidden_activation"); ok && act != "gelu" && act != "geglu" {
		return nil, fmt.Errorf("unsupported %s activation %q", arch, act)
	}
	if pool, ok := g.Int(arch + ".pooling_type"); ok && pool != 1 {
		return nil, fmt.Errorf("unsupported %s pooling type %d (mean is 1)", arch, pool)
	}
	period := meta("attention.sliding_window_pattern")
	if period == 0 {
		period = 3
	}
	global, _ := g.Float(arch + ".rope.freq_base")
	local, ok := g.Float(arch + ".rope.freq_base_swa")
	if !ok {
		local = global
	}

	tok, err := embedTokenizer(g)
	if err != nil {
		return nil, err
	}
	e.tok = tok
	e.bos, e.eos = -1, -1
	if v, _ := g.KV("tokenizer.ggml.add_bos_token"); v == true {
		if id, ok := g.Int("tokenizer.ggml.bos_token_id"); ok {
			e.bos = int(id)
		}
	}
	if v, _ := g.KV("tokenizer.ggml.add_eos_token"); v == true {
		if id, ok := g.Int("tokenizer.ggml.eos_token_id"); ok {
			e.eos = int(id)
		}
	}

	vec := func(name string) ([]float32, error) {
		t, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		return t.Data, nil
	}
	// Projections arrive [out, in]; the matmuls want [in, out]. Stored
	// k-quant and Q8_0 blocks repack straight into the quantized layout,
	// which keeps their own rounding instead of adding a second one;
	// anything else goes through float32.
	lin := func(name string) (*tensai.Matrix, *qmat, error) {
		if q := repackDirect(g, name, bits); q != nil {
			g.Release(name)
			return nil, q, nil
		}
		t, err := g.Tensor(name)
		if err != nil {
			return nil, nil, err
		}
		m, err := t.Matrix()
		if err != nil {
			return nil, nil, err
		}
		g.Release(name)
		w := m.T()
		if bits == 0 {
			return w, nil, nil
		}
		return nil, quantizeMat(w, bits), nil
	}
	if e.embNrm, err = vec("token_embd_norm.weight"); err != nil {
		return nil, err
	}
	if e.outNrm, err = vec("output_norm.weight"); err != nil {
		return nil, err
	}
	e.emb = newEmbedTable(g, "token_embd.weight")
	_, shape, ok := g.Info("token_embd.weight")
	if !ok || len(shape) != 2 || shape[1] != e.hs {
		return nil, fmt.Errorf("gguf has no %d-wide token_embd.weight", e.hs)
	}
	e.vocab = shape[0]
	e.blocks = make([]embedBlock, layers)
	for i := range e.blocks {
		b := &e.blocks[i]
		p := fmt.Sprintf("blk.%d.", i)
		if i > 0 {
			if b.attnNorm, err = vec(p + "attn_norm.weight"); err != nil {
				return nil, err
			}
		}
		if b.ffnNorm, err = vec(p + "ffn_norm.weight"); err != nil {
			return nil, err
		}
		if b.wQKV, b.qQKV, err = lin(p + "attn_qkv.weight"); err != nil {
			return nil, err
		}
		if b.wOut, b.qOut, err = lin(p + "attn_output.weight"); err != nil {
			return nil, err
		}
		if b.wUp, b.qUp, err = lin(p + "ffn_up.weight"); err != nil {
			return nil, err
		}
		if b.wDown, b.qDown, err = lin(p + "ffn_down.weight"); err != nil {
			return nil, err
		}
		// The first layer and every period-th after it see the whole text.
		b.local = e.window > 0 && i%period != 0
		theta := global
		if b.local {
			theta = local
		}
		b.freq = make([]float64, e.headSz/2)
		for j := range b.freq {
			b.freq[j] = math.Pow(theta, -2*float64(j)/float64(e.headSz))
		}
	}
	fmt.Fprintf(vlog, "%s: %d layers, hidden %d, %d heads, ff %d, window %d, ctx %d, int%d\n",
		arch, layers, e.hs, e.heads, e.ff, 2*e.window, e.maxPos, bits)
	return e, nil
}

// repackDirect repacks one stored tensor into the bits-wide layout
// without a float32 detour, or returns nil when its encoding has no
// such path.
func repackDirect(g *gguf.File, name string, bits int) *qmat {
	typ, shape, ok := g.Info(name)
	if !ok || len(shape) != 2 || bits == 0 {
		return nil
	}
	out, in := shape[0], shape[1]
	var q8 *quant.Q8GMatrix
	var q4 *quant.Q4Matrix
	var pack func(raw []byte)
	switch {
	case typ == "Q8_0" && bits == 8 && in%32 == 0:
		q8 = quant.NewQ8GMatrix(in, out, 0)
		pack = func(raw []byte) { repackQ8(q8, raw, out, in, 0, nil) }
	case in%256 != 0:
		return nil
	case typ == "Q4_K" && bits == 8:
		q8 = quant.NewQ8GMatrix(in, out, 0)
		pack = func(raw []byte) { repackQ4K8(q8, raw, out, in, 0, nil) }
	case typ == "Q4_K":
		q4 = quant.NewQ4Matrix(in, out, 32, true)
		pack = func(raw []byte) { repackQ4K(q4, raw, out, in, 0, nil) }
	case typ == "Q5_K" && bits == 8:
		q8 = quant.NewQ8GMatrix(in, out, 0)
		pack = func(raw []byte) { repackQ5K8(q8, raw, out, in, 0, nil) }
	case typ == "Q5_K":
		q4 = quant.NewQ4Matrix(in, out, 32, true)
		pack = func(raw []byte) { repackQ5K4(q4, raw, out, in, 0, nil) }
	case typ == "Q6_K" && bits == 8:
		q8 = quant.NewQ8GMatrix(in, out, 16)
		pack = func(raw []byte) { repackQ6K(q8, raw, out, in, 0, nil) }
	case typ == "Q6_K":
		q4 = quant.NewQ4Matrix(in, out, 32, true)
		pack = func(raw []byte) { repackQ6K4(q4, raw, out, in, 0, nil) }
	default:
		return nil
	}
	_, raw, err := g.RawTensor(name)
	if err != nil {
		return nil
	}
	pack(raw)
	if q8 != nil {
		return qmatQ8G(q8)
	}
	return qmatQ4(q4)
}

// embedTokenizer builds the tokenizer an encoder GGUF embeds. A
// SentencePiece vocabulary here is a Unigram model (Ruri v3's), searched
// as one rather than merged pairwise the way llama.cpp does, and without
// the leading space llama.cpp assumes when the file does not say: the
// tokenizer.json it was converted from prepends none.
func embedTokenizer(g *gguf.File) (*tokenizer.Tokenizer, error) {
	if model, _ := g.String("tokenizer.ggml.model"); model != "llama" {
		return ggufTokenizer(g)
	}
	tokens := g.Strings("tokenizer.ggml.tokens")
	scores := g.Floats("tokenizer.ggml.scores")
	types64 := g.Ints("tokenizer.ggml.token_type")
	types := make([]int32, len(types64))
	for i, v := range types64 {
		types[i] = int32(v)
	}
	prefix := false
	if v, ok := g.KV("tokenizer.ggml.add_space_prefix"); ok {
		prefix, _ = v.(bool)
	}
	return tokenizer.NewUnigram(tokens, scores, types, prefix)
}

// Close releases the model file.
func (e *Embedder) Close() error { return e.g.Close() }

// Dim is the length of every vector Embed returns.
func (e *Embedder) Dim() int { return e.hs }

// Vocab is the number of token ids EmbedIDs accepts.
func (e *Embedder) Vocab() int { return e.vocab }

// Tokenize returns the ids Embed runs the encoder on, the model's
// framing tokens included.
func (e *Embedder) Tokenize(text string) []int {
	ids := e.tok.Encode(text)
	if e.bos >= 0 {
		ids = append([]int{e.bos}, ids...)
	}
	if e.eos >= 0 {
		ids = append(ids, e.eos)
	}
	return ids
}

// Embed returns text's unit-length embedding and how many tokens the
// encoder read.
func (e *Embedder) Embed(text string) ([]float32, int, error) {
	ids := e.Tokenize(text)
	v, err := e.EmbedIDs(ids)
	return v, len(ids), err
}

// EmbedIDs runs the encoder over ids as they are.
func (e *Embedder) EmbedIDs(ids []int) ([]float32, error) {
	vs, err := e.EmbedBatch([][]int{ids})
	if err != nil {
		return nil, err
	}
	return vs[0], nil
}

// batchTokens is about how many tokens one pass of EmbedBatch packs.
// Each pass streams every weight once, which on a short text is most of
// the cost, so texts share passes; the cap bounds the activations, at
// about 40KB a token.
var batchTokens = 2048

// EmbedBatch embeds several token sequences, packed into as few passes
// over the weights as fit batchTokens. A sequence's positions start at
// zero and its tokens attend only among themselves, so each vector is
// the one EmbedIDs gives it alone.
func (e *Embedder) EmbedBatch(idss [][]int) ([][]float32, error) {
	for i, ids := range idss {
		if len(ids) == 0 {
			return nil, fmt.Errorf("sequence %d: nothing to embed", i)
		}
		if e.maxPos > 0 && len(ids) > e.maxPos {
			return nil, fmt.Errorf("sequence %d: %d tokens is more than the model's %d", i, len(ids), e.maxPos)
		}
		for _, id := range ids {
			if id < 0 || id >= e.vocab {
				return nil, fmt.Errorf("sequence %d: token %d is outside the vocabulary", i, id)
			}
		}
	}
	out := make([][]float32, 0, len(idss))
	for lo := 0; lo < len(idss); {
		hi, n := lo, 0
		for hi < len(idss) && (hi == lo || n+len(idss[hi]) <= batchTokens) {
			n += len(idss[hi])
			hi++
		}
		vs, err := e.forward(idss[lo:hi])
		if err != nil {
			return nil, err
		}
		out = append(out, vs...)
		lo = hi
	}
	return out, nil
}

// forward runs one pass over the sequences laid end to end.
func (e *Embedder) forward(idss [][]int) ([][]float32, error) {
	hs := e.hs
	// span[t] is the stretch of rows token t's sequence occupies.
	var span [][2]int
	for _, ids := range idss {
		s := [2]int{len(span), len(span) + len(ids)}
		for range ids {
			span = append(span, s)
		}
	}
	n := len(span)
	x := tensai.NewMatrix(n, hs)
	t := 0
	for _, ids := range idss {
		for _, id := range ids {
			if err := e.emb.row(id, x.Data[t*hs:(t+1)*hs]); err != nil {
				return nil, err
			}
			t++
		}
	}
	a := tensai.NewMatrix(n, hs)
	e.normRows(x, x, e.embNrm)
	attn := tensai.NewMatrix(n, hs)
	gated := tensai.NewMatrix(n, e.ff)
	for i := range e.blocks {
		b := &e.blocks[i]
		if b.attnNorm != nil {
			e.normRows(a, x, b.attnNorm)
		} else {
			copy(a.Data, x.Data)
		}
		qkv := mmb(a, b.wQKV, b.qQKV, nil)
		e.attend(b, qkv, attn, span)
		addInto(x, mmb(attn, b.wOut, b.qOut, nil))

		e.normRows(a, x, b.ffnNorm)
		up := mmb(a, b.wUp, b.qUp, nil)
		ff := e.ff
		workpool.Run(n, 1, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				row := up.Data[t*2*ff : (t+1)*2*ff]
				dst := gated.Data[t*ff : (t+1)*ff]
				kernels.GeluFwd(dst, row[:ff])
				for j, g := range row[ff:] {
					dst[j] *= g
				}
			}
		})
		addInto(x, mmb(gated, b.wDown, b.qDown, nil))
	}
	e.normRows(x, x, e.outNrm)

	out := make([][]float32, len(idss))
	t = 0
	for i, ids := range idss {
		sum := make([]float64, hs)
		for range ids {
			for j, v := range x.Data[t*hs : (t+1)*hs] {
				sum[j] += float64(v)
			}
			t++
		}
		var sq float64
		for _, v := range sum {
			sq += v * v
		}
		v := make([]float32, hs)
		if sq > 0 {
			inv := 1 / math.Sqrt(sq)
			for j, s := range sum {
				v[j] = float32(s * inv)
			}
		}
		out[i] = v
	}
	return out, nil
}

// normRows is LayerNorm without a bias over each row of x, into out.
func (e *Embedder) normRows(out, x *tensai.Matrix, w []float32) {
	hs := e.hs
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			layerNormInto(out.Data[t*hs:(t+1)*hs], x.Data[t*hs:(t+1)*hs], w, e.eps)
		}
	})
}

// layerNormInto writes (x - mean) / sqrt(var + eps) * w into out.
func layerNormInto(out, x, w []float32, eps float64) {
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var vr float64
	for _, v := range x {
		d := float64(v) - mean
		vr += d * d
	}
	inv := 1 / math.Sqrt(vr/float64(len(x))+eps)
	for i, v := range x {
		out[i] = float32((float64(v)-mean)*inv) * w[i]
	}
}

// addInto adds y to x element by element.
func addInto(x, y *tensai.Matrix) {
	for i, v := range y.Data {
		x.Data[i] += v
	}
}

// attend rotates the queries and keys in qkv by position and writes
// every head's attention output into out: bidirectional within a
// token's own sequence, and limited to the window on a local layer.
func (e *Embedder) attend(b *embedBlock, qkv, out *tensai.Matrix, span [][2]int) {
	n, hs, d := qkv.Rows, e.hs, e.headSz
	w := 3 * hs
	half := d / 2
	longest := 0
	for _, s := range span {
		longest = max(longest, s[1]-s[0])
	}
	workpool.Run(n, 1, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := qkv.Data[t*w : (t+1)*w]
			pos := float64(t - span[t][0])
			for j, f := range b.freq {
				s, c := math.Sincos(pos * f)
				for h := 0; h < 2*e.heads; h++ {
					v := row[h*d : (h+1)*d]
					x0, x1 := float64(v[j]), float64(v[j+half])
					v[j] = float32(x0*c - x1*s)
					v[j+half] = float32(x1*c + x0*s)
				}
			}
		}
	})
	rows := make([][]float32, n)
	for t := range rows {
		rows[t] = qkv.Data[t*w : (t+1)*w]
	}
	scale := float32(1 / math.Sqrt(float64(d)))
	workpool.Run(n*e.heads, 1, func(lo, hi int) {
		scores := make([]float32, longest)
		for item := lo; item < hi; item++ {
			t, h := item/e.heads, item%e.heads
			from, to := span[t][0], span[t][1]
			if b.local {
				from, to = max(from, t-e.window), min(to, t+e.window+1)
			}
			q := rows[t][h*d : (h+1)*d]
			kOff, vOff := hs+h*d, 2*hs+h*d
			si := scores[:to-from]
			maxs := float32(math.Inf(-1))
			for j := from; j < to; j++ {
				s := kernels.DotVec(q, rows[j][kOff:kOff+d]) * scale
				si[j-from] = s
				maxs = max(maxs, s)
			}
			kernels.ExpShift(si, si, maxs)
			var sum float32
			for _, v := range si {
				sum += v
			}
			inv := 1 / sum
			for i := range si {
				si[i] *= inv
			}
			dst := out.Data[t*hs+h*d : t*hs+(h+1)*d]
			clear(dst)
			kernels.AxpyRows(dst, si, rows[from:to], vOff)
		}
	})
}

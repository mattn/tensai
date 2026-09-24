package qwenimage

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime/debug"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/internal/workpool"
	"github.com/mattn/tensai/tokenizer"
)

// The prompt reaches the denoising transformer as the hidden states of a
// vision-language model, not as tokens. For text-to-image only its
// language half runs, and that half is an ordinary Qwen3 decoder: 36
// layers of grouped-query attention and a SwiGLU feed-forward, each
// behind an RMS norm, with a further norm on every head's queries and
// keys.
//
// Two details decide whether the prompt lands where the transformer
// expects it. The states are taken from the last layer's output, before
// the final norm the checkpoint also carries — that norm belongs to
// token prediction, which is not what the prompt is for. And the
// template's system preamble is dropped afterwards, so what the
// transformer reads starts at the user's own words.
//
// The checkpoint's rotary embedding is the multimodal kind, which gives
// the three position axes their own slices of the frequencies. With no
// image in the prompt all three axes hold the same position, so every
// slice computes the angle a plain rotary embedding would, and this runs
// the plain one.

const (
	teDim       = 4096
	teHeads     = 32
	teKVHeads   = 8
	teHeadDim   = 128
	teKVDim     = teKVHeads * teHeadDim
	teMLP       = 12288
	teLayers    = 36
	teEps       = 1e-6
	teRopeTheta = 5000000
)

type teLayer struct {
	inNorm, postNorm []tensai.Float
	q, k, v, o       *linear
	qNorm, kNorm     []tensai.Float
	gate, up, down   *linear
}

// TextEncoder is the language half of the checkpoint's vision-language
// model. It holds the layers and keeps the file open, because the
// embedding table is 600M parameters and a prompt touches a few dozen
// rows of it.
type TextEncoder struct {
	w      textSource
	layers []*teLayer
	// release unmaps the cache the layers point into, when they came
	// from one.
	release func() error
}

// LoadTextEncoder reads the encoder, quantizing its weights when bits is
// 8. Close it once the prompt is encoded: the denoising transformer
// wants the memory.
func LoadTextEncoder(dir string, bits int) (*TextEncoder, error) {
	return loadTextEncoder(dir, bits, teLayers)
}

// loadTextEncoder can stop short of the checkpoint's layers, which is
// what lets the wiring be checked against a reference on a machine that
// cannot hold all 36.
func loadTextEncoder(dir string, bits, layers int) (*TextEncoder, error) {
	w, err := openTextSource(dir)
	if err != nil {
		return nil, err
	}
	t := &TextEncoder{w: w, layers: make([]*teLayer, layers)}
	for i := range t.layers {
		t.layers[i] = &teLayer{}
	}
	// The embedding table is read from the checkpoint either way, so the
	// cache only stands in for the layers.
	full := bits != 0 && layers == teLayers
	if full {
		if release, err := readCache(dir, bits, t.walk); err == nil {
			t.release = release
			return t, nil
		}
	}
	for i := 0; i < layers; i++ {
		p := fmt.Sprintf("model.language_model.layers.%d.", i)
		l := t.layers[i]
		for _, f := range []struct {
			dst        **linear
			name       string
			rows, cols int
		}{
			{&l.q, p + "self_attn.q_proj.weight", teDim, teDim},
			{&l.k, p + "self_attn.k_proj.weight", teKVDim, teDim},
			{&l.v, p + "self_attn.v_proj.weight", teKVDim, teDim},
			{&l.o, p + "self_attn.o_proj.weight", teDim, teDim},
			{&l.gate, p + "mlp.gate_proj.weight", teMLP, teDim},
			{&l.up, p + "mlp.up_proj.weight", teMLP, teDim},
			{&l.down, p + "mlp.down_proj.weight", teDim, teMLP},
		} {
			if *f.dst, err = loadLinear(w, f.name, f.rows, f.cols, bits, convRotGroup); err != nil {
				w.Close()
				return nil, err
			}
		}
		for _, f := range []struct {
			dst  *[]tensai.Float
			name string
			n    int
		}{
			{&l.inNorm, p + "input_layernorm.weight", teDim},
			{&l.postNorm, p + "post_attention_layernorm.weight", teDim},
			{&l.qNorm, p + "self_attn.q_norm.weight", teHeadDim},
			{&l.kNorm, p + "self_attn.k_norm.weight", teHeadDim},
		} {
			if *f.dst, err = vector(w, f.name, f.n); err != nil {
				w.Close()
				return nil, err
			}
		}
	}
	if full {
		_ = writeCache(dir, bits, t.walk)
	}
	return t, nil
}

// Close releases the checkpoint and, if the layers came from a cache,
// the mapping they point into. Nothing may use the encoder afterwards.
func (t *TextEncoder) Close() error {
	err := t.w.Close()
	if t.release != nil {
		if rerr := t.release(); err == nil {
			err = rerr
		}
		t.release = nil
	}
	return err
}

// embed reads one row per token out of the embedding table, which is far
// too large to hold for the handful of rows a prompt needs.
func (t *TextEncoder) embed(ids []int) (*tensai.Matrix, error) {
	return t.w.embedRows("model.language_model.embed_tokens.weight", ids, teDim)
}

// textSource is where the encoder's weights come from: the diffusers
// checkpoint's shards, or ComfyUI's single file. Either keeps the file
// open for the embedding rows a prompt asks for.
type textSource interface {
	weights
	Close() error
	embedRows(name string, ids []int, dim int) (*tensai.Matrix, error)
}

// openTextSource opens the encoder at path: a directory of shards, or
// one ComfyUI file.
func openTextSource(path string) (textSource, error) {
	if singleFile(path) {
		return openComfy(path, comfyTextName)
	}
	w, err := safetensors.OpenSharded(filepath.Join(path, "model.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	return shardText{w}, nil
}

// shardText reads the diffusers checkpoint, whose table is bfloat16.
type shardText struct{ *safetensors.Shards }

func (s shardText) embedRows(name string, ids []int, dim int) (*tensai.Matrix, error) {
	raw, shape, err := s.Raw(name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[1] != dim {
		return nil, fmt.Errorf("qwenimage: embedding table has shape %v", shape)
	}
	return embedBF16(raw, shape[0], dim, ids)
}

// embedBF16 widens the rows ids of a bfloat16 table.
func embedBF16(raw []byte, rows, dim int, ids []int) (*tensai.Matrix, error) {
	out := tensai.NewMatrix(len(ids), dim)
	for r, id := range ids {
		if id < 0 || id >= rows {
			return nil, fmt.Errorf("qwenimage: token %d is outside the %d-row table", id, rows)
		}
		row := raw[id*dim*2:]
		for c := 0; c < dim; c++ {
			// bfloat16 is the top half of a float32's bits.
			bits := uint32(row[2*c]) | uint32(row[2*c+1])<<8
			out.Data[r*dim+c] = math.Float32frombits(bits << 16)
		}
	}
	return out, nil
}

// rmsNorm scales each row by its root-mean-square, the plain form with
// the weight as stored.
func rmsNormRows(x *tensai.Matrix, w []tensai.Float) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x.Data[r*x.Cols : (r+1)*x.Cols]
			sq := kernels.SquaredDeviations64(row, 0)
			inv := tensai.Float(1 / math.Sqrt(sq/float64(x.Cols)+teEps))
			kernels.ScaleWeights(row, w, inv, 0)
		}
	})
}

// rmsNormPerHead normalizes each head's slice of every token, for the
// separate norms this checkpoint puts on queries and keys.
func rmsNormPerHead(x *tensai.Matrix, w []tensai.Float, heads int) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for h := 0; h < heads; h++ {
				head := x.Data[r*x.Cols+h*teHeadDim:][:teHeadDim]
				sq := kernels.SquaredDeviations64(head, 0)
				inv := tensai.Float(1 / math.Sqrt(sq/teHeadDim+teEps))
				kernels.ScaleWeights(head, w, inv, 0)
			}
		}
	})
}

// teRope rotates each head by its token's position. This checkpoint
// splits a head in half and rotates the halves against each other,
// which is a different convention from the denoising transformer's
// adjacent pairs.
func teRope(x *tensai.Matrix, heads int) {
	cos, sin := teRopeTable(x.Rows)
	teRopeWithTable(x, heads, cos, sin)
}

// teRopeTable computes each position once, shared by all heads and layers.
func teRopeTable(rows int) (cos, sin []tensai.Float) {
	const half = teHeadDim / 2
	cos, sin = make([]tensai.Float, rows*half), make([]tensai.Float, rows*half)
	for j := 0; j < half; j++ {
		freq := math.Pow(teRopeTheta, -2*float64(j)/teHeadDim)
		for r := 0; r < rows; r++ {
			sn, cs := math.Sincos(float64(r) * freq)
			cos[r*half+j], sin[r*half+j] = tensai.Float(cs), tensai.Float(sn)
		}
	}
	return
}
func teRopeWithTable(x *tensai.Matrix, heads int, cos, sin []tensai.Float) {
	const half = teHeadDim / 2
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for h := 0; h < heads; h++ {
				kernels.RopeSplit(x.Data[r*x.Cols+h*teHeadDim:][:teHeadDim], cos[r*half:][:half], sin[r*half:][:half])
			}
		}
	})
}

// teAttention is causal grouped-query attention: four query heads share
// each key and value head.
func teAttention(out, q, k, v *tensai.Matrix) {
	n := q.Rows
	scale := tensai.Float(1 / math.Sqrt(teHeadDim))
	workpool.Run(teHeads, 1, func(lohi, hihi int) {
		scores := make([]tensai.Float, n)
		for h := lohi; h < hihi; h++ {
			qo := h * teHeadDim
			ko := (h / (teHeads / teKVHeads)) * teHeadDim
			for i := 0; i < n; i++ {
				qi := q.Data[i*q.Cols+qo:][:teHeadDim]
				for j := 0; j <= i; j++ {
					s := tensai.DotVec(qi, k.Data[j*k.Cols+ko:][:teHeadDim]) * scale
					scores[j] = s
				}
				kernels.Softmax(scores[:i+1])
				dst := out.Data[i*out.Cols+qo:][:teHeadDim]
				clear(dst)
				for j, e := range scores[:i+1] {
					kernels.Axpy(e, v.Data[j*v.Cols+ko:][:teHeadDim], dst)
				}
			}
		}
	})
}

// Encode runs the prompt's tokens through the encoder and returns the
// last layer's output, one row per token. The caller drops whatever
// preamble the template added.
func (t *TextEncoder) Encode(ids []int) (*tensai.Matrix, error) {
	x, err := t.embed(ids)
	if err != nil {
		return nil, err
	}
	n := len(ids)
	cos, sin := teRopeTable(n)
	norm := tensai.NewMatrix(n, teDim)
	q := tensai.NewMatrix(n, teDim)
	k := tensai.NewMatrix(n, teKVDim)
	v := tensai.NewMatrix(n, teKVDim)
	attn := tensai.NewMatrix(n, teDim)
	gate := tensai.NewMatrix(n, teMLP)
	up := tensai.NewMatrix(n, teMLP)

	for _, l := range t.layers {
		copy(norm.Data, x.Data)
		rmsNormRows(norm, l.inNorm)
		for _, p := range []struct {
			dst *tensai.Matrix
			w   *linear
		}{{q, l.q}, {k, l.k}, {v, l.v}} {
			if err := p.w.apply(p.dst, norm); err != nil {
				return nil, err
			}
		}
		rmsNormPerHead(q, l.qNorm, teHeads)
		rmsNormPerHead(k, l.kNorm, teKVHeads)
		teRopeWithTable(q, teHeads, cos, sin)
		teRopeWithTable(k, teKVHeads, cos, sin)
		teAttention(attn, q, k, v)
		if err := l.o.apply(norm, attn); err != nil {
			return nil, err
		}
		kernels.AddSlice(x.Data, norm.Data)

		copy(norm.Data, x.Data)
		rmsNormRows(norm, l.postNorm)
		if err := l.gate.apply(gate, norm); err != nil {
			return nil, err
		}
		if err := l.up.apply(up, norm); err != nil {
			return nil, err
		}
		kernels.SiluMul(gate.Data, up.Data)
		if err := l.down.apply(norm, gate); err != nil {
			return nil, err
		}
		kernels.AddSlice(x.Data, norm.Data)
	}
	return x, nil
}

// The template the checkpoint was trained with. It is fed to the
// tokenizer as a plain string rather than through a chat template,
// because the two tokenize differently and this is the one the weights
// expect.
const (
	promptSystem   = "Comprehend and analyze the provided prompt."
	promptPreamble = "<|im_start|>system\n" + promptSystem + "<|im_end|>\n"
	promptTemplate = promptPreamble + "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"

	// PromptDrop is how many tokens of the preamble the transformer
	// never sees: what it reads starts at the user's own words.
	PromptDrop = 14
)

// tokenizerLike is the part of the tokenizer package this needs.
type tokenizerLike interface{ Encode(string) []int }

// PromptTokens turns a prompt into the token sequence the encoder runs
// over, preamble and all.
func PromptTokens(tk tokenizerLike, prompt string) []int {
	if prompt == "" {
		// Qwen has no beginning-of-sequence token, so an empty prompt
		// would leave the encoder with nothing to read.
		prompt = " "
	}
	return tk.Encode(fmt.Sprintf(promptTemplate, prompt))
}

// EncodePrompt turns a prompt into the hidden states the denoising
// transformer reads.
func EncodePrompt(dir ModelDir, prompt string, bits int) (*tensai.Matrix, error) {
	out, err := EncodePrompts(dir, []string{prompt}, bits)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// EncodePrompts encodes several prompts in one pass over the encoder,
// which is loaded and released here because it is another seven
// gigabytes and the denoising transformer wants them. Guidance needs two
// prompts and there is no reason to pay for the encoder twice.
func EncodePrompts(dir ModelDir, prompts []string, bits int) ([]*tensai.Matrix, error) {
	tk, err := tokenizer.Load(dir.Tokenizer())
	if err != nil {
		return nil, err
	}
	ids := make([][]int, len(prompts))
	for i, p := range prompts {
		ids[i] = PromptTokens(tk, p)
		if len(ids[i]) <= PromptDrop {
			return nil, fmt.Errorf("qwenimage: prompt %d tokenized to %d tokens, all preamble", i, len(ids[i]))
		}
	}
	e, err := LoadTextEncoder(dir.TextEncoder(), bits)
	if err != nil {
		return nil, err
	}
	out := make([]*tensai.Matrix, len(ids))
	for i, id := range ids {
		hidden, err := e.Encode(id)
		if err != nil {
			e.Close()
			return nil, err
		}
		// What the transformer reads starts at the user's own words.
		kept := len(id) - PromptDrop
		out[i] = tensai.NewMatrix(kept, teDim)
		copy(out[i].Data, hidden.Data[PromptDrop*teDim:])
	}
	e.Close()
	// Hand the encoder's seven gigabytes back to the system rather than
	// keeping them on Go's free list: the transformer wants them, and a
	// collection alone would leave them counted against us.
	debug.FreeOSMemory()
	return out, nil
}

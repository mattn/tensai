package qwenimage

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/internal/kernels"
)

// The transformer predicts, for a noisy latent and a timestep, the
// velocity that carries it towards the image. Around the 32 blocks sit
// the pieces that build their inputs: the latent and the prompt each get
// projected to the model's width and concatenated, and the timestep
// becomes the modulation every block reads.
//
// Under causal_condition the timestep enters twice — once as itself and
// once as zero. The target image's tokens modulate from the real one and
// everything before them from zero, which is what keeps the prompt's
// contribution identical at every step of a denoising loop.

const (
	ditLatent   = 64  // channels of a latent token
	ditTimeFreq = 256 // width of the sinusoidal timestep
)

// Transformer is the denoising model.
type Transformer struct {
	imgIn          *linear
	txtNorm        []tensai.Float // zero-centred, so the scale is weight+1
	txtIn, txtOut  *linear
	timeIn, timeUp *linear
	modulation     *linear
	normOut        *linear
	projOut        *linear
	blocks         []*Block
}

// ditLayers is how many blocks the checkpoint has.
const ditLayers = 32

// LoadTransformer reads the denoising transformer. With bits set to 8
// the weights quantize as they arrive, which takes the 14GB checkpoint
// to around 7GB; 0 keeps them as floats, which few machines can hold.
func LoadTransformer(dir string, bits int) (*Transformer, error) {
	return loadTransformer(dir, bits, ditLayers)
}

// loadTransformer can stop short of the checkpoint's blocks. A model
// missing blocks predicts nothing worth looking at, but everything
// around them is the same, which is what lets the wiring be checked
// against a reference on a machine that cannot hold all 32.
func loadTransformer(dir string, bits, layers int) (*Transformer, error) {
	w, err := safetensors.OpenSharded(filepath.Join(dir, "diffusion_pytorch_model.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	defer w.Close()

	m := &Transformer{}
	for _, f := range []struct {
		dst        **linear
		name       string
		rows, cols int
	}{
		{&m.imgIn, "img_in.weight", ditDim, ditLatent},
		{&m.txtIn, "txt_in.in_layer.weight", ditDim, ditDim},
		{&m.txtOut, "txt_in.out_layer.weight", ditDim, ditDim},
		{&m.timeIn, "time_text_embed.timestep_embedder.linear_1.weight", ditDim, ditTimeFreq},
		{&m.timeUp, "time_text_embed.timestep_embedder.linear_2.weight", ditDim, ditDim},
		{&m.modulation, "modulation.1.weight", 4 * ditDim, ditDim},
		{&m.normOut, "norm_out.linear.weight", ditDim, ditDim},
		{&m.projOut, "proj_out.weight", ditLatent, ditDim},
	} {
		// The pieces outside the blocks are a rounding error of the
		// model's size and sit on every token's path, so they stay in
		// float whatever the blocks do.
		if *f.dst, err = loadLinear(w, f.name, f.rows, f.cols, 0); err != nil {
			return nil, err
		}
	}
	if m.txtNorm, err = vector(w, "txt_in.text_norm.weight", ditDim); err != nil {
		return nil, err
	}
	for i := 0; i < layers; i++ {
		b, err := LoadBlock(w, i, bits)
		if err != nil {
			return nil, err
		}
		m.blocks = append(m.blocks, b)
	}
	return m, nil
}

// timestepEmbedding is the sinusoidal encoding the checkpoint expects:
// cosines in the first half of the channels and sines in the second,
// over a timestep the model reads in thousandths.
func timestepEmbedding(t float64) []tensai.Float {
	const half = ditTimeFreq / 2
	out := make([]tensai.Float, ditTimeFreq)
	for i := 0; i < half; i++ {
		a := 1000 * t * math.Exp(-math.Log(10000)*float64(i)/half)
		out[i] = tensai.Float(math.Cos(a))
		out[half+i] = tensai.Float(math.Sin(a))
	}
	return out
}

func silu(x *tensai.Matrix) *tensai.Matrix {
	out := &tensai.Matrix{Rows: x.Rows, Cols: x.Cols, Data: append([]tensai.Float(nil), x.Data...)}
	kernels.Silu(out.Data)
	return out
}

// geluTanh is the tanh approximation of GELU, the one the text
// projection was trained with.
func geluTanh(x *tensai.Matrix) {
	for i, v := range x.Data {
		f := float64(v)
		x.Data[i] = tensai.Float(0.5 * f * (1 + math.Tanh(0.7978845608028654*(f+0.044715*f*f*f))))
	}
}

// rmsNormZeroCentred normalizes rows by their root-mean-square and
// scales by weight+1, the form the checkpoint stores.
func rmsNormZeroCentred(x *tensai.Matrix, w []tensai.Float) {
	for r := 0; r < x.Rows; r++ {
		row := x.Data[r*x.Cols : (r+1)*x.Cols]
		var sq float64
		for _, v := range row {
			sq += float64(v) * float64(v)
		}
		inv := tensai.Float(1 / math.Sqrt(sq/float64(x.Cols)+ditEps))
		for i, v := range row {
			row[i] = v * inv * (1 + w[i])
		}
	}
}

// conditioning is what the timestep becomes: the modulation every block
// reads and the scale the final norm takes, one row per timestep the
// sequence modulates from.
func (m *Transformer) conditioning(t float64) (*Modulation, *tensai.Matrix, error) {
	// Row 0 is the sampled timestep, row 1 is zero.
	freq := tensai.NewMatrix(2, ditTimeFreq)
	copy(freq.Data, timestepEmbedding(t))
	copy(freq.Data[ditTimeFreq:], timestepEmbedding(0))

	hidden := tensai.NewMatrix(2, ditDim)
	if err := m.timeIn.apply(hidden, freq); err != nil {
		return nil, nil, err
	}
	kernels.Silu(hidden.Data)
	temb := tensai.NewMatrix(2, ditDim)
	if err := m.timeUp.apply(temb, hidden); err != nil {
		return nil, nil, err
	}

	act := silu(temb)
	mod := tensai.NewMatrix(2, 4*ditDim)
	if err := m.modulation.apply(mod, act); err != nil {
		return nil, nil, err
	}
	scale := tensai.NewMatrix(2, ditDim)
	if err := m.normOut.apply(scale, act); err != nil {
		return nil, nil, err
	}
	out := &Modulation{}
	for row := 0; row < 2; row++ {
		part := func(i int) []tensai.Float { return mod.Data[row*4*ditDim+i*ditDim:][:ditDim] }
		out.Scale1 = append(out.Scale1, part(0))
		out.Gate1 = append(out.Gate1, part(1))
		out.Scale2 = append(out.Scale2, part(2))
		out.Gate2 = append(out.Gate2, part(3))
	}
	return out, scale, nil
}

// Velocity predicts, for a noisy latent and a timestep in [0,1], the
// direction the latent should move. latents is (height*width, 64) and
// text the prompt's hidden states, (tokens, 4096).
func (m *Transformer) Velocity(latents, text *tensai.Matrix, t float64, l *Layout, s *Scratch) (*tensai.Matrix, error) {
	if latents.Rows != l.Height*l.Width || latents.Cols != ditLatent {
		return nil, fmt.Errorf("qwenimage: latents are %dx%d, want %dx%d",
			latents.Rows, latents.Cols, l.Height*l.Width, ditLatent)
	}
	if text.Rows != l.TextLen || text.Cols != ditDim {
		return nil, fmt.Errorf("qwenimage: prompt is %dx%d, want %dx%d", text.Rows, text.Cols, l.TextLen, ditDim)
	}
	mod, outScale, err := m.conditioning(t)
	if err != nil {
		return nil, err
	}

	// The joint sequence: the projected prompt, then the projected
	// latents.
	x := tensai.NewMatrix(l.Tokens(), ditDim)
	txt := &tensai.Matrix{Rows: l.TextLen, Cols: ditDim, Data: append([]tensai.Float(nil), text.Data...)}
	rmsNormZeroCentred(txt, m.txtNorm)
	head := &tensai.Matrix{Rows: l.TextLen, Cols: ditDim, Data: x.Data[:l.TextLen*ditDim]}
	if err := m.txtIn.apply(head, txt); err != nil {
		return nil, err
	}
	geluTanh(head)
	copy(txt.Data, head.Data)
	if err := m.txtOut.apply(head, txt); err != nil {
		return nil, err
	}
	img := &tensai.Matrix{Rows: latents.Rows, Cols: ditDim, Data: x.Data[l.TextLen*ditDim:]}
	if err := m.imgIn.apply(img, latents); err != nil {
		return nil, err
	}

	rope := l.Rope()
	for _, b := range m.blocks {
		if err := b.Forward(x, mod, l, rope, s); err != nil {
			return nil, err
		}
	}

	// The final norm scales rather than shifts, and only the target
	// image's rows are worth projecting.
	norm := tensai.NewMatrix(x.Rows, ditDim)
	layerNorm(norm, x)
	modulate(norm, [][]tensai.Float{outScale.Data[:ditDim], outScale.Data[ditDim:]}, l.Row)
	tail := &tensai.Matrix{Rows: latents.Rows, Cols: ditDim, Data: norm.Data[l.TextLen*ditDim:]}
	out := tensai.NewMatrix(latents.Rows, ditLatent)
	if err := m.projOut.apply(out, tail); err != nil {
		return nil, err
	}
	return out, nil
}

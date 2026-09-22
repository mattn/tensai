package qwenimage

import (
	"fmt"
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/internal/workpool"
	"github.com/mattn/tensai/quant"
)

// The denoising transformer is single-stream: text and image tokens run
// down one sequence through 32 identical blocks, each an attention and a
// SwiGLU feed-forward around parameter-free layer norms. What the norms
// lack in weights the modulation supplies — a scale and a gate per half
// of the block, computed once from the timestep for the whole model
// rather than per block.
//
// Two rows of that modulation exist at a time. The target image's tokens
// read the sampled timestep's row; text and condition-image tokens read
// a row computed at t=0, which is what makes their keys and values the
// same at every denoising step and so worth caching.

const (
	ditHeads   = 32
	ditHeadDim = 128
	ditDim     = ditHeads * ditHeadDim // 4096
	ditMLP     = 3 * ditDim            // 12288
	ditEps     = 1e-6
	ropePairs  = ditHeadDim / 2 // 64 complex pairs per head
)

// Modulation is the scale and gate each block half applies, one entry
// per distinct timestep the sequence modulates from. Which of them a
// token reads is the layout's business.
type Modulation struct {
	Scale1, Gate1 [][]tensai.Float // per row, ditDim wide
	Scale2, Gate2 [][]tensai.Float
}

// Rope carries the rotation each token's head dimensions take, already
// split into the cosine and sine of the three axes' angles.
type Rope struct {
	Cos, Sin *tensai.Matrix // (tokens, ropePairs)
}

// linear is one of a block's weight matrices. The checkpoint is 14GB in
// the form it ships, so a machine that cannot hold that keeps the
// weights quantized instead and the float form stays nil.
type linear struct {
	f  *tensai.Matrix  // (out, in), as the checkpoint stores it
	q  *quant.QMatrix  // (in, out), the layout the int8 kernels want
	q4 *quant.Q4Matrix // the same at four bits
}

func (l *linear) apply(out, x *tensai.Matrix) error {
	switch {
	case l.q != nil:
		return l.q.MatMul(x, out)
	case l.q4 != nil:
		return l.q4.MatMul(x, out)
	}
	return tensai.DotTBInto(out, x, l.f)
}

// Block is one of the transformer's 32 layers.
type Block struct {
	toQ, toK, toV, toOut *linear
	normQ, normK         []tensai.Float
	mlpProj, mlpGate     *linear
	mlpOut               *linear
	// dev holds the feed-forward's weights when they live on a device,
	// and g is what to reach it through.
	dev *deviceWeights
	g   *gpu.Device
}

// devOf returns the device this block's weights were uploaded to.
func (b *Block) devOf() *gpu.Device { return b.g }

// weights is what a Block loads from: either a single file or the
// checkpoint's two shards.
type weights interface {
	Tensor(string) (*tensai.Tensor, error)
}

func matrix(w weights, name string, rows, cols int) (*tensai.Matrix, error) {
	t, err := w.Tensor(name)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 2 || t.Shape[0] != rows || t.Shape[1] != cols {
		return nil, fmt.Errorf("qwenimage: %s: shape %v, want [%d %d]", name, t.Shape, rows, cols)
	}
	return &tensai.Matrix{Rows: rows, Cols: cols, Data: t.Data}, nil
}

// LoadBlock reads one transformer block out of a checkpoint. With bits
// set to 8 every weight is quantized as it arrives, which is what makes
// the model fit where its float form would not; 0 keeps the floats.
func LoadBlock(w weights, i, bits int) (*Block, error) {
	p := fmt.Sprintf("transformer_blocks.%d.", i)
	b := &Block{}
	var err error
	for _, f := range []struct {
		dst        **linear
		name       string
		rows, cols int
	}{
		{&b.toQ, p + "attn.to_q.weight", ditDim, ditDim},
		{&b.toK, p + "attn.to_k.weight", ditDim, ditDim},
		{&b.toV, p + "attn.to_v.weight", ditDim, ditDim},
		{&b.toOut, p + "attn.to_out.0.weight", ditDim, ditDim},
		{&b.mlpProj, p + "img_mlp.proj.weight", ditMLP, ditDim},
		{&b.mlpGate, p + "img_mlp.gate_layer.weight", ditMLP, ditDim},
		{&b.mlpOut, p + "img_mlp.out.weight", ditDim, ditMLP},
	} {
		if *f.dst, err = loadLinear(w, f.name, f.rows, f.cols, bits); err != nil {
			return nil, err
		}
	}
	if b.normQ, err = vector(w, p+"attn.norm_q.weight", ditHeadDim); err != nil {
		return nil, err
	}
	if b.normK, err = vector(w, p+"attn.norm_k.weight", ditHeadDim); err != nil {
		return nil, err
	}
	return b, nil
}

// loadLinear reads one weight matrix, quantizing it on the way in when
// asked. The float form is dropped as soon as the quantized one exists,
// so loading a 7B model never needs its float32 size.
func loadLinear(w weights, name string, rows, cols, bits int) (*linear, error) {
	m, err := matrix(w, name, rows, cols)
	if err != nil {
		return nil, err
	}
	if bits != 8 && bits != 4 {
		return &linear{f: m}, nil
	}
	// The kernels contract over the stored matrix's rows, so the
	// checkpoint's (out, in) has to change hands before it quantizes;
	// the scale then lands per output feature, which is the axis whose
	// weights share a range.
	t := tensai.NewMatrix(cols, rows)
	for o := 0; o < rows; o++ {
		for i := 0; i < cols; i++ {
			t.Data[i*rows+o] = m.Data[o*cols+i]
		}
	}
	if bits == 4 {
		q, err := quant.Quantize4(t)
		if err != nil {
			return nil, err
		}
		return &linear{q4: q}, nil
	}
	return &linear{q: quant.Quantize(t)}, nil
}

func vector(w weights, name string, n int) ([]tensai.Float, error) {
	t, err := w.Tensor(name)
	if err != nil {
		return nil, err
	}
	if len(t.Data) != n {
		return nil, fmt.Errorf("qwenimage: %s: %d values, want %d", name, len(t.Data), n)
	}
	return t.Data, nil
}

// layerNorm normalizes each row to zero mean and unit variance, with no
// weights of its own: the block's scale arrives through the modulation.
func layerNorm(dst, src *tensai.Matrix) {
	n := src.Cols
	workpool.Run(src.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := src.Data[r*n : (r+1)*n]
			var mean, sq float64
			for _, v := range row {
				mean += float64(v)
			}
			mean /= float64(n)
			for _, v := range row {
				d := float64(v) - mean
				sq += d * d
			}
			inv := 1 / math.Sqrt(sq/float64(n)+ditEps)
			out := dst.Data[r*n : (r+1)*n]
			for i, v := range row {
				out[i] = tensai.Float((float64(v) - mean) * inv)
			}
		}
	})
}

// modulate applies a block half's scale, which the checkpoint stores
// centred on zero so that an untrained scale leaves the row alone.
func modulate(x *tensai.Matrix, scale [][]tensai.Float, row []int) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			s := scale[row[r]]
			out := x.Data[r*x.Cols : (r+1)*x.Cols]
			for i, v := range out {
				out[i] = v * (1 + s[i])
			}
		}
	})
}

// rmsNormHeads normalizes each head's slice of every token in place.
func rmsNormHeads(x *tensai.Matrix, w []tensai.Float) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for h := 0; h < ditHeads; h++ {
				head := x.Data[r*x.Cols+h*ditHeadDim:][:ditHeadDim]
				var sq float64
				for _, v := range head {
					sq += float64(v) * float64(v)
				}
				inv := tensai.Float(1 / math.Sqrt(sq/ditHeadDim+ditEps))
				for i, v := range head {
					head[i] = v * inv * w[i]
				}
			}
		}
	})
}

// applyRope rotates each head's dimensions in pairs. The three position
// axes contribute 8, 28 and 28 of the 64 pairs, which the caller has
// already folded into one cosine and one sine per pair.
func applyRope(x *tensai.Matrix, rope *Rope) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			cos := rope.Cos.Data[r*ropePairs:][:ropePairs]
			sin := rope.Sin.Data[r*ropePairs:][:ropePairs]
			for h := 0; h < ditHeads; h++ {
				head := x.Data[r*x.Cols+h*ditHeadDim:][:ditHeadDim]
				for j := 0; j < ropePairs; j++ {
					re, im := head[2*j], head[2*j+1]
					head[2*j] = re*cos[j] - im*sin[j]
					head[2*j+1] = re*sin[j] + im*cos[j]
				}
			}
		}
	})
}

// attention runs every head over the sequence and writes the
// concatenated heads into out. A query reads the keys its limit allows,
// which is what makes the prompt causal and the image bidirectional.
//
// A head's scores are a product, not a scan: taking them one query at a
// time spends its whole life in 128-element dot products, which at an
// image's sequence length left attention half the step while carrying
// two per cent of its arithmetic. Gathering a head into its own pair of
// matrices and multiplying costs two copies and buys the same kernels
// the projections run on.
func attention(out, q, k, v *tensai.Matrix, keyLimit []int, s *Scratch) error {

	scale := tensai.Float(1 / math.Sqrt(ditHeadDim))
	for h := 0; h < ditHeads; h++ {
		off := h * ditHeadDim
		gather(s.qh, q, off)
		gather(s.kh, k, off)
		gather(s.vh, v, off)
		if err := tensai.DotTBInto(s.scores, s.qh, s.kh); err != nil {
			return err
		}
		softmaxRows(s.scores, scale, keyLimit)
		if err := tensai.DotInto(s.oh, s.scores, s.vh); err != nil {
			return err
		}
		scatter(out, s.oh, off)
	}
	return nil
}

// gather copies one head's slice of every token into its own matrix.
func gather(dst, src *tensai.Matrix, off int) {
	workpool.Run(src.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			copy(dst.Data[r*ditHeadDim:(r+1)*ditHeadDim], src.Data[r*src.Cols+off:])
		}
	})
}

// scatter is the inverse, writing a head's output back where it belongs.
func scatter(dst, src *tensai.Matrix, off int) {
	workpool.Run(src.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			copy(dst.Data[r*dst.Cols+off:][:ditHeadDim], src.Data[r*ditHeadDim:])
		}
	})
}

// softmaxRows scales a head's scores and normalizes each row over the
// keys its limit allows, leaving the rest at zero so the value product
// can read the whole row.
func softmaxRows(x *tensai.Matrix, scale tensai.Float, keyLimit []int) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x.Data[r*x.Cols : (r+1)*x.Cols]
			lim := keyLimit[r]
			maxs := tensai.Float(math.Inf(-1))
			for i, s := range row[:lim] {
				s *= scale
				row[i] = s
				if s > maxs {
					maxs = s
				}
			}
			var sum tensai.Float
			for i, s := range row[:lim] {
				e := tensai.Float(math.Exp(float64(s - maxs)))
				row[i] = e
				sum += e
			}
			inv := 1 / sum
			for i := range row[:lim] {
				row[i] *= inv
			}
			clear(row[lim:])
		}
	})
}

// Forward runs one block over the joint sequence, in place.
func (b *Block) Forward(x *tensai.Matrix, m *Modulation, l *Layout, rope *Rope, s *Scratch) error {
	if x.Cols != ditDim {
		return fmt.Errorf("qwenimage: block wants %d columns, got %d", ditDim, x.Cols)
	}
	s.reset(x.Rows)

	layerNorm(s.norm, x)
	modulate(s.norm, m.Scale1, l.Row)
	for _, p := range []struct {
		dst *tensai.Matrix
		w   *linear
	}{{s.q, b.toQ}, {s.k, b.toK}, {s.v, b.toV}} {
		if err := p.w.apply(p.dst, s.norm); err != nil {
			return err
		}
	}
	rmsNormHeads(s.q, b.normQ)
	rmsNormHeads(s.k, b.normK)
	if rope != nil {
		applyRope(s.q, rope)
		applyRope(s.k, rope)
	}
	if err := attention(s.attn, s.q, s.k, s.v, l.KeyLimit, s); err != nil {
		return err
	}
	if err := b.toOut.apply(s.norm, s.attn); err != nil {
		return err
	}
	addGated(x, s.norm, m.Gate1, l.Row)

	layerNorm(s.norm, x)
	modulate(s.norm, m.Scale2, l.Row)
	if b.dev != nil {
		if err := b.mlpOnDevice(s.attn, s.norm); err != nil {
			return err
		}
		addGated(x, s.attn, m.Gate2, l.Row)
		return nil
	}
	if err := b.mlpGate.apply(s.gate, s.norm); err != nil {
		return err
	}
	if err := b.mlpProj.apply(s.up, s.norm); err != nil {
		return err
	}
	kernels.SiluMul(s.gate.Data, s.up.Data)
	if err := b.mlpOut.apply(s.norm, s.gate); err != nil {
		return err
	}
	addGated(x, s.norm, m.Gate2, l.Row)
	return nil
}

// addGated accumulates a block half's output, squashed by its gate. The
// tanh keeps an untrained gate at zero and bounds what one half can add.
func addGated(x, y *tensai.Matrix, gate [][]tensai.Float, row []int) {
	// The gate is per row of the modulation, not per token, so the
	// squash runs once for each of its rows rather than once a token.
	squashed := make([][]tensai.Float, len(gate))
	for i, g := range gate {
		s := make([]tensai.Float, len(g))
		for j, v := range g {
			s[j] = tensai.Float(math.Tanh(float64(v)))
		}
		squashed[i] = s
	}
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			g := squashed[row[r]]
			dst := x.Data[r*x.Cols : (r+1)*x.Cols]
			src := y.Data[r*y.Cols : (r+1)*y.Cols]
			for i, v := range src {
				dst[i] += g[i] * v
			}
		}
	})
}

// Scratch holds a block's intermediates, reused across the 32 blocks and
// the whole denoising loop.
type Scratch struct {
	norm, q, k, v, attn *tensai.Matrix
	gate, up            *tensai.Matrix
	// One head at a time, gathered out of the packed projections, plus
	// its square of scores.
	qh, kh, vh, oh *tensai.Matrix
	scores         *tensai.Matrix
}

// NewScratch sizes the buffers for a sequence of at most n tokens.
func NewScratch(n int) *Scratch {
	s := &Scratch{}
	for _, m := range []**tensai.Matrix{&s.norm, &s.q, &s.k, &s.v, &s.attn} {
		*m = tensai.NewMatrix(n, ditDim)
	}
	s.gate = tensai.NewMatrix(n, ditMLP)
	s.up = tensai.NewMatrix(n, ditMLP)
	for _, m := range []**tensai.Matrix{&s.qh, &s.kh, &s.vh, &s.oh} {
		*m = tensai.NewMatrix(n, ditHeadDim)
	}
	s.scores = tensai.NewMatrix(n, n)
	return s
}

func (s *Scratch) reset(n int) {
	for _, m := range []*tensai.Matrix{s.norm, s.q, s.k, s.v, s.attn, s.gate, s.up, s.qh, s.kh, s.vh, s.oh} {
		cols := m.Cols
		m.Rows = n
		m.Data = m.Data[:n*cols]
	}
	s.scores.Rows, s.scores.Cols = n, n
	s.scores.Data = s.scores.Data[:n*n]
}

// OpenTransformer opens the checkpoint's sharded transformer weights.
func OpenTransformer(indexPath string) (*safetensors.Shards, error) {
	return safetensors.OpenSharded(indexPath)
}

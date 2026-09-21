package qwenimage

import (
	"fmt"
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/internal/kernels"
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

// Block is one of the transformer's 32 layers.
type Block struct {
	toQ, toK, toV, toOut *tensai.Matrix
	normQ, normK         []tensai.Float
	mlpProj, mlpGate     *tensai.Matrix
	mlpOut               *tensai.Matrix
}

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

// LoadBlock reads one transformer block out of a checkpoint.
func LoadBlock(w weights, i int) (*Block, error) {
	p := fmt.Sprintf("transformer_blocks.%d.", i)
	b := &Block{}
	var err error
	for _, f := range []struct {
		dst        **tensai.Matrix
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
		if *f.dst, err = matrix(w, f.name, f.rows, f.cols); err != nil {
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
	for r := 0; r < src.Rows; r++ {
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
}

// modulate applies a block half's scale, which the checkpoint stores
// centred on zero so that an untrained scale leaves the row alone.
func modulate(x *tensai.Matrix, scale [][]tensai.Float, row []int) {
	for r := 0; r < x.Rows; r++ {
		s := scale[row[r]]
		out := x.Data[r*x.Cols : (r+1)*x.Cols]
		for i, v := range out {
			out[i] = v * (1 + s[i])
		}
	}
}

// rmsNormHeads normalizes each head's slice of every token in place.
func rmsNormHeads(x *tensai.Matrix, w []tensai.Float) {
	for r := 0; r < x.Rows; r++ {
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
}

// applyRope rotates each head's dimensions in pairs. The three position
// axes contribute 8, 28 and 28 of the 64 pairs, which the caller has
// already folded into one cosine and one sine per pair.
func applyRope(x *tensai.Matrix, rope *Rope) {
	for r := 0; r < x.Rows; r++ {
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
}

// attention runs every head over the sequence and writes the
// concatenated heads into out. A query reads the keys its limit allows,
// which is what makes the prompt causal and the image bidirectional.
func attention(out, q, k, v *tensai.Matrix, keyLimit []int) {
	n := q.Rows
	scale := tensai.Float(1 / math.Sqrt(ditHeadDim))
	scores := make([]tensai.Float, n)
	for h := 0; h < ditHeads; h++ {
		off := h * ditHeadDim
		for i := 0; i < n; i++ {
			lim := keyLimit[i]
			qi := q.Data[i*q.Cols+off:][:ditHeadDim]
			maxs := tensai.Float(math.Inf(-1))
			for j := 0; j < lim; j++ {
				s := tensai.DotVec(qi, k.Data[j*k.Cols+off:][:ditHeadDim]) * scale
				scores[j] = s
				if s > maxs {
					maxs = s
				}
			}
			var sum tensai.Float
			for j, s := range scores[:lim] {
				e := tensai.Float(math.Exp(float64(s - maxs)))
				scores[j] = e
				sum += e
			}
			dst := out.Data[i*out.Cols+off:][:ditHeadDim]
			clear(dst)
			for j, e := range scores[:lim] {
				kernels.Axpy(e/sum, v.Data[j*v.Cols+off:][:ditHeadDim], dst)
			}
		}
	}
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
		w   *tensai.Matrix
	}{{s.q, b.toQ}, {s.k, b.toK}, {s.v, b.toV}} {
		if err := tensai.DotTBInto(p.dst, s.norm, p.w); err != nil {
			return err
		}
	}
	rmsNormHeads(s.q, b.normQ)
	rmsNormHeads(s.k, b.normK)
	if rope != nil {
		applyRope(s.q, rope)
		applyRope(s.k, rope)
	}
	attention(s.attn, s.q, s.k, s.v, l.KeyLimit)
	if err := tensai.DotTBInto(s.norm, s.attn, b.toOut); err != nil {
		return err
	}
	addGated(x, s.norm, m.Gate1, l.Row)

	layerNorm(s.norm, x)
	modulate(s.norm, m.Scale2, l.Row)
	if err := tensai.DotTBInto(s.gate, s.norm, b.mlpGate); err != nil {
		return err
	}
	if err := tensai.DotTBInto(s.up, s.norm, b.mlpProj); err != nil {
		return err
	}
	kernels.SiluMul(s.gate.Data, s.up.Data)
	if err := tensai.DotTBInto(s.norm, s.gate, b.mlpOut); err != nil {
		return err
	}
	addGated(x, s.norm, m.Gate2, l.Row)
	return nil
}

// addGated accumulates a block half's output, squashed by its gate. The
// tanh keeps an untrained gate at zero and bounds what one half can add.
func addGated(x, y *tensai.Matrix, gate [][]tensai.Float, row []int) {
	for r := 0; r < x.Rows; r++ {
		g := gate[row[r]]
		dst := x.Data[r*x.Cols : (r+1)*x.Cols]
		src := y.Data[r*y.Cols : (r+1)*y.Cols]
		for i, v := range src {
			dst[i] += tensai.Float(math.Tanh(float64(g[i]))) * v
		}
	}
}

// Scratch holds a block's intermediates, reused across the 32 blocks and
// the whole denoising loop.
type Scratch struct {
	norm, q, k, v, attn *tensai.Matrix
	gate, up            *tensai.Matrix
}

// NewScratch sizes the buffers for a sequence of at most n tokens.
func NewScratch(n int) *Scratch {
	s := &Scratch{}
	for _, m := range []**tensai.Matrix{&s.norm, &s.q, &s.k, &s.v, &s.attn} {
		*m = tensai.NewMatrix(n, ditDim)
	}
	s.gate = tensai.NewMatrix(n, ditMLP)
	s.up = tensai.NewMatrix(n, ditMLP)
	return s
}

func (s *Scratch) reset(n int) {
	for _, m := range []*tensai.Matrix{s.norm, s.q, s.k, s.v, s.attn, s.gate, s.up} {
		cols := m.Cols
		m.Rows = n
		m.Data = m.Data[:n*cols]
	}
}

// OpenTransformer opens the checkpoint's sharded transformer weights.
func OpenTransformer(indexPath string) (*safetensors.Shards, error) {
	return safetensors.OpenSharded(indexPath)
}

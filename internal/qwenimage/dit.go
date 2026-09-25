package qwenimage

import (
	"fmt"
	"math"
	"runtime"
	"sync"

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
	// rot is the width of the Hadamard groups the quantized weights
	// were rotated in, 0 for none. The input is rotated the same way
	// before the product: H is orthogonal, so x W^T = (x H)(W H)^T, and
	// the rotation spreads the outlier columns of both across their
	// group, which is where eight bits lose the most.
	rot  int
	lora *lowRank
}

// inputs is how many input features the layer reads.
func (l *linear) inputs() int {
	switch {
	case l.q != nil:
		return l.q.Rows
	case l.q4 != nil:
		return l.q4.Rows
	}
	return l.f.Cols
}

// convRotGroup is the rotation width: ComfyUI's, so its int8 weights are
// already in the form the kernels want.
const convRotGroup = 256

func (l *linear) apply(out, x *tensai.Matrix) error {
	if l.rot > 0 {
		r := rotatedCopy(x, l.rot)
		defer rotPool.Put(r)
		x = r
	}
	var err error
	switch {
	case l.q != nil:
		err = l.q.MatMul(x, out)
	case l.q4 != nil:
		err = l.q4.MatMul(x, out)
	default:
		err = tensai.DotTBInto(out, x, l.f)
	}
	if err == nil && l.lora != nil {
		err = l.lora.add(out, x)
	}
	return err
}

// Block is one of the transformer's 32 layers.
type Block struct {
	toQ, toK, toV, toOut *linear
	normQ, normK         []tensai.Float
	mlpProj, mlpGate     *linear
	mlpOut               *linear
	// dev holds the feed-forward's weights when they live on a device,
	// and g is what to reach it through.
	dev               *deviceWeights
	g                 *gpu.Device
	streamProjections bool
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
		if *f.dst, err = loadLinear(w, f.name, f.rows, f.cols, bits, convRotGroup); err != nil {
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
func loadLinear(w weights, name string, rows, cols, bits, rot int) (*linear, error) {
	m, err := matrix(w, name, rows, cols)
	if err != nil {
		return nil, err
	}
	if bits != 8 && bits != 4 {
		return &linear{f: m}, nil
	}
	if rot > 0 && (cols%rot != 0 || noRotate) {
		rot = 0
	}
	if rot > 0 {
		rotateRows(m, rot)
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
		return &linear{q4: q, rot: rot}, nil
	}
	return &linear{q: quant.Quantize(t), rot: rot}, nil
}

// noRotate turns the rotation off, for measuring what it buys.
var noRotate bool

// rotateRows rotates each row of m in groups of width g, in place.
func rotateRows(m *tensai.Matrix, g int) {
	workpool.Run(m.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := m.Data[r*m.Cols : (r+1)*m.Cols]
			for c := 0; c+g <= len(row); c += g {
				hadamard(row[c : c+g])
			}
		}
	})
}

// rotPool recycles the rotated copies of activations.
var rotPool sync.Pool

// rotatedCopy returns x with every row rotated in groups of g, in a
// matrix from rotPool that the caller puts back.
func rotatedCopy(x *tensai.Matrix, g int) *tensai.Matrix {
	r, _ := rotPool.Get().(*tensai.Matrix)
	if r == nil {
		r = &tensai.Matrix{}
	}
	n := x.Rows * x.Cols
	if cap(r.Data) < n {
		r.Data = make([]tensai.Float, n)
	}
	r.Rows, r.Cols, r.Data = x.Rows, x.Cols, r.Data[:n]
	copy(r.Data, x.Data)
	rotateRows(r, g)
	return r
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
	workpool.Run(src.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			kernels.LayerNorm64(dst.Data[r*src.Cols:(r+1)*src.Cols], src.Data[r*src.Cols:(r+1)*src.Cols], ditEps)
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
			kernels.ScaleWeights(out, s, 1, 1)
		}
	})
}

// rmsNormHeads normalizes each head's slice of every token in place.
func rmsNormHeads(x *tensai.Matrix, w []tensai.Float) {
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for h := 0; h < ditHeads; h++ {
				head := x.Data[r*x.Cols+h*ditHeadDim:][:ditHeadDim]
				sq := kernels.SquaredDeviations64(head, 0)
				inv := tensai.Float(1 / math.Sqrt(sq/ditHeadDim+ditEps))
				kernels.ScaleWeights(head, w, inv, 0)
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
				kernels.RopePairs(head, cos, sin)
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
// Heads are independent, so a worker takes whole heads rather than a
// slice of every stage of one: a head's gather, both its products and
// the softmax between them stay on the goroutine whose buffers they run
// through, and one barrier stands where three per head used to.
//
// Queries go a tile at a time. A tile's scores are made and spent inside
// it, so the sequence's square is never written out to memory and read
// back twice -- and that square was the one buffer here that grew with
// the square of the image, where a tile's rows do not.
func attention(out, q, k, v *tensai.Matrix, keyLimit []int, s *Scratch) error {
	scale := tensai.Float(1 / math.Sqrt(ditHeadDim))
	errs := make([]error, len(s.heads))
	var wg sync.WaitGroup
	for w := 1; w < len(s.heads); w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			errs[w] = attentionHeads(out, q, k, v, keyLimit, s.heads[w], scale, w, len(s.heads))
		}(w)
	}
	errs[0] = attentionHeads(out, q, k, v, keyLimit, s.heads[0], scale, 0, len(s.heads))
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// attentionHeads runs the heads w, w+workers, ... through one worker's
// buffers, a tile of queries at a time.
func attentionHeads(out, q, k, v *tensai.Matrix, keyLimit []int, b *headBuf, scale tensai.Float, w, workers int) error {
	n := q.Rows
	for h := w; h < ditHeads; h += workers {
		off := h * ditHeadDim
		gatherRows(b.kh, k, off, 0, n)
		gatherRows(b.vh, v, off, 0, n)
		for t := 0; t < n; t += attnTile {
			rows := min(attnTile, n-t)
			b.tile(rows, n)
			gatherRows(b.qt, q, off, t, t+rows)
			if err := tensai.DotTBIntoSerial(b.st, b.qt, b.kh); err != nil {
				return err
			}
			softmaxTile(b.st, scale, keyLimit[t:t+rows])
			if err := tensai.DotIntoSerial(b.ot, b.st, b.vh); err != nil {
				return err
			}
			scatterRows(out, b.ot, off, t)
		}
	}
	return nil
}

// gatherRows copies one head's slice of the rows lo..hi into a matrix of
// its own, which is where the kernels want it.
func gatherRows(dst, src *tensai.Matrix, off, lo, hi int) {
	for r := lo; r < hi; r++ {
		copy(dst.Data[(r-lo)*ditHeadDim:][:ditHeadDim], src.Data[r*src.Cols+off:])
	}
}

// scatterRows is the inverse, writing a tile of one head's output back
// where it belongs among the others.
func scatterRows(dst, src *tensai.Matrix, off, at int) {
	for r := 0; r < src.Rows; r++ {
		copy(dst.Data[(at+r)*dst.Cols+off:][:ditHeadDim], src.Data[r*ditHeadDim:])
	}
}

// softmaxTile scales a tile of a head's scores and normalizes each row
// over the keys its limit allows, leaving the rest at zero so the value
// product can read the whole row.
func softmaxTile(x *tensai.Matrix, scale tensai.Float, keyLimit []int) {
	for r := 0; r < x.Rows; r++ {
		row := x.Data[r*x.Cols : (r+1)*x.Cols]
		lim := keyLimit[r]
		kernels.ScaleSlice(row[:lim], scale)
		kernels.Softmax(row[:lim])
		clear(row[lim:])
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
		w   *linear
	}{{s.q, b.toQ}, {s.k, b.toK}, {s.v, b.toV}} {
		if err := b.project(p.dst, s.norm, p.w); err != nil {
			return err
		}
	}
	rmsNormHeads(s.q, b.normQ)
	rmsNormHeads(s.k, b.normK)
	if rope != nil {
		applyRope(s.q, rope)
		applyRope(s.k, rope)
	}
	if b.dev != nil {
		if err := b.attentionOnDevice(s.attn, s.q, s.k, s.v, l, s); err != nil {
			return err
		}
	} else if err := attention(s.attn, s.q, s.k, s.v, l.KeyLimit, s); err != nil {
		return err
	}
	if err := b.project(s.norm, s.attn, b.toOut); err != nil {
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
		kernels.TanhFwd(s, g)
		squashed[i] = s
	}
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			g := squashed[row[r]]
			dst := x.Data[r*x.Cols : (r+1)*x.Cols]
			src := y.Data[r*y.Cols : (r+1)*y.Cols]
			kernels.MulAddSlice(dst, g, src)
		}
	})
}

// Scratch holds a block's intermediates, reused across the 32 blocks and
// the whole denoising loop.
type Scratch struct {
	norm, q, k, v, attn *tensai.Matrix
	gate, up            *tensai.Matrix
	// One set of head buffers per attention worker.
	heads []*headBuf
}

// headBuf is one worker's room for a head: the keys and values it
// gathers out of the packed projections, and the tile of queries it is
// on with their scores and their outputs.
type headBuf struct {
	kh, vh *tensai.Matrix // the whole sequence, one head wide
	qt, ot *tensai.Matrix // attnTile rows of it
	st     *tensai.Matrix // attnTile queries over every key
}

// attnTile is how many queries one pass of a head takes. Sixty-four rows
// of a 512x512 image's keys is a quarter of a megabyte, which stays in
// cache from the product that writes the scores to the one that spends
// them.
const attnTile = 64

// NewScratch sizes the buffers for a sequence of at most n tokens.
func NewScratch(n int) *Scratch {
	s := &Scratch{}
	for _, m := range []**tensai.Matrix{&s.norm, &s.q, &s.k, &s.v, &s.attn} {
		*m = tensai.NewMatrix(n, ditDim)
	}
	s.gate = tensai.NewMatrix(n, ditMLP)
	s.up = tensai.NewMatrix(n, ditMLP)
	s.heads = make([]*headBuf, min(runtime.GOMAXPROCS(0), ditHeads))
	for i := range s.heads {
		s.heads[i] = &headBuf{
			kh: tensai.NewMatrix(n, ditHeadDim),
			vh: tensai.NewMatrix(n, ditHeadDim),
			qt: tensai.NewMatrix(attnTile, ditHeadDim),
			ot: tensai.NewMatrix(attnTile, ditHeadDim),
			st: tensai.NewMatrix(attnTile, n),
		}
	}
	return s
}

func (s *Scratch) reset(n int) {
	for _, m := range []*tensai.Matrix{s.norm, s.q, s.k, s.v, s.attn, s.gate, s.up} {
		cols := m.Cols
		m.Rows = n
		m.Data = m.Data[:n*cols]
	}
	for _, b := range s.heads {
		for _, m := range []*tensai.Matrix{b.kh, b.vh} {
			m.Rows = n
			m.Data = m.Data[:n*ditHeadDim]
		}
	}
}

// tile points a worker's buffers at rows queries over n keys.
func (b *headBuf) tile(rows, n int) {
	for _, m := range []*tensai.Matrix{b.qt, b.ot} {
		m.Rows = rows
		m.Data = m.Data[:rows*ditHeadDim]
	}
	b.st.Rows, b.st.Cols = rows, n
	b.st.Data = b.st.Data[:rows*n]
}

// OpenTransformer opens the checkpoint's sharded transformer weights.
func OpenTransformer(indexPath string) (*safetensors.Shards, error) {
	return safetensors.OpenSharded(indexPath)
}

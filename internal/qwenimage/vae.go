// Package qwenimage decodes Qwen-Image-2.1 latents into pixels.
//
// The checkpoint's VAE is Wan's causal 3D autoencoder, but Qwen-Image
// feeds it one frame at a time: every temporal convolution sits behind a
// feature cache that is empty on the first (and only) chunk, so it never
// runs, and the temporal half of each shortcut collapses to its last
// slot. What is left is a plain 2D decoder, which is what this package
// implements — five upsampling stages from a 64-channel latent to RGBA
// pixels at sixteen times the latent's width.
package qwenimage

import (
	"fmt"
	"image"
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/internal/workpool"
)

// Feature maps are channels-last: a Matrix of one row per pixel and one
// column per channel. Every operation here wants a pixel's channels
// together — the norm reduces over them, the convolutions contract over
// them, and the attention block reads them as its model dimension — so
// the layout the weights arrive in (channels-first) is converted once,
// at load, rather than transposed between layers.

// conv is a 2D convolution with a square kernel and, for 3x3, the zero
// padding that keeps the resolution. Its weights are stored as the
// matrix an im2col product multiplies: one row per output channel, and
// its columns grouped by kernel position, then input channel.
type conv struct {
	w        *tensai.Matrix
	b        []tensai.Float
	in, out  int
	ksz, pad int
}

// imcolBudget caps one im2col tile. The expanded columns are the
// decoder's largest intermediate by far (nine input channels' worth per
// pixel), so they are built a tile at a time and reused.
const imcolBudget = 64 << 20

func loadConv(f weights, prefix string, ksz int) (*conv, error) {
	wt, err := f.Tensor(prefix + ".weight")
	if err != nil {
		return nil, err
	}
	if len(wt.Shape) != 4 || wt.Shape[2] != ksz || wt.Shape[3] != ksz {
		return nil, fmt.Errorf("qwenimage: %s: want a %dx%d kernel, got shape %v", prefix, ksz, ksz, wt.Shape)
	}
	out, in := wt.Shape[0], wt.Shape[1]
	c := &conv{
		w:   tensai.NewMatrix(out, ksz*ksz*in),
		in:  in,
		out: out,
		ksz: ksz,
		pad: (ksz - 1) / 2,
	}
	// The checkpoint stores (out, in, ky, kx); the product wants a
	// column order of (ky, kx, in) to match how im2col lays a pixel's
	// neighbourhood out.
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			for k := 0; k < ksz*ksz; k++ {
				c.w.Data[o*c.w.Cols+k*in+i] = wt.Data[(o*in+i)*ksz*ksz+k]
			}
		}
	}
	bt, err := f.Tensor(prefix + ".bias")
	if err != nil {
		return nil, err
	}
	c.b = bt.Data
	return c, nil
}

// apply runs the convolution over a (h*w, in) feature map.
func (c *conv) apply(x *tensai.Matrix, h, w int, g *gpu.Device) (*tensai.Matrix, error) {
	if x.Cols != c.in || x.Rows != h*w {
		return nil, fmt.Errorf("qwenimage: conv wants %dx%d, got %dx%d", h*w, c.in, x.Rows, x.Cols)
	}
	if g != nil {
		return c.onGPU(g, x, h, w)
	}
	out := tensai.NewMatrix(h*w, c.out)
	if c.ksz == 1 {
		// A 1x1 kernel is the product itself, no gathering.
		if err := tensai.DotTBInto(out, x, c.w); err != nil {
			return nil, err
		}
		addBias(out, c.b)
		return out, nil
	}
	cols := c.w.Cols
	tile := max(64, imcolBudget/(cols*4))
	buf := tensai.NewMatrix(min(tile, h*w), cols)
	for lo := 0; lo < h*w; lo += tile {
		hi := min(lo+tile, h*w)
		buf.Rows = hi - lo
		buf.Data = buf.Data[:buf.Rows*cols]
		c.imcol(buf, x, h, w, lo, hi)
		chunk := &tensai.Matrix{Rows: hi - lo, Cols: c.out, Data: out.Data[lo*c.out : hi*c.out]}
		if err := tensai.DotTBInto(chunk, buf, c.w); err != nil {
			return nil, err
		}
	}
	addBias(out, c.b)
	return out, nil
}

// imcol fills one tile of the expanded column matrix: row r holds the
// nine neighbours of pixel lo+r, each neighbour contributing its whole
// channel vector, with out-of-image neighbours left zero.
func (c *conv) imcol(buf, x *tensai.Matrix, h, w, lo, hi int) {
	clear(buf.Data)
	in := c.in
	for r := lo; r < hi; r++ {
		y, px := r/w, r%w
		dst := buf.Data[(r-lo)*buf.Cols:]
		for k := 0; k < c.ksz*c.ksz; k++ {
			sy := y + k/c.ksz - c.pad
			sx := px + k%c.ksz - c.pad
			if sy < 0 || sy >= h || sx < 0 || sx >= w {
				continue
			}
			copy(dst[k*in:(k+1)*in], x.Data[(sy*w+sx)*in:])
		}
	}
}

func addBias(x *tensai.Matrix, b []tensai.Float) {
	for r := 0; r < x.Rows; r++ {
		kernels.AddSlice(x.Data[r*x.Cols:(r+1)*x.Cols], b)
	}
}

// rmsNorm is the checkpoint's RMS_norm: an L2 normalize across a pixel's
// channels, scaled back up by the square root of their count, which
// leaves the row divided by its root-mean-square, times gamma. The
// reference guards the norm rather than the mean square, so a pixel that
// is all zeros survives.
func rmsNorm(x *tensai.Matrix, gamma []tensai.Float) {
	scale := math.Sqrt(float64(x.Cols))
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x.Data[r*x.Cols : (r+1)*x.Cols]
			ss := kernels.SquaredDeviations64(row, 0)
			inv := tensai.Float(scale / math.Max(math.Sqrt(ss), 1e-12))
			kernels.ScaleWeights(row, gamma, inv, 0)
		}
	})
}

// resBlock is the decoder's residual unit: two normalized, activated
// convolutions around a shortcut that only needs a 1x1 projection when
// the channel count changes.
type resBlock struct {
	norm1, norm2 []tensai.Float
	conv1, conv2 *conv
	shortcut     *conv // nil when the block keeps its width
}

func loadResBlock(f vaeWeights, prefix string) (*resBlock, error) {
	b := &resBlock{}
	var err error
	if b.norm1, err = gamma(f, prefix+".norm1.gamma"); err != nil {
		return nil, err
	}
	if b.norm2, err = gamma(f, prefix+".norm2.gamma"); err != nil {
		return nil, err
	}
	if b.conv1, err = loadConv(f, prefix+".conv1", 3); err != nil {
		return nil, err
	}
	if b.conv2, err = loadConv(f, prefix+".conv2", 3); err != nil {
		return nil, err
	}
	if _, _, ok := f.Info(prefix + ".conv_shortcut.weight"); ok {
		if b.shortcut, err = loadConv(f, prefix+".conv_shortcut", 1); err != nil {
			return nil, err
		}
	}
	return b, nil
}

func (b *resBlock) apply(x *tensai.Matrix, h, w int, g *gpu.Device) (*tensai.Matrix, error) {
	skip := x
	if b.shortcut != nil {
		var err error
		if skip, err = b.shortcut.apply(x, h, w, g); err != nil {
			return nil, err
		}
	}
	y := clone(x)
	rmsNorm(y, b.norm1)
	kernels.Silu(y.Data)
	y, err := b.conv1.apply(y, h, w, g)
	if err != nil {
		return nil, err
	}
	rmsNorm(y, b.norm2)
	kernels.Silu(y.Data)
	if y, err = b.conv2.apply(y, h, w, g); err != nil {
		return nil, err
	}
	kernels.AddSlice(y.Data, skip.Data)
	return y, nil
}

// attnBlock is one head of self-attention over the latent's pixels, the
// only place in the decoder where a pixel sees any other.
type attnBlock struct {
	norm []tensai.Float
	qkv  *conv
	proj *conv
}

func loadAttn(f weights, prefix string) (*attnBlock, error) {
	a := &attnBlock{}
	var err error
	if a.norm, err = gamma(f, prefix+".norm.gamma"); err != nil {
		return nil, err
	}
	if a.qkv, err = loadConv(f, prefix+".to_qkv", 1); err != nil {
		return nil, err
	}
	if a.proj, err = loadConv(f, prefix+".proj", 1); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *attnBlock) apply(x *tensai.Matrix, h, w int, g *gpu.Device) (*tensai.Matrix, error) {
	n, dim := x.Rows, x.Cols
	y := clone(x)
	rmsNorm(y, a.norm)
	qkv, err := a.qkv.apply(y, h, w, g)
	if err != nil {
		return nil, err
	}
	// The projection's three parts sit side by side in a row, so the
	// query, key and value of a pixel are three slices of its row.
	att := tensai.NewMatrix(n, dim)
	scores := make([]tensai.Float, n)
	scale := tensai.Float(1 / math.Sqrt(float64(dim)))
	for i := 0; i < n; i++ {
		q := qkv.Data[i*3*dim : i*3*dim+dim]
		for j := 0; j < n; j++ {
			k := qkv.Data[j*3*dim+dim : j*3*dim+2*dim]
			s := tensai.DotVec(q, k) * scale
			scores[j] = s
		}
		kernels.Softmax(scores)
		row := att.Data[i*dim : (i+1)*dim]
		for j, e := range scores {
			kernels.Axpy(e, qkv.Data[j*3*dim+2*dim:j*3*dim+3*dim], row)
		}
	}
	if att, err = a.proj.apply(att, h, w, g); err != nil {
		return nil, err
	}
	kernels.AddSlice(att.Data, x.Data)
	return att, nil
}

// upBlock is one stage of the decoder: a run of residual blocks, an
// optional 2x upsample, and — this checkpoint being the residual variant
// — a parameter-free shortcut that reaches the stage's output by
// shuffling the input's channels into the new pixels.
type upBlock struct {
	resnets []*resBlock
	up      *conv // 3x3 after a nearest-neighbour doubling; nil on the last stage
	// dupOut and dupRepeat describe the shuffled shortcut; dupOut is 0
	// when the stage has none. dupT is 2 where the checkpoint would also
	// have doubled time, which for a single frame only decides which
	// half of the channel groups survives.
	dupOut, dupRepeat, dupT int
}

func (u *upBlock) apply(x *tensai.Matrix, h, w int, g *gpu.Device) (*tensai.Matrix, int, int, error) {
	src := x
	var err error
	for _, r := range u.resnets {
		if x, err = r.apply(x, h, w, g); err != nil {
			return nil, 0, 0, err
		}
	}
	oh, ow := h, w
	if u.up != nil {
		x = nearest2x(x, h, w)
		oh, ow = 2*h, 2*w
		if x, err = u.up.apply(x, oh, ow, g); err != nil {
			return nil, 0, 0, err
		}
	}
	if u.dupOut != 0 {
		kernels.AddSlice(x.Data, dupUp(src, h, w, u.dupOut, u.dupRepeat, u.dupT).Data)
	}
	return x, oh, ow, nil
}

// nearest2x doubles both axes by repeating each pixel into a 2x2 block.
func nearest2x(x *tensai.Matrix, h, w int) *tensai.Matrix {
	c := x.Cols
	out := tensai.NewMatrix(4*h*w, c)
	for y := 0; y < h; y++ {
		for px := 0; px < w; px++ {
			row := x.Data[(y*w+px)*c:][:c]
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					copy(out.Data[((2*y+dy)*2*w+2*px+dx)*c:], row)
				}
			}
		}
	}
	return out
}

// dupUp is the residual shortcut across an upsampling stage. It repeats
// each input channel `repeat` times, reads the result as a grid of
// (output channel, time, y, x) groups, and scatters those groups into
// the doubled resolution. With one frame only the last time group
// survives, which is what the reference's first-chunk slice keeps.
func dupUp(x *tensai.Matrix, h, w, outC, repeat, factorT int) *tensai.Matrix {
	out := tensai.NewMatrix(4*h*w, outC)
	ft := factorT - 1
	// Walk pixels before channels so each output cache line is filled
	// once, rather than revisited for every channel across the whole map.
	var channels [4][]int
	for p := range channels {
		channels[p] = make([]int, outC)
		for oc := range channels[p] {
			channels[p][oc] = (oc*factorT*4 + ft*4 + p) / repeat
		}
	}
	workpool.Run(h, 1, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			for px := 0; px < w; px++ {
				src := x.Data[(y*w+px)*x.Cols:][:x.Cols]
				for p, mapping := range channels {
					dst := out.Data[((2*y+p/2)*2*w+2*px+p%2)*outC:][:outC]
					for oc, ic := range mapping {
						dst[oc] = src[ic]
					}
				}
			}
		}
	})
	return out
}

// Decoder turns a latent into pixels.
type Decoder struct {
	postQuant *conv
	convIn    *conv
	midRes    [2]*resBlock
	midAttn   *attnBlock
	ups       []*upBlock
	normOut   []tensai.Float
	convOut   *conv
}

// stage describes one upsampling block: its channel counts, whether it
// upsamples, and whether the checkpoint pairs that with a doubling in
// time. They follow dim_mult reversed against decoder_base_dim 144.
var stages = []struct {
	in, out      int
	up, temporal bool
}{
	{1152, 1152, true, true},
	{1152, 1152, true, true},
	{1152, 576, true, true},
	{576, 288, true, false},
	{288, 144, false, false},
}

// vaeWeights is what the decoder reads: tensors, and whether an optional
// one (a resnet's shortcut) is there.
type vaeWeights interface {
	weights
	Info(string) (string, []int, bool)
}

// LoadDecoder reads the decoder half of a Qwen-Image-2.1 VAE checkpoint.
func LoadDecoder(path string) (*Decoder, error) {
	file, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// ComfyUI ships the same decoder under Wan's own names.
	var f vaeWeights = file
	if isComfyVAE(file) {
		f = comfyVAE{file}
	}

	d := &Decoder{}
	if d.postQuant, err = loadConv(f, "post_quant_conv", 1); err != nil {
		return nil, err
	}
	if d.convIn, err = loadConv(f, "decoder.conv_in", 3); err != nil {
		return nil, err
	}
	for i := range d.midRes {
		if d.midRes[i], err = loadResBlock(f, fmt.Sprintf("decoder.mid_block.resnets.%d", i)); err != nil {
			return nil, err
		}
	}
	if d.midAttn, err = loadAttn(f, "decoder.mid_block.attentions.0"); err != nil {
		return nil, err
	}
	for i, s := range stages {
		u := &upBlock{}
		// num_res_blocks is 2 in the config and the decoder runs one
		// more than that per stage.
		for j := 0; j < 3; j++ {
			r, err := loadResBlock(f, fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", i, j))
			if err != nil {
				return nil, err
			}
			u.resnets = append(u.resnets, r)
		}
		if s.up {
			if u.up, err = loadConv(f, fmt.Sprintf("decoder.up_blocks.%d.upsampler.resample.1", i), 3); err != nil {
				return nil, err
			}
			u.dupT = 1
			if s.temporal {
				u.dupT = 2
			}
			u.dupOut = s.out
			u.dupRepeat = s.out * u.dupT * 4 / s.in
		}
		d.ups = append(d.ups, u)
	}
	if d.normOut, err = gamma(f, "decoder.norm_out.gamma"); err != nil {
		return nil, err
	}
	if d.convOut, err = loadConv(f, "decoder.conv_out", 3); err != nil {
		return nil, err
	}
	return d, nil
}

// Decode turns a (64, h, w) latent into a (4, 16h, 16w) RGBA image whose
// samples run from -1 to 1.
func Decode(d *Decoder, z *tensai.Tensor) (*tensai.Tensor, error) {
	return decode(d, z, nil)
}

// DecodeGPU runs decoder convolutions on a GPU, uploading one filter at
// a time so full-resolution features do not compete with resident weights.
func DecodeGPU(d *Decoder, z *tensai.Tensor) (*tensai.Tensor, error) {
	g, err := gpu.Open(gpu.HighPerformance)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	return decode(d, z, g)
}

func decode(d *Decoder, z *tensai.Tensor, g *gpu.Device) (*tensai.Tensor, error) {
	if len(z.Shape) != 3 || z.Shape[0] != d.postQuant.in {
		return nil, fmt.Errorf("qwenimage: want a (%d, h, w) latent, got shape %v", d.postQuant.in, z.Shape)
	}
	h, w := z.Shape[1], z.Shape[2]
	x, err := d.postQuant.apply(channelsLast(z), h, w, g)
	if err != nil {
		return nil, err
	}
	if x, err = d.convIn.apply(x, h, w, g); err != nil {
		return nil, err
	}
	if x, err = d.midRes[0].apply(x, h, w, g); err != nil {
		return nil, err
	}
	if x, err = d.midAttn.apply(x, h, w, g); err != nil {
		return nil, err
	}
	if x, err = d.midRes[1].apply(x, h, w, g); err != nil {
		return nil, err
	}
	for _, u := range d.ups {
		if x, h, w, err = u.apply(x, h, w, g); err != nil {
			return nil, err
		}
	}
	rmsNorm(x, d.normOut)
	kernels.Silu(x.Data)
	if x, err = d.convOut.apply(x, h, w, g); err != nil {
		return nil, err
	}
	out := tensai.NewTensor(x.Cols, h, w)
	for p := 0; p < h*w; p++ {
		for c := 0; c < x.Cols; c++ {
			out.Data[c*h*w+p] = min(max(x.Data[p*x.Cols+c], -1), 1)
		}
	}
	return out, nil
}

// channelsLast turns a (c, h, w) tensor into the per-pixel rows the
// decoder works in.
func channelsLast(z *tensai.Tensor) *tensai.Matrix {
	c, n := z.Shape[0], z.Shape[1]*z.Shape[2]
	m := tensai.NewMatrix(n, c)
	for i := 0; i < c; i++ {
		for p := 0; p < n; p++ {
			m.Data[p*c+i] = z.Data[i*n+p]
		}
	}
	return m
}

func clone(x *tensai.Matrix) *tensai.Matrix {
	return &tensai.Matrix{Rows: x.Rows, Cols: x.Cols, Data: append([]tensai.Float(nil), x.Data...)}
}

func gamma(f weights, name string) ([]tensai.Float, error) {
	t, err := f.Tensor(name)
	if err != nil {
		return nil, err
	}
	return t.Data, nil
}

// Image turns a decoded (4, h, w) tensor into an RGBA image. The
// decoder's samples run from -1 to 1, and the fourth channel is the
// alpha this checkpoint generates natively.
func Image(t *tensai.Tensor) (*image.NRGBA, error) {
	if len(t.Shape) != 3 || t.Shape[0] != 4 {
		return nil, fmt.Errorf("qwenimage: want a (4, h, w) image, got shape %v", t.Shape)
	}
	h, w := t.Shape[1], t.Shape[2]
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for p := 0; p < h*w; p++ {
		for c := 0; c < 4; c++ {
			v := (t.Data[c*h*w+p] + 1) * 0.5 * 255
			img.Pix[4*p+c] = uint8(min(max(v+0.5, 0), 255))
		}
	}
	return img, nil
}

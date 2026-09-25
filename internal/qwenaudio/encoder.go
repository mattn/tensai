package qwenaudio

import (
	"fmt"
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/internal/workpool"
)

// Whisper-large-v3's encoder, as Qwen2-Audio's audio_config sets it out.
const (
	encDim    = 1280
	encHeads  = 20
	encHeadSz = encDim / encHeads // 64
	encFFN    = 5120
	encLayers = 32
	encEps    = 1e-5
	encPos    = melFrames / 2 // 1500 positions, one per two mel frames
	lmDim     = 4096          // the language model's hidden size
)

// affine is a linear layer with an optional bias, (out, in) as stored.
type affine struct {
	w *tensai.Matrix
	b []tensai.Float
}

func (a affine) apply(x *tensai.Matrix) *tensai.Matrix {
	out := tensai.NewMatrix(x.Rows, a.w.Rows)
	if err := tensai.DotTBInto(out, x, a.w); err != nil {
		panic(err)
	}
	if a.b != nil {
		for r := range out.Rows {
			row := out.Data[r*out.Cols : (r+1)*out.Cols]
			for i, b := range a.b {
				row[i] += b
			}
		}
	}
	return out
}

type norm struct{ w, b []tensai.Float }

// apply is LayerNorm over each row.
func (n norm) apply(x *tensai.Matrix) *tensai.Matrix {
	out := tensai.NewMatrix(x.Rows, x.Cols)
	workpool.Run(x.Rows, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x.Data[r*x.Cols : (r+1)*x.Cols]
			var mean, v float64
			for _, x := range row {
				mean += float64(x)
			}
			mean /= float64(len(row))
			for _, x := range row {
				d := float64(x) - mean
				v += d * d
			}
			inv := 1 / math.Sqrt(v/float64(len(row))+encEps)
			o := out.Data[r*x.Cols : (r+1)*x.Cols]
			for i, x := range row {
				o[i] = tensai.Float((float64(x)-mean)*inv)*n.w[i] + n.b[i]
			}
		}
	})
	return out
}

type encLayer struct {
	ln1, ln2   norm
	q, k, v, o affine
	fc1, fc2   affine
}

// Encoder is the audio tower and the projection into the language
// model's embedding space.
type Encoder struct {
	conv1, conv2 affine // kernels flattened to (out, in*3)
	pos          *tensai.Matrix
	layers       []encLayer
	lnPost       norm
	proj         affine
}

// LoadEncoder reads the audio tower and projector out of a Qwen2-Audio
// checkpoint, given its model.safetensors.index.json. layers cuts the
// tower short for testing against a reference of the same depth; 0 reads
// all of it. The weights stay float32: at 640M parameters they are 2.5GB,
// and the encoder is released before the language model loads.
func LoadEncoder(index string, layers int) (*Encoder, error) {
	f, err := safetensors.OpenSharded(index)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if layers <= 0 {
		layers = encLayers
	}
	var loadErr error
	tensor := func(name string, shape ...int) *tensai.Tensor {
		if loadErr != nil {
			return nil
		}
		t, err := f.Tensor(name)
		if err != nil {
			loadErr = err
			return nil
		}
		n := 1
		for _, s := range shape {
			n *= s
		}
		if len(t.Data) != n {
			loadErr = fmt.Errorf("qwenaudio: %s: shape %v, want %v", name, t.Shape, shape)
			return nil
		}
		return t
	}
	vec := func(name string, n int) []tensai.Float {
		if t := tensor(name, n); t != nil {
			return t.Data
		}
		return nil
	}
	mat := func(name string, rows, cols int) *tensai.Matrix {
		if t := tensor(name, rows, cols); t != nil {
			return &tensai.Matrix{Rows: rows, Cols: cols, Data: t.Data}
		}
		return nil
	}
	lin := func(p string, out, in int, bias bool) affine {
		a := affine{w: mat(p+".weight", out, in)}
		if bias {
			a.b = vec(p+".bias", out)
		}
		return a
	}
	const t = "audio_tower."
	e := &Encoder{
		// Conv1d stores (out, in, 3), which is already the (out, in*3)
		// an unrolled input multiplies against.
		conv1:  lin(t+"conv1", encDim, nMels*3, true),
		conv2:  lin(t+"conv2", encDim, encDim*3, true),
		pos:    mat(t+"embed_positions.weight", encPos, encDim),
		lnPost: norm{vec(t+"layer_norm.weight", encDim), vec(t+"layer_norm.bias", encDim)},
		proj:   lin("multi_modal_projector.linear", lmDim, encDim, true),
	}
	for i := range layers {
		p := fmt.Sprintf(t+"layers.%d.", i)
		e.layers = append(e.layers, encLayer{
			ln1: norm{vec(p+"self_attn_layer_norm.weight", encDim), vec(p+"self_attn_layer_norm.bias", encDim)},
			ln2: norm{vec(p+"final_layer_norm.weight", encDim), vec(p+"final_layer_norm.bias", encDim)},
			q:   lin(p+"self_attn.q_proj", encDim, encDim, true),
			k:   lin(p+"self_attn.k_proj", encDim, encDim, false),
			v:   lin(p+"self_attn.v_proj", encDim, encDim, true),
			o:   lin(p+"self_attn.out_proj", encDim, encDim, true),
			fc1: lin(p+"fc1", encFFN, encDim, true),
			fc2: lin(p+"fc2", encDim, encFFN, true),
		})
	}
	if loadErr != nil {
		return nil, loadErr
	}
	return e, nil
}

// Tokens is how many audio tokens frames of log-mel become: the second
// convolution halves them and the pooling after the encoder halves them
// again.
func Tokens(frames int) int {
	return (encTokens(frames)-2)/2 + 1
}

func encTokens(frames int) int { return (frames-1)/2 + 1 }

// Encode turns a log-mel spectrogram from LogMel into the language
// model's input embeddings for the audio, one row per token.
//
// The reference pads to thirty seconds and masks the padding out of the
// attention. The padding's outputs are thrown away, so this runs the
// encoder over the audio's own positions only, which is the same answer
// for a fraction of the work on a short clip.
func (e *Encoder) Encode(mel *tensai.Matrix, frames int) *tensai.Matrix {
	n := encTokens(frames)
	x := gelu(e.conv2.apply(unroll(gelu(e.conv1.apply(unroll(transpose(mel), 1, min(melFrames, 2*n+1)))), 2, n)))
	for i := range x.Data {
		x.Data[i] += e.pos.Data[i]
	}
	for _, l := range e.layers {
		a := l.attention(l.ln1.apply(x))
		for i := range x.Data {
			x.Data[i] += a.Data[i]
		}
		h := l.fc2.apply(gelu(l.fc1.apply(l.ln2.apply(x))))
		for i := range x.Data {
			x.Data[i] += h.Data[i]
		}
	}
	// Average neighbouring pairs; an odd last token has no partner and
	// the pooling drops it.
	out := tensai.NewMatrix(Tokens(frames), encDim)
	for r := range out.Rows {
		a, b := x.Data[2*r*encDim:(2*r+1)*encDim], x.Data[(2*r+1)*encDim:(2*r+2)*encDim]
		for i := range encDim {
			out.Data[r*encDim+i] = (a[i] + b[i]) / 2
		}
	}
	return e.proj.apply(e.lnPost.apply(out))
}

func (l *encLayer) attention(x *tensai.Matrix) *tensai.Matrix {
	q, k, v := l.q.apply(x), l.k.apply(x), l.v.apply(x)
	n := x.Rows
	ctx := tensai.NewMatrix(n, encDim)
	scale := tensai.Float(1 / math.Sqrt(encHeadSz))
	workpool.Run(encHeads, 1, func(lo, hi int) {
		qh, kh, vh := tensai.NewMatrix(n, encHeadSz), tensai.NewMatrix(n, encHeadSz), tensai.NewMatrix(encHeadSz, n)
		s := tensai.NewMatrix(n, n)
		o := tensai.NewMatrix(n, encHeadSz)
		for h := lo; h < hi; h++ {
			for t := range n {
				src := t*encDim + h*encHeadSz
				for i := range encHeadSz {
					qh.Data[t*encHeadSz+i] = q.Data[src+i] * scale
					kh.Data[t*encHeadSz+i] = k.Data[src+i]
					vh.Data[i*n+t] = v.Data[src+i]
				}
			}
			if err := tensai.DotTBIntoSerial(s, qh, kh); err != nil {
				panic(err)
			}
			for t := range n {
				softmax(s.Data[t*n : (t+1)*n])
			}
			if err := tensai.DotTBIntoSerial(o, s, vh); err != nil {
				panic(err)
			}
			for t := range n {
				copy(ctx.Data[t*encDim+h*encHeadSz:t*encDim+(h+1)*encHeadSz], o.Data[t*encHeadSz:(t+1)*encHeadSz])
			}
		}
	})
	return l.o.apply(ctx)
}

func softmax(x []tensai.Float) {
	top := x[0]
	for _, v := range x {
		top = max(top, v)
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - top))
		x[i] = tensai.Float(e)
		sum += e
	}
	inv := tensai.Float(1 / sum)
	for i := range x {
		x[i] *= inv
	}
}

// gelu is the exact, erf-based GELU, in place.
func gelu(x *tensai.Matrix) *tensai.Matrix {
	workpool.Run(len(x.Data), 1024, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			v := float64(x.Data[i])
			x.Data[i] = tensai.Float(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
		}
	})
	return x
}

func transpose(m *tensai.Matrix) *tensai.Matrix {
	t := tensai.NewMatrix(m.Cols, m.Rows)
	for r := range m.Rows {
		for c := range m.Cols {
			t.Data[c*m.Rows+r] = m.Data[r*m.Cols+c]
		}
	}
	return t
}

// unroll lays out a kernel-3, padding-1 convolution's inputs so it is one
// matrix product: row t holds, channel by channel, the three time steps
// the output at t reads, zero past either end. x is (time, channels).
func unroll(x *tensai.Matrix, stride, outs int) *tensai.Matrix {
	c := x.Cols
	u := tensai.NewMatrix(outs, c*3)
	workpool.Run(outs, 1, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := u.Data[t*c*3 : (t+1)*c*3]
			for k := range 3 {
				src := t*stride + k - 1
				if src < 0 || src >= x.Rows {
					continue
				}
				in := x.Data[src*c : (src+1)*c]
				for ch, v := range in {
					row[ch*3+k] = v
				}
			}
		}
	})
	return u
}

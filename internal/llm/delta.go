package llm

// Gated delta rule, the linear-attention half of a qwen3_5 model. Most of
// its layers run this instead of attention: a short causal convolution
// mixes each channel with its three predecessors, and a per-head matrix
// state absorbs the token and answers the query. The state is a fixed
// [key_dim, value_dim] per head however long the context grows, which is
// what the architecture buys — and what makes it not a KV cache, since
// nothing in it can be indexed by position or rolled back.

import (
	"fmt"
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// deltaWeights is one linear-attention layer.
type deltaWeights struct {
	heads    int // value heads; key heads are the same count here
	kDim     int // per-head key/query width
	vDim     int // per-head value width
	convK    int
	convDim  int       // kDim*heads*2 + vDim*heads, the width the conv spans
	conv     []float32 // [convDim, convK], one causal filter per channel
	aLog     []float32 // [heads]
	dtBias   []float32 // [heads]
	norm     []float32 // [vDim], the gated RMSNorm weight
	wQKV     *tensai.Matrix
	qQKV     *qmat
	wZ, wOut *tensai.Matrix
	qZ, qOut *qmat
	// a and b are [hidden, heads], small enough to leave in float32.
	wA, wB *tensai.Matrix
}

// deltaState is what a linear-attention layer carries between tokens: the
// recurrent state per head and the tail of the convolution window.
type deltaState struct {
	s    []float32 // [heads, kDim, vDim]
	conv []float32 // [convK-1, convDim], oldest first
}

func (d *deltaWeights) newState() *deltaState {
	return &deltaState{
		s:    make([]float32, d.heads*d.kDim*d.vDim),
		conv: make([]float32, (d.convK-1)*d.convDim),
	}
}

// linearLayer reports whether layer i runs the delta rule rather than
// attention. Layers are named one by one in the checkpoint; the interval
// is only a description of the pattern they follow.
func (c config) linearLayer(i int) bool {
	if i < len(c.LayerTypes) {
		return c.LayerTypes[i] == "linear_attention"
	}
	return false
}

// loadDelta reads a linear-attention layer's weights. The conv arrives as
// [convDim, 1, convK] and stays in that order; everything else follows the
// loader's usual transpose-and-quantize path.
func loadDelta(cfg config, p string, vec func(string) []float32,
	linq func(string) (*tensai.Matrix, *qmat),
	linqF32 func(string) *tensai.Matrix) *deltaWeights {
	d := &deltaWeights{
		heads:  cfg.LinearValueHeads,
		kDim:   cfg.LinearKeyDim,
		vDim:   cfg.LinearValueDim,
		convK:  cfg.LinearConvK,
		aLog:   vec(p + "linear_attn.A_log"),
		dtBias: vec(p + "linear_attn.dt_bias"),
		norm:   vec(p + "linear_attn.norm.weight"),
		conv:   vec(p + "linear_attn.conv1d.weight"),
	}
	d.convDim = cfg.LinearKeyHeads*cfg.LinearKeyDim*2 + cfg.LinearValueHeads*cfg.LinearValueDim
	d.wQKV, d.qQKV = linq(p + "linear_attn.in_proj_qkv.weight")
	d.wZ, d.qZ = linq(p + "linear_attn.in_proj_z.weight")
	d.wOut, d.qOut = linq(p + "linear_attn.out_proj.weight")
	// a and b are [hidden, heads] -- sixteen columns. Quantizing them
	// would save nothing and cost the precision the decay depends on.
	d.wA = linqF32(p + "linear_attn.in_proj_a.weight")
	d.wB = linqF32(p + "linear_attn.in_proj_b.weight")
	if err := d.check(); err != nil {
		panic(err)
	}
	return d
}

// check reports a shape the forward pass could not survive, which is
// cheaper to say at load than to debug as garbage tokens.
func (d *deltaWeights) check() error {
	if got := len(d.conv); got != d.convDim*d.convK {
		return fmt.Errorf("delta conv1d has %d weights, want %d", got, d.convDim*d.convK)
	}
	for _, x := range []struct {
		name string
		got  int
		want int
	}{
		{"A_log", len(d.aLog), d.heads},
		{"dt_bias", len(d.dtBias), d.heads},
		{"norm", len(d.norm), d.vDim},
	} {
		if x.got != x.want {
			return fmt.Errorf("delta %s has %d weights, want %d", x.name, x.got, x.want)
		}
	}
	return nil
}

// l2norm scales v to unit length, matching the reference's epsilon inside
// the square root rather than outside it.
func l2norm(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	inv := float32(1 / math.Sqrt(float64(sum)+1e-6))
	for i := range v {
		v[i] *= inv
	}
}

func softplus(x float32) float32 {
	// log1p(exp(x)) without overflowing on the large side.
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// step advances one token through the layer, in place on x. conv holds the
// previous convK-1 rows; the state absorbs the token and answers the query.
func (d *deltaWeights) step(st *deltaState, x []float32, scratch *deltaScratch) []float32 {
	qkv := scratch.qkv
	mvInto(qkv, x, d.wQKV, d.qQKV, nil)
	z := scratch.z
	mvInto(z, x, d.wZ, d.qZ, nil)
	a := scratch.a
	mvInto(a, x, d.wA, nil, nil)
	b := scratch.b
	mvInto(b, x, d.wB, nil, nil)

	// Causal depthwise convolution over the qkv channels, then silu. The
	// window is the state's convK-1 rows followed by this token.
	kw := d.convK
	prev := st.conv
	out := scratch.conv
	for c := 0; c < d.convDim; c++ {
		var acc float32
		for t := 0; t < kw-1; t++ {
			acc += prev[t*d.convDim+c] * d.conv[c*kw+t]
		}
		acc += qkv[c] * d.conv[c*kw+kw-1]
		out[c] = silu(acc)
	}
	// Roll the window forward by one.
	copy(prev, prev[d.convDim:])
	copy(prev[(kw-2)*d.convDim:], qkv)

	kd, vd, h := d.kDim, d.vDim, d.heads
	keyDim := kd * h
	qs, ks, vs := out[:keyDim], out[keyDim:2*keyDim], out[2*keyDim:]
	res := scratch.out
	qScale := float32(1 / math.Sqrt(float64(kd)))
	for hi := 0; hi < h; hi++ {
		q := qs[hi*kd : (hi+1)*kd]
		k := ks[hi*kd : (hi+1)*kd]
		v := vs[hi*vd : (hi+1)*vd]
		l2norm(q)
		l2norm(k)
		for i := range q {
			q[i] *= qScale
		}
		beta := 1 / (1 + float32(math.Exp(float64(-b[hi]))))
		decay := float32(math.Exp(float64(-float32(math.Exp(float64(d.aLog[hi]))) * softplus(a[hi]+d.dtBias[hi]))))

		s := st.s[hi*kd*vd : (hi+1)*kd*vd]
		mem := scratch.mem[:vd]
		for j := range mem {
			mem[j] = 0
		}
		// Decay the state and read what it already holds for this key.
		for i := 0; i < kd; i++ {
			row := s[i*vd : (i+1)*vd]
			kernels.ScaleSlice(row, decay)
			if k[i] != 0 {
				tensai.Axpy(k[i], row, mem)
			}
		}
		// Write the difference back along the key, then answer the query.
		o := res[hi*vd : (hi+1)*vd]
		for j := range o {
			o[j] = 0
		}
		for i := 0; i < kd; i++ {
			row := s[i*vd : (i+1)*vd]
			if ki := k[i]; ki != 0 {
				for j := 0; j < vd; j++ {
					row[j] += ki * (v[j] - mem[j]) * beta
				}
			}
			if q[i] != 0 {
				tensai.Axpy(q[i], row, o)
			}
		}
		// Gated RMSNorm over the value dimension, then the gate.
		var ss float32
		for _, y := range o {
			ss += y * y
		}
		inv := float32(1 / math.Sqrt(float64(ss)/float64(vd)+1e-6))
		zg := z[hi*vd : (hi+1)*vd]
		for j := range o {
			o[j] = d.norm[j] * (o[j] * inv) * silu(zg[j])
		}
	}
	y := scratch.proj
	mvInto(y, res, d.wOut, d.qOut, nil)
	return y
}

// deltaScratch is one token's working set, reused across layers.
type deltaScratch struct {
	qkv, conv, z, a, b, mem, out, proj []float32
}

func newDeltaScratch(d *deltaWeights, hidden int) *deltaScratch {
	return &deltaScratch{
		qkv:  make([]float32, d.convDim),
		conv: make([]float32, d.convDim),
		z:    make([]float32, d.vDim*d.heads),
		a:    make([]float32, d.heads),
		b:    make([]float32, d.heads),
		mem:  make([]float32, d.vDim),
		out:  make([]float32, d.vDim*d.heads),
		proj: make([]float32, hidden),
	}
}

// hasDelta reports whether any layer runs the delta rule, which decides
// what the engine can offer around it.
func (m *qwen) hasDelta() bool {
	for i := range m.blocks {
		if m.blocks[i].delta != nil {
			return true
		}
	}
	return false
}

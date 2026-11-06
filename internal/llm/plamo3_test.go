package llm

import (
	"math"
	"testing"
)

// The real PLaMo 3 Translate 0.8B config: YaRN on the full layers, the
// sliding ones at their own rope base.
func TestLoadPlamo3Config(t *testing.T) {
	c, err := loadConfig("testdata/plamo3-config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		got  any
		want any
	}{
		{"model_type", c.ModelType, "plamo3"},
		{"layers", c.Layers, 16},
		{"hidden", c.HiddenSize, 1536},
		{"head_dim", c.HeadDim, 128},
		{"heads", c.Heads, 16},
		{"kv heads", c.KVHeads, 1},
		{"intermediate", c.Intermediate, 6144},
		{"vocab", c.Vocab, 107520},
		{"tied", c.TieEmbedding, false},
		{"rope theta", c.RopeTheta, 1000000.0},
		{"local rope theta", c.RopeLocalTheta, 10000.0},
		{"window", c.WindowSize, 2048},
		{"pattern", c.SWAEvery, 8},
		{"yarn factor", c.YarnFactor, 256.0},
		{"yarn ctx", c.YarnOrigCtx, 4096},
		{"yarn beta fast", c.YarnBetaFast, 32.0},
		{"yarn beta slow", c.YarnBetaSlow, 1.0},
		{"yarn exact", c.YarnExact, true},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %v, want %v", tt.name, tt.got, tt.want)
		}
	}
	var full []int
	for i := range c.Layers {
		if plamo3Full(c, i) {
			full = append(full, i)
		}
	}
	if len(full) != 2 || full[0] != 7 || full[1] != 15 {
		t.Errorf("full-attention layers = %v, want [7 15]", full)
	}
}

// plamo3Rope is the reference's rotary embedding for one head at one
// position, transcribed from modeling_plamo.py: inverse frequencies
// blended between interpolation and extrapolation over an unrounded
// ramp, and cos/sin scaled by the concentration.
func plamo3Rope(h []float64, pos int, theta, factor float64, orig int, betaFast, betaSlow float64) []float64 {
	dim := len(h)
	dh := dim / 2
	out := make([]float64, dim)
	low := float64(dh) * math.Log(float64(orig)/(betaFast*2*math.Pi)) / math.Log(theta)
	high := float64(dh) * math.Log(float64(orig)/(betaSlow*2*math.Pi)) / math.Log(theta)
	conc := 1.0
	if factor > 1 {
		conc = 0.1*math.Log(factor) + 1
	}
	for i := 0; i < dh; i++ {
		freq := math.Pow(theta, float64(2*i)/float64(dim))
		inv := 1 / freq
		if factor > 1 {
			ramp := (float64(i) - low) / (high - low)
			mask := 1 - math.Min(1, math.Max(0, ramp))
			inv = 1/(factor*freq)*(1-mask) + 1/freq*mask
		}
		s, c := math.Sincos(float64(pos) * inv)
		s, c = s*conc, c*conc
		out[i] = h[i]*c - h[i+dh]*s
		out[i+dh] = h[i+dh]*c + h[i]*s
	}
	return out
}

func TestPlamo3Rope(t *testing.T) {
	c, err := loadConfig("testdata/plamo3-config.json")
	if err != nil {
		t.Fatal(err)
	}
	m := &qwen{cfg: c, headSz: c.HeadDim, blocks: make([]qblock, 2)}
	// Block 0 slides, block 1 is full.
	m.blocks[0] = qblock{window: c.WindowSize, ropeTheta: c.RopeLocalTheta, noYarn: true}
	m.initRopeFreqs()
	for bi, b := range []*qblock{&m.blocks[0], &m.blocks[1]} {
		for _, pos := range []int{0, 1, 17, 3000, 100000} {
			h := make([]float32, c.HeadDim)
			ref := make([]float64, c.HeadDim)
			for i := range h {
				h[i] = float32(math.Sin(float64(i)*0.37 + 1))
				ref[i] = float64(h[i])
			}
			m.rope(h, pos, b)
			var want []float64
			if bi == 0 {
				want = plamo3Rope(ref, pos, c.RopeLocalTheta, 1, 0, 0, 0)
			} else {
				want = plamo3Rope(ref, pos, c.RopeTheta, c.YarnFactor, c.YarnOrigCtx, c.YarnBetaFast, c.YarnBetaSlow)
			}
			for i := range h {
				if d := math.Abs(float64(h[i]) - want[i]); d > 1e-4 {
					t.Fatalf("block %d pos %d dim %d: got %v, want %v", bi, pos, i, h[i], want[i])
				}
			}
		}
	}
}

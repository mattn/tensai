package llm

// PLaMo 3 (Preferred Networks): a Gemma-3-shaped decoder -- sandwich
// norms around attention and the MLP, per-head QK-norm, and one
// full-attention layer in every sliding_window_pattern with the rest
// seeing a window of window_size -- that differs in its details. q, k,
// and v come out of one fused projection and gate|up out of another,
// already in the column order the decoder wants; the MLP is SwiGLU, not
// GeGLU; the embedding is not scaled; and every RMSNorm multiplies by
// (offset + w) with an offset that depends on the norm: 1 before
// attention and the MLP and on q and k, 1/5 after attention, and
// 1/5^1.5 after the MLP. The offsets fold into the weights here, so the
// kernels see plain RMSNorms. The full layers turn RoPE with YaRN at
// rope_theta, the sliding ones plainly at rope_local_theta.

import (
	"encoding/json"
	"fmt"
	"math"

	tensai "github.com/mattn/tensai"
)

// plamo3Config is the part of PLaMo 3's config.json that loadConfig's
// generic fields do not cover.
type plamo3Config struct {
	RopeParameters struct {
		Full struct {
			Type     string  `json:"rope_type"`
			Theta    float64 `json:"rope_theta"`
			Factor   float64 `json:"factor"`
			OrigCtx  int     `json:"original_max_position_embeddings"`
			BetaFast float64 `json:"beta_fast"`
			BetaSlow float64 `json:"beta_slow"`
			Truncate *bool   `json:"truncate"`
		} `json:"full_attention"`
	} `json:"rope_parameters"`
	ScalingFactor float64 `json:"rope_scaling_factor"`
	InitialCtx    int     `json:"initial_context_length"`
	ScaleEmbed    bool    `json:"scale_embedding"`
	LinearType    string  `json:"linear_type"`
	ImageFeature  *int    `json:"image_feature_size"`
}

func applyPlamo3(c *config, raw []byte) error {
	var p plamo3Config
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("parsing plamo3 config: %w", err)
	}
	if p.ScaleEmbed {
		return fmt.Errorf("plamo3 with scale_embedding is not supported")
	}
	if p.LinearType != "" && p.LinearType != "normal" {
		return fmt.Errorf("plamo3 linear_type %q is not supported", p.LinearType)
	}
	if c.SWAEvery <= 0 {
		c.SWAEvery = 8
	}
	if c.WindowSize <= 0 {
		c.WindowSize = 2048
	}
	if c.RopeLocalTheta == 0 {
		c.RopeLocalTheta = 10000
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 150000
	}
	full := p.RopeParameters.Full
	switch {
	case full.Type == "yarn":
		if full.Theta != 0 {
			c.RopeTheta = full.Theta
		}
		if full.Truncate != nil && *full.Truncate {
			return fmt.Errorf("plamo3 YaRN with truncate=true is not supported")
		}
		c.YarnFactor = full.Factor
		c.YarnOrigCtx = full.OrigCtx
		c.YarnBetaFast = full.BetaFast
		c.YarnBetaSlow = full.BetaSlow
	case full.Type == "" && p.ScalingFactor > 1:
		// Older configs state only the factor; the rest are the
		// reference's defaults.
		c.YarnFactor = p.ScalingFactor
		c.YarnOrigCtx = p.InitialCtx
		c.YarnBetaFast, c.YarnBetaSlow = 32, 1
	}
	if c.YarnFactor > 1 {
		c.YarnExact = true
		if c.YarnOrigCtx == 0 {
			c.YarnOrigCtx = 4096
		}
		if c.YarnBetaFast == 0 {
			c.YarnBetaFast = 32
		}
		if c.YarnBetaSlow == 0 {
			c.YarnBetaSlow = 1
		}
	}
	// The HF loader keeps the language model under model.layers.layers.N;
	// the embedding, final norm, and head sit where the others' do.
	c.LayerTypes = nil
	return nil
}

// plamo3Full reports whether layer i attends to the whole context.
func plamo3Full(c config, i int) bool {
	return (i+1)%c.SWAEvery == 0
}

// loadPlamo3Block reads one decoder layer.
func loadPlamo3Block(b *qblock, cfg config, i int,
	vec func(string) []float32,
	linq func(string) (*tensai.Matrix, *qmat)) {
	p := fmt.Sprintf("model.layers.layers.%d.", i)
	norm := func(name string, offset float64) []float32 {
		v := vec(p + name)
		w := make([]float32, len(v))
		for j, x := range v {
			w[j] = float32(offset + float64(x))
		}
		return w
	}
	b.ln1 = norm("pre_mixer_norm.weight", 1)
	b.postAttn = norm("post_mixer_norm.weight", 1.0/5)
	b.ln2 = norm("pre_mlp_norm.weight", 1)
	b.postFFN = norm("post_mlp_norm.weight", 1/math.Pow(5, 1.5))
	b.qNorm = norm("mixer.q_norm.weight", 1)
	b.kNorm = norm("mixer.k_norm.weight", 1)
	b.wQKV, b.qQKV = linq(p + "mixer.qkv_proj.weight")
	b.wo, b.qo = linq(p + "mixer.o_proj.weight")
	b.wGU, b.qGU = linq(p + "mlp.gate_up_proj.weight")
	b.wDown, b.qDown = linq(p + "mlp.down_proj.weight")
	if !plamo3Full(cfg, i) {
		b.window = cfg.WindowSize
		b.ropeTheta = cfg.RopeLocalTheta
		b.noYarn = true
	}
}

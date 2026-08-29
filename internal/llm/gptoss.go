package llm

// gpt-oss's published checkpoint keeps its mixture-of-experts weights
// MXFP4-packed inside the safetensors file rather than as bf16: each
// expert projection arrives as a U8 "_blocks" tensor of nibble pairs
// beside a U8 "_scales" tensor of e8m0 exponents, one per group of 32
// inputs. Everything else in the block -- attention, router, norms -- is
// ordinary bf16.
//
// Two things differ from the same weights seen through a GGUF, and both
// are silent if you get them wrong, because a permuted weight still
// produces fluent-looking nonsense:
//
//   - Nibble order. llama.cpp splits a 16-byte group as value i and
//     value i+16; the reference unpacker here pairs them, low nibble to
//     2i and high to 2i+1.
//   - gate/up interleaving. gpt-oss reads its fused projection as
//     gate = out[..., 0::2] and up = out[..., 1::2], while moeFFN wants
//     them concatenated. The GGUF converter splits them into separate
//     ffn_gate_exps/ffn_up_exps tensors, so the GGUF path never sees
//     this; loading the original weights has to de-interleave.

import (
	"fmt"
	"math"

	tensai "github.com/mattn/tensai"
)

// mxfp4Table is FP4 (E2M1) doubled onto an integer grid, matching
// encoding/gguf's; the block scale halves to compensate.
var mxfp4Table = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// dequantMXFP4Row expands one row: len(scales) groups of 32 values from
// 16 bytes each. dst must hold 32*len(scales) floats.
func dequantMXFP4Row(blocks, scales []byte, dst []float32) {
	for g, e := range scales {
		s := float32(math.Ldexp(1, int(e)-128))
		blk := blocks[g*16 : g*16+16]
		out := dst[g*32 : g*32+32]
		for i, q := range blk {
			out[2*i] = s * mxfp4Table[q&0x0F]
			out[2*i+1] = s * mxfp4Table[q>>4]
		}
	}
}

// mxfp4Expert reads one expert's projection out of the [experts, out,
// groups, 16] block tensor and its [experts, out, groups] scales, and
// returns it already transposed to the [in, out] a matvec wants.
//
// interleaved de-interleaves gpt-oss's fused gate|up on the way: output
// row 2k is gate column k and row 2k+1 is up column k, so the result
// carries gate in the first half of its columns and up in the second,
// which is the layout moeFFN slices.
func mxfp4Expert(blocks, scales []byte, shape []int, e int, interleaved bool) (*tensai.Matrix, error) {
	if len(shape) != 4 || shape[3] != 16 {
		return nil, fmt.Errorf("gpt-oss: expert blocks have shape %v, want [experts, out, groups, 16]", shape)
	}
	experts, out, groups := shape[0], shape[1], shape[2]
	if e < 0 || e >= experts {
		return nil, fmt.Errorf("gpt-oss: expert %d out of range %d", e, experts)
	}
	if interleaved && out%2 != 0 {
		return nil, fmt.Errorf("gpt-oss: fused gate|up has odd width %d", out)
	}
	in := groups * 32
	if want := experts * out * groups * 16; len(blocks) != want {
		return nil, fmt.Errorf("gpt-oss: expert blocks are %d bytes, want %d", len(blocks), want)
	}
	if want := experts * out * groups; len(scales) != want {
		return nil, fmt.Errorf("gpt-oss: expert scales are %d bytes, want %d", len(scales), want)
	}
	w := tensai.NewMatrix(in, out)
	row := make([]float32, in)
	half := out / 2
	for r := 0; r < out; r++ {
		i := e*out + r
		dequantMXFP4Row(blocks[i*groups*16:(i+1)*groups*16], scales[i*groups:(i+1)*groups], row)
		c := r
		if interleaved {
			if r%2 == 0 {
				c = r / 2
			} else {
				c = half + r/2
			}
		}
		for k, v := range row {
			w.Data[k*out+c] = v
		}
	}
	return w, nil
}

// deinterleaveBias reorders a fused gate|up bias the same way
// mxfp4Expert reorders the weight's columns.
func deinterleaveBias(v []float32) []float32 {
	if v == nil {
		return nil
	}
	out := make([]float32, len(v))
	half := len(v) / 2
	for i, x := range v {
		if i%2 == 0 {
			out[i/2] = x
		} else {
			out[half+i/2] = x
		}
	}
	return out
}

// gptOssConfig fills in what gpt-oss's config.json spells differently
// from the fields the runtime already reads from GGUF metadata. Its
// mixture-of-experts sizes, YaRN parameters, and per-layer attention
// spans all live under Hugging Face names.
type gptOssConfig struct {
	NumLocalExperts int      `json:"num_local_experts"`
	ExpertsPerTok   int      `json:"num_experts_per_tok"`
	ExpertsPerToken int      `json:"experts_per_token"`
	LayerTypes      []string `json:"layer_types"`
	RopeScaling     *struct {
		Type     string  `json:"rope_type"`
		Factor   float64 `json:"factor"`
		BetaFast float64 `json:"beta_fast"`
		BetaSlow float64 `json:"beta_slow"`
		OrigCtx  int     `json:"original_max_position_embeddings"`
	} `json:"rope_scaling"`
}

// applyGptOss folds the Hugging Face spelling into the config the rest of
// the loader and the runtime already understand.
func applyGptOss(c *config, hf gptOssConfig) {
	c.NExpert = hf.NumLocalExperts
	c.NExpertUsed = hf.ExpertsPerTok
	if c.NExpertUsed == 0 {
		c.NExpertUsed = hf.ExpertsPerToken
	}
	// The experts are the FFN; intermediate_size is their width.
	c.MoeFF = c.Intermediate
	if r := hf.RopeScaling; r != nil && r.Type == "yarn" {
		c.YarnFactor = r.Factor
		c.YarnOrigCtx = r.OrigCtx
		c.YarnBetaFast = r.BetaFast
		c.YarnBetaSlow = r.BetaSlow
	}
}

// gptOssSlides reports whether layer i uses the sliding window. The
// checkpoint lists this per layer; gpt-oss alternates, starting sliding.
func gptOssSlides(hf gptOssConfig, i int) bool {
	if i < len(hf.LayerTypes) {
		return hf.LayerTypes[i] == "sliding_attention"
	}
	return i%2 == 0
}

// loadGptOssExperts fills in a block's mixture of experts from the
// MXFP4-packed safetensors tensors. Each expert is dequantized to
// float32, requantized to the runtime's width, and dropped again, so the
// peak is one expert's worth of float32 per worker rather than the whole
// layer's.
func loadGptOssExperts(
	b *qblock,
	cfg config,
	p string,
	bits int,
	raw func(string) ([]byte, []int),
	vecOpt func(string) []float32,
	router func(string) *tensai.Matrix,
) {
	b.router = router(p + "mlp.router.weight")
	b.routerBias = vecOpt(p + "mlp.router.bias")
	b.topK = cfg.NExpertUsed
	b.softmaxK = true // gpt-oss softmaxes over the chosen experts, not all
	b.oaiGLU = true   // clamped SwiGLU with the (up+1) linear term
	b.experts = make([]expertFFN, cfg.NExpert)

	guBlocks, guShape := raw(p + "mlp.experts.gate_up_proj_blocks")
	guScales, _ := raw(p + "mlp.experts.gate_up_proj_scales")
	dnBlocks, dnShape := raw(p + "mlp.experts.down_proj_blocks")
	dnScales, _ := raw(p + "mlp.experts.down_proj_scales")
	guBias := vecOpt(p + "mlp.experts.gate_up_proj_bias")
	dnBias := vecOpt(p + "mlp.experts.down_proj_bias")

	guWidth := guShape[1]
	for e := range b.experts {
		gu, err := mxfp4Expert(guBlocks, guScales, guShape, e, true)
		if err != nil {
			panic(err)
		}
		dn, err := mxfp4Expert(dnBlocks, dnScales, dnShape, e, false)
		if err != nil {
			panic(err)
		}
		ex := &b.experts[e]
		if bits == 0 {
			panic("gpt-oss needs -q8 or -q4: its experts are already 4-bit in the checkpoint")
		}
		ex.qGU = quantizeMat(gu, bits)
		ex.qDown = quantizeMat(dn, bits)
		if guBias != nil {
			ex.guBias = deinterleaveBias(guBias[e*guWidth : (e+1)*guWidth])
		}
		if dnBias != nil {
			ex.downBias = dnBias[e*cfg.HiddenSize : (e+1)*cfg.HiddenSize]
		}
	}
}

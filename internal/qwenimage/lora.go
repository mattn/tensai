package qwenimage

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/kernels"
)

// lowRank stays separate from the quantized base weights. Folding it into
// those weights would round away much of the distilled update.
type lowRank struct {
	a, b  *tensai.Matrix
	scale tensai.Float
}

func (m *Transformer) turboLayers() map[string]*linear {
	ls := map[string]*linear{
		"modulation.1": m.modulation,
		"time_text_embed.timestep_embedder.linear_1": m.timeIn,
		"time_text_embed.timestep_embedder.linear_2": m.timeUp,
	}
	for i, b := range m.blocks {
		for name, l := range map[string]*linear{
			"attn.to_q": b.toQ, "attn.to_k": b.toK, "attn.to_v": b.toV, "attn.to_out.0": b.toOut,
			"img_mlp.proj": b.mlpProj, "img_mlp.gate_layer": b.mlpGate, "img_mlp.out": b.mlpOut,
		} {
			ls[fmt.Sprintf("transformer_blocks.%d.%s", i, name)] = l
		}
	}
	return ls
}

// LoadTurboLoRA attaches a Viggle Qwen-Image-2.1 six-step adapter before
// UseGPU. Both rank-128 and rank-256 diffusers-format safetensors are supported. The
// adapter must be used with NewTurboSchedule and without CFG.
func LoadTurboLoRA(m *Transformer, path string) error {
	if m.dev != nil {
		return fmt.Errorf("qwenimage: load the adapter before enabling the GPU")
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var meta struct {
		Rank   int                `json:"transformer.r"`
		Alpha  float64            `json:"transformer.lora_alpha"`
		RS     bool               `json:"transformer.use_rslora"`
		Bias   string             `json:"transformer.bias"`
		Ranks  map[string]int     `json:"transformer.rank_pattern"`
		Alphas map[string]float64 `json:"transformer.alpha_pattern"`
	}
	if err := json.Unmarshal([]byte(f.Metadata()["lora_adapter_metadata"]), &meta); err != nil {
		return fmt.Errorf("qwenimage: invalid LoRA metadata: %w", err)
	}
	if meta.Rank <= 0 || meta.Alpha <= 0 || math.IsInf(meta.Alpha, 0) || meta.RS || meta.Bias != "none" || len(meta.Ranks) > 0 || len(meta.Alphas) > 0 {
		return fmt.Errorf("qwenimage: unsupported LoRA configuration")
	}
	ls := m.turboLayers()
	if len(f.Names()) != 2*len(ls) {
		return fmt.Errorf("qwenimage: adapter needs %d A/B pairs, got %d tensors", len(ls), len(f.Names()))
	}
	pending := make(map[*linear]*lowRank, len(ls))
	for name, l := range ls {
		if l == nil {
			return fmt.Errorf("qwenimage: missing base layer %s", name)
		}
		if l.lora != nil {
			return fmt.Errorf("qwenimage: an adapter is already loaded")
		}
		a, err := f.Tensor("transformer." + name + ".lora_A.weight")
		if err != nil {
			return err
		}
		b, err := f.Tensor("transformer." + name + ".lora_B.weight")
		if err != nil {
			return err
		}
		out := 0
		switch {
		case l.q != nil:
			out = l.q.Cols
		case l.q4 != nil:
			out = l.q4.Cols
		default:
			out = l.f.Rows
		}
		if len(a.Shape) != 2 || len(b.Shape) != 2 || a.Shape[0] != meta.Rank || a.Shape[1] != l.inputs() || b.Shape[0] != out || b.Shape[1] != meta.Rank {
			return fmt.Errorf("qwenimage: %s adapter shapes %v, %v do not match base %dx%d and rank %d", name, a.Shape, b.Shape, out, l.inputs(), meta.Rank)
		}
		// Own the data even for F32, whose file tensor aliases the mmap.
		aa := &tensai.Matrix{Rows: a.Shape[0], Cols: a.Shape[1], Data: append([]tensai.Float(nil), a.Data...)}
		bb := &tensai.Matrix{Rows: b.Shape[0], Cols: b.Shape[1], Data: append([]tensai.Float(nil), b.Data...)}
		if l.rot > 0 {
			rotateRows(aa, l.rot)
		}
		pending[l] = &lowRank{a: aa, b: bb, scale: tensai.Float(meta.Alpha / float64(meta.Rank))}
	}
	for l, lr := range pending {
		l.lora = lr
	}
	return nil
}

func (lr *lowRank) add(out, x *tensai.Matrix) error {
	h := tensai.NewMatrix(x.Rows, lr.a.Rows)
	if err := tensai.DotTBInto(h, x, lr.a); err != nil {
		return err
	}
	delta := tensai.NewMatrix(x.Rows, lr.b.Rows)
	if err := tensai.DotTBInto(delta, h, lr.b); err != nil {
		return err
	}
	kernels.Axpy(lr.scale, delta.Data, out.Data)
	return nil
}

// Adapter weights are streamed: keeping all float adapters resident
// would exceed the integrated GPU's budget even with an int8 base.
func (lr *lowRank) addGPU(g *gpu.Device, out, x *gpu.Tensor) error {
	a, err := g.Upload(lr.a.Tensor())
	if err != nil {
		return err
	}
	defer a.Free()
	b, err := g.Upload(lr.b.Tensor())
	if err != nil {
		return err
	}
	defer b.Free()
	h, err := x.MatMulT(a)
	if err != nil {
		return err
	}
	defer h.Free()
	delta, err := h.MatMulT(b)
	if err != nil {
		return err
	}
	defer delta.Free()
	if lr.scale != 1 {
		if err := delta.Scale(lr.scale); err != nil {
			return err
		}
	}
	return out.Add(delta)
}

func (lr *lowRank) bytes() uint64 {
	if lr == nil {
		return 0
	}
	return uint64(len(lr.a.Data)+len(lr.b.Data)) * 4
}

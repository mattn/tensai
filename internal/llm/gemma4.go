package llm

// Gemma 4 is a Gemma 3 block with three additions. Its layers alternate
// two head widths, two rope bases, and two feed-forward widths instead of
// one; the deeper two thirds project queries only and attend against an
// earlier layer's KV cache; and every block ends with a per-layer
// embedding — a slice of a second, vocabulary-sized table that is far too
// large to hold in memory and is read one token's row at a time.

import (
	"fmt"
	"math"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/gguf"
)

// gemma4Config reads the per-layer geometry out of the GGUF metadata.
// Everything here is stated per layer or not at all, so a missing array
// is a file this loader cannot honestly claim to understand.
func gemma4Config(g *gguf.File, cfg *config) error {
	for _, n := range g.Ints("gemma4.feed_forward_length") {
		cfg.FFPerLayer = append(cfg.FFPerLayer, int(n))
		cfg.Intermediate = max(cfg.Intermediate, int(n))
	}
	cfg.SWAPattern = g.Bools("gemma4.attention.sliding_window_pattern")
	if len(cfg.FFPerLayer) != cfg.Layers || len(cfg.SWAPattern) != cfg.Layers {
		return fmt.Errorf("gemma4 states %d feed-forward widths and %d window flags for %d layers",
			len(cfg.FFPerLayer), len(cfg.SWAPattern), cfg.Layers)
	}
	cfg.HeadDimSWA = int(mustInt(g, "gemma4.attention.key_length_swa"))
	cfg.RopeThetaSWA, _ = g.Float("gemma4.rope.freq_base_swa")
	cfg.PLEDim = int(mustInt(g, "gemma4.embedding_length_per_layer_input"))
	cfg.LogitCap, _ = g.Float("gemma4.final_logit_softcapping")
	// The last shared_kv_layers layers keep no cache of their own.
	shared := int(mustInt(g, "gemma4.attention.shared_kv_layers"))
	cfg.KVFromStart = cfg.Layers - shared
	if cfg.KVFromStart < 2 {
		return fmt.Errorf("gemma4 shares the KV cache of %d layers among %d", shared, cfg.Layers)
	}
	// The reuse rule below picks the last global layer and the last
	// local one that still keep a cache, which only works if those are
	// the two layers immediately before the boundary.
	if cfg.SWAPattern[cfg.KVFromStart-1] || !cfg.SWAPattern[cfg.KVFromStart-2] {
		return fmt.Errorf("gemma4 layer %d is not the last global layer with a KV cache", cfg.KVFromStart-1)
	}
	if cfg.HeadDimSWA == 0 || cfg.RopeThetaSWA == 0 || cfg.PLEDim == 0 {
		return fmt.Errorf("gemma4 is missing its local-attention or per-layer embedding dimensions")
	}
	return nil
}

func mustInt(g *gguf.File, key string) int64 {
	n, _ := g.Int(key)
	return n
}

// pleTable is Gemma 4's per-layer embedding table, one row per token of
// the vocabulary and one slice of that row per layer. It stays on disk:
// 2.3 billion of E2B's 4.6 billion parameters sit in this one tensor,
// which no decode step needs more than a single row of.
type pleTable struct {
	f     *gguf.File
	name  string
	width int     // layers * per-layer dimension
	scale float32 // sqrt(per-layer dimension), Gemma's embedding scale
}

// newPLETable opens its own handle on the file so the table outlives the
// load, which unmaps everything else once the weights are repacked.
func newPLETable(path string, cfg config) (*pleTable, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	const name = "per_layer_token_embd.weight"
	_, shape, ok := g.Info(name)
	width := cfg.Layers * cfg.PLEDim
	if !ok || len(shape) != 2 || shape[1] != width {
		g.Close()
		return nil, fmt.Errorf("gguf: %s does not hold %d layers of %d", name, cfg.Layers, cfg.PLEDim)
	}
	return &pleTable{
		f:     g,
		name:  name,
		width: width,
		scale: float32(math.Sqrt(float64(cfg.PLEDim))),
	}, nil
}

// row writes one token's whole per-layer embedding into dst.
func (p *pleTable) row(token int, dst []float32) error {
	t, err := p.f.TensorRows(p.name, token, token+1)
	if err != nil {
		return err
	}
	for i, v := range t.Data {
		dst[i] = v * p.scale
	}
	return nil
}

// pleInputs builds the per-layer inputs for a batch of tokens: each
// token's table row, plus a projection of its (already scaled) token
// embedding through the same space, the two averaged. Row t of the
// result holds all the layers' slices for tokens[t], layer by layer.
func (m *qwen) pleInputs(tokens []int, embed *tensai.Matrix) (*tensai.Matrix, error) {
	cfg := m.cfg
	out := mmb(embed, m.wPleIn, m.qPleIn, nil)
	projScale := float32(1 / math.Sqrt(float64(cfg.HiddenSize)))
	const inputScale = 0.7071067811865476 // 1/sqrt(2)
	dim := cfg.PLEDim
	table := make([]float32, m.ple.width)
	for t, tok := range tokens {
		if err := m.ple.row(tok, table); err != nil {
			return nil, err
		}
		row := out.Data[t*m.ple.width : (t+1)*m.ple.width]
		for i := range row {
			row[i] *= projScale
		}
		for l := 0; l < cfg.Layers; l++ {
			sl := row[l*dim : (l+1)*dim]
			rmsnormInto(sl, sl, m.pleNorm, cfg.RMSEps)
		}
		for i := range row {
			row[i] = (row[i] + table[i]) * inputScale
		}
	}
	return out, nil
}

// pleBlock is the second residual at the end of a Gemma 4 block: a gelu
// gate on the block's own output selects from this layer's slice of the
// per-layer embedding, and the result projects back to the hidden width.
// gate and proj are scratch of PLEDim and HiddenSize floats.
func (m *qwen) pleBlock(b *qblock, x, pe, gate, proj []float32) {
	mvInto(gate, x, b.wPleGate, b.qPleGate, nil)
	gelu(gate)
	for i := range gate {
		gate[i] *= pe[i]
	}
	mvInto(proj, gate, b.wPleProj, b.qPleProj, nil)
	rmsnormInto(proj, proj, b.plePost, m.cfg.RMSEps)
	for i := range x {
		x[i] += proj[i]
	}
}

// gelu applies the tanh-approximated gelu in place, the same one Gemma's
// feed-forward gate uses.
func gelu(v []float32) {
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i, g64 := range v {
		g := float64(g64)
		v[i] = float32(0.5 * g * (1 + math.Tanh(c*(g+0.044715*g*g*g))))
	}
}

// capLogits squashes the final logits through Gemma's tanh soft cap.
func (m *qwen) capLogits(logits []float32) []float32 {
	cap := float32(m.cfg.LogitCap)
	if cap == 0 {
		return logits
	}
	for i, v := range logits {
		logits[i] = float32(math.Tanh(float64(v/cap))) * cap
	}
	return logits
}

// pleBatch is pleBlock over a whole batch: the gate and the projection
// are ordinary matmuls, and only the elementwise selection and the norm
// run per row.
func (m *qwen) pleBatch(b *qblock, li int, x, pe *tensai.Matrix) {
	dim, hs := m.cfg.PLEDim, m.cfg.HiddenSize
	width := m.ple.width
	gate := mmb(x, b.wPleGate, b.qPleGate, nil)
	gelu(gate.Data)
	for t := 0; t < x.Rows; t++ {
		row := gate.Data[t*dim : (t+1)*dim]
		sl := pe.Data[t*width+li*dim : t*width+(li+1)*dim]
		for i := range row {
			row[i] *= sl[i]
		}
	}
	proj := mmb(gate, b.wPleProj, b.qPleProj, nil)
	for t := 0; t < x.Rows; t++ {
		r := proj.Data[t*hs : (t+1)*hs]
		rmsnormInto(r, r, b.plePost, m.cfg.RMSEps)
	}
	for i := range x.Data {
		x.Data[i] += proj.Data[i]
	}
}

package llm

import (
	"path/filepath"
	"testing"
)

// gemma4TestConfig is Gemma 4 E2B's geometry as the GGUF states it:
// 35 layers, every fifth attending globally with the wide head, the
// feed-forward width doubling at layer 15, which is also where the KV
// cache stops being per layer.
func gemma4TestConfig() config {
	cfg := config{
		ModelType: "gemma4", Layers: 35, Heads: 8, KVHeads: 1,
		HiddenSize: 1536, HeadDim: 512, HeadDimSWA: 256,
		SlidingWin: 512, RopeTheta: 1e6, RopeThetaSWA: 10000,
		KVFromStart: 15, PLEDim: 256, LogitCap: 30,
	}
	for i := 0; i < cfg.Layers; i++ {
		ff := 6144
		if i >= 15 {
			ff = 12288
		}
		cfg.FFPerLayer = append(cfg.FFPerLayer, ff)
		cfg.SWAPattern = append(cfg.SWAPattern, (i+1)%5 != 0)
	}
	return cfg
}

func TestGemma4BlockShape(t *testing.T) {
	cfg := gemma4TestConfig()
	blocks := make([]qblock, cfg.Layers)
	for i := range blocks {
		blockShape(&blocks[i], cfg, i)
	}
	for _, tt := range []struct {
		layer  int
		head   int
		window int
		theta  float64
		ff     int
		kvFrom int // -1 for a layer that keeps its own
	}{
		{0, 256, 512, 10000, 6144, -1},   // local, its own cache
		{4, 512, 0, 0, 6144, -1},         // global, its own cache
		{14, 512, 0, 0, 6144, -1},        // the last layer with a cache
		{15, 256, 512, 10000, 12288, 13}, // local, against the last local cache
		{19, 512, 0, 0, 12288, 14},       // global, against the last global one
		{34, 512, 0, 0, 12288, 14},
	} {
		b := &blocks[tt.layer]
		if b.headSz != tt.head || b.window != tt.window || b.ropeTheta != tt.theta {
			t.Errorf("layer %d: head %d window %d theta %v, want %d %d %v",
				tt.layer, b.headSz, b.window, b.ropeTheta, tt.head, tt.window, tt.theta)
		}
		kvFrom := -1
		if b.kvShared {
			kvFrom = b.kvFrom
		}
		if b.ff != tt.ff || kvFrom != tt.kvFrom {
			t.Errorf("layer %d: ff %d kvFrom %d, want %d %d", tt.layer, b.ff, kvFrom, tt.ff, tt.kvFrom)
		}
		if !b.geglu || !b.vNorm || !b.unitQK {
			t.Errorf("layer %d: geglu %v vNorm %v unitQK %v", tt.layer, b.geglu, b.vNorm, b.unitQK)
		}
		// A layer reusing a cache must find the same geometry there.
		if b.kvShared && blocks[b.kvFrom].headSz != b.headSz {
			t.Errorf("layer %d reuses layer %d, whose head is %d not %d",
				tt.layer, b.kvFrom, blocks[b.kvFrom].headSz, b.headSz)
		}
	}
	// Nothing else in the loader may pick up a shared cache by accident.
	for _, arch := range []string{"llama", "qwen3", "gemma3", "gpt-oss"} {
		cfg := config{ModelType: arch, Layers: 4, SlidingWin: 128}
		var b qblock
		blockShape(&b, cfg, 1)
		if b.kvShared || b.ff != 0 || b.unitQK {
			t.Errorf("%s: kvShared %v ff %d unitQK %v", arch, b.kvShared, b.ff, b.unitQK)
		}
	}
}

// The per-layer rope frequencies: the local layers turn their whole
// 256-wide head at the local base, and the global ones turn only the
// first eighth of a 512-wide head, the frequency factors flattening the
// rest to nothing.
func TestGemma4RopeFreqs(t *testing.T) {
	cfg := gemma4TestConfig()
	m := &qwen{cfg: cfg, headSz: cfg.HeadDim, blocks: make([]qblock, cfg.Layers)}
	ff := make([]float32, cfg.HeadDim/2)
	for i := range ff {
		ff[i] = 1
		if i >= 64 {
			ff[i] = 1e30
		}
	}
	for i := range m.blocks {
		blockShape(&m.blocks[i], cfg, i)
		if !cfg.SWAPattern[i] {
			m.blocks[i].ropeFF = ff
		}
	}
	m.initRopeFreqs()
	if got := len(m.blocks[0].ropeFreq); got != 128 {
		t.Fatalf("local layer has %d frequencies, want 128", got)
	}
	global := m.blocks[4].ropeFreq
	if len(global) != 256 {
		t.Fatalf("global layer has %d frequencies, want 256", len(global))
	}
	if global[0] != 1 {
		t.Errorf("global frequency 0 = %v, want 1", global[0])
	}
	if global[63] <= 0 || global[64] > 1e-25 {
		t.Errorf("the rotation should stop after pair 63: %v then %v", global[63], global[64])
	}
	// The local layers rotate the whole head at the local base.
	if m.blocks[0].ropeFreq[0] != 1 || m.blocks[0].ropeFreq[127] >= 1e-3 {
		t.Errorf("local frequencies run %v .. %v", m.blocks[0].ropeFreq[0], m.blocks[0].ropeFreq[127])
	}
}

func TestCapLogits(t *testing.T) {
	m := &qwen{cfg: config{LogitCap: 30}}
	logits := []float32{0, 30, 1000, -1000}
	m.capLogits(logits)
	if logits[0] != 0 {
		t.Errorf("0 capped to %v", logits[0])
	}
	if logits[2] <= 29.9 || logits[2] > 30 || logits[3] >= -29.9 || logits[3] < -30 {
		t.Errorf("large logits capped to %v and %v, want +/-30", logits[2], logits[3])
	}
	// Without a cap the logits are left alone.
	plain := &qwen{cfg: config{}}
	logits = []float32{1000}
	plain.capLogits(logits)
	if logits[0] != 1000 {
		t.Errorf("uncapped logit changed to %v", logits[0])
	}
}

// -gpu on a model the device path cannot run has to be caught before the
// load, because it changes how the weights are repacked: the wrong
// answer costs a few gigabytes of pointless requantization. A file that
// is not there answers nothing, which leaves the load to report it.
func TestGPUCannotRun(t *testing.T) {
	if why := gpuCannotRun(filepath.Join(t.TempDir(), "absent.gguf")); why != "" {
		t.Errorf("a missing file should not decide the question: %q", why)
	}
}

// E4B's geometry, which differs from E2B's in every way the loader has
// to read rather than assume: 42 layers on a six-layer window cycle,
// two KV heads, one feed-forward width for the whole model, and the KV
// boundary in a different place.
func TestGemma4E4BShape(t *testing.T) {
	cfg := config{
		ModelType: "gemma4", Layers: 42, Heads: 8, KVHeads: 2,
		HiddenSize: 2560, HeadDim: 512, HeadDimSWA: 256,
		SlidingWin: 512, RopeTheta: 1e6, RopeThetaSWA: 10000,
		KVFromStart: 42 - 18, PLEDim: 256, LogitCap: 30, Intermediate: 10240,
	}
	for i := 0; i < cfg.Layers; i++ {
		cfg.FFPerLayer = append(cfg.FFPerLayer, 10240)
		cfg.SWAPattern = append(cfg.SWAPattern, (i+1)%6 != 0)
	}
	blocks := make([]qblock, cfg.Layers)
	for i := range blocks {
		blockShape(&blocks[i], cfg, i)
	}
	// The layer the reuse rule picks has to be the same kind, which is
	// what makes the shared cache the right shape.
	for i, b := range blocks {
		if (i < cfg.KVFromStart) == b.kvShared {
			t.Fatalf("layer %d: kvShared %v against a boundary at %d", i, b.kvShared, cfg.KVFromStart)
		}
		if b.kvShared && blocks[b.kvFrom].headSz != b.headSz {
			t.Errorf("layer %d (head %d) reuses layer %d (head %d)",
				i, b.headSz, b.kvFrom, blocks[b.kvFrom].headSz)
		}
		if b.ff != 10240 {
			t.Errorf("layer %d: ff %d", i, b.ff)
		}
	}
	if blocks[23].headSz != 512 || blocks[24].headSz != 256 {
		t.Errorf("the window cycle is off: layer 23 head %d, layer 24 head %d",
			blocks[23].headSz, blocks[24].headSz)
	}
}

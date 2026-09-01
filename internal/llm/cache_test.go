package llm

import (
	"os"
	"path/filepath"
	"testing"

	tensai "github.com/mattn/tensai"
)

// Every weight slot the model can hold has to survive the round trip, or
// a cached load quietly answers with a model missing a layer's worth of
// something. gemma4's per-layer embedding weights are the newest slots
// and the easiest to forget.
func TestWeightCacheRoundTrip(t *testing.T) {
	cfg := gemma4TestConfig()
	cfg.Layers = 2
	cfg.FFPerLayer, cfg.SWAPattern = cfg.FFPerLayer[:2], cfg.SWAPattern[:2]
	cfg.KVFromStart = 2
	vec := func(n int, v float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = v + float32(i)
		}
		return out
	}
	mat := func(r, c int, v float32) *tensai.Matrix {
		m := tensai.NewMatrix(r, c)
		for i := range m.Data {
			m.Data[i] = v + float32(i)
		}
		return m
	}
	src := &qwen{cfg: cfg, headSz: cfg.HeadDim}
	src.embed = tensai.NewTensor(3, 4)
	copy(src.embed.Data, vec(12, 1))
	src.normW = vec(cfg.HiddenSize, 2)
	src.pleNorm = vec(cfg.PLEDim, 3)
	src.wPleIn = mat(2, 3, 4)
	src.blocks = make([]qblock, cfg.Layers)
	for i := range src.blocks {
		blockShape(&src.blocks[i], cfg, i)
		b := &src.blocks[i]
		b.ln1, b.ln2 = vec(4, float32(i)), vec(4, float32(i)+1)
		b.plePost = vec(5, float32(i)+2)
		b.outScale = []float32{0.5 + float32(i)}
		b.ropeFF = vec(b.headSz/2, float32(i)+3)
		b.wPleGate = mat(2, 2, float32(i))
		b.wPleProj = mat(2, 3, float32(i)+1)
		b.wQKV = mat(2, 4, float32(i)+2)
	}

	dir := t.TempDir()
	gguf := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(gguf, []byte("not really a gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	cpath := cachePath(gguf, 4, true)
	if err := writeWeightCache(cpath, gguf, 4, true, src); err != nil {
		t.Fatal(err)
	}
	// The reader reopens the per-layer embedding table from the source
	// file, which this fake has none of, so read it back without one.
	cfg.PLEDim = 0
	got, err := loadWeightCache(cpath, gguf, 4, true, cfg, cfg.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	same := func(name string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Errorf("%s: %d values back, want %d", name, len(b), len(a))
			return
		}
		for i := range a {
			if a[i] != b[i] {
				t.Errorf("%s[%d] = %v, want %v", name, i, b[i], a[i])
				return
			}
		}
	}
	same("embed", src.embed.Data, got.embed.Data)
	same("normW", src.normW, got.normW)
	same("pleNorm", src.pleNorm, got.pleNorm)
	same("wPleIn", src.wPleIn.Data, got.wPleIn.Data)
	for i := range src.blocks {
		a, b := &src.blocks[i], &got.blocks[i]
		same("ln1", a.ln1, b.ln1)
		same("plePost", a.plePost, b.plePost)
		same("outScale", a.outScale, b.outScale)
		same("ropeFF", a.ropeFF, b.ropeFF)
		same("wPleGate", a.wPleGate.Data, b.wPleGate.Data)
		same("wPleProj", a.wPleProj.Data, b.wPleProj.Data)
		same("wQKV", a.wQKV.Data, b.wQKV.Data)
		// The geometry comes from the config, not from the cache.
		if b.ff != a.ff || b.headSz != a.headSz || b.kvShared != a.kvShared || !b.unitQK {
			t.Errorf("layer %d geometry: ff %d head %d kvShared %v unitQK %v",
				i, b.ff, b.headSz, b.kvShared, b.unitQK)
		}
	}
}

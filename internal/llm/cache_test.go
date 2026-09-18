package llm

import (
	"os"
	"path/filepath"
	"testing"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/quant"
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
	got, err := loadWeightCache(cpath, gguf, 4, true, cfg, cfg.HeadDim, nil)
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

// A delta layer's weights, a ternary matrix, and the rotation a weight's
// input takes all have to come back from the cache: the rotation is
// rebuilt from the source's declaration, so what is checked is that the
// weight answers an input the same way before and after.
func TestWeightCacheDeltaTernary(t *testing.T) {
	cfg := config{
		ModelType: "qwen3_5", Layers: 2, HiddenSize: 8, Heads: 2, KVHeads: 1, HeadDim: 4,
		LayerTypes:     []string{"linear_attention", "full_attention"},
		LinearKeyHeads: 2, LinearValueHeads: 4, LinearKeyDim: 4, LinearValueDim: 2, LinearConvK: 4,
	}
	hspec := &hadamardSpec{
		block:    8,
		signs:    map[int][]float32{8: {1, -1, 1, 1, -1, 1, -1, -1}},
		weights:  map[string]bool{"x": true},
		inverses: map[string]bool{},
		vGrouped: true,
	}
	tern := func(rows, cols int, seed int) *qmat {
		q := quant.NewTernaryMatrix(rows, cols)
		for j := 0; j < cols; j++ {
			for i := 0; i < rows; i++ {
				q.Set(i, j, int8((i*3+j+seed)%3)-1)
			}
			q.Scale[q.TableIndex(0, j)] = 0.5 + float32(j)/8
		}
		return qmatT(q)
	}
	src := &qwen{cfg: cfg, headSz: cfg.HeadDim}
	src.normW = []float32{1, 2, 3, 4, 5, 6, 7, 8}
	src.qLmT = tern(8, 16, 1)
	h, _ := hspec.forWidth(8)
	src.qLmT.rotate(h, rotPlain)
	src.blocks = make([]qblock, 2)
	for i := range src.blocks {
		blockShape(&src.blocks[i], cfg, i)
	}
	d := newDeltaHeader(cfg)
	d.conv = make([]float32, d.convDim*d.convK)
	d.aLog, d.dtBias, d.norm = []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, []float32{9, 10}
	d.qQZ = tern(8, d.convDim+d.vDim*d.heads, 2)
	d.qQZ.rotate(h, rotPlain)
	d.qOut = tern(8, 8, 3)
	hg, _ := hspec.forWidth(8)
	hg.perm = tiledToGrouped(2, 2, 2)
	d.qOut.rotate(hg, rotGrouped)
	d.wAB = tensai.NewMatrix(8, 8)
	src.blocks[0].delta = d
	src.blocks[1].qQKV = tern(8, 16, 4)

	dir := t.TempDir()
	gguf := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(gguf, []byte("not really a gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	cpath := cachePath(gguf, 0, true)
	if err := writeWeightCache(cpath, gguf, 0, true, src); err != nil {
		t.Fatal(err)
	}
	// Without the source's declaration a rotated cache is refused.
	if _, err := loadWeightCache(cpath, gguf, 0, true, cfg, cfg.HeadDim, nil); err == nil {
		t.Fatal("a rotated cache loaded without a rotation to rebuild")
	}
	// The embedding table is reopened from the source, which this fake
	// is not; read past that by giving the fake an expanded one.
	src.embed = tensai.NewTensor(2, 8)
	if err := writeWeightCache(cpath, gguf, 0, true, src); err != nil {
		t.Fatal(err)
	}
	got, err := loadWeightCache(cpath, gguf, 0, true, cfg, cfg.HeadDim, hspec)
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{0.5, -1, 2, 0.25, -0.75, 1.5, -2, 1}
	agree := func(name string, a, b *qmat) {
		t.Helper()
		if (a == nil) != (b == nil) {
			t.Fatalf("%s: %v back, want %v", name, b != nil, a != nil)
		}
		if a == nil {
			return
		}
		if a.rot != b.rot {
			t.Fatalf("%s: rotation %d back, want %d", name, b.rot, a.rot)
		}
		oa, ob := make([]float32, a.cols), make([]float32, b.cols)
		a.f(x, oa)
		b.f(x, ob)
		for i := range oa {
			if oa[i] != ob[i] {
				t.Fatalf("%s[%d] = %v back, want %v", name, i, ob[i], oa[i])
			}
		}
	}
	agree("lm head", src.qLmT, got.qLmT)
	gd := got.blocks[0].delta
	if gd == nil || gd.heads != 4 || gd.kHeads != 2 || !gd.tiled {
		t.Fatalf("delta layer back as %+v", gd)
	}
	agree("qQZ", d.qQZ, gd.qQZ)
	agree("qOut", d.qOut, gd.qOut)
	agree("qQKV", src.blocks[1].qQKV, got.blocks[1].qQKV)
	if got.blocks[1].delta != nil {
		t.Fatal("the attention layer came back with a delta layer")
	}
	for i, v := range d.dtBias {
		if gd.dtBias[i] != v {
			t.Fatalf("dtBias[%d] = %v, want %v", i, gd.dtBias[i], v)
		}
	}
}

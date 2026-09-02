package llm

import (
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The invariants every checkpoint on this machine has to satisfy, run
// against whatever is in the model cache. Two loaders shape a model very
// differently -- one reads a gguf, the other a safetensors directory --
// and a change that suits one has broken the other more than once, in
// ways no unit test with a hand-built model could see. Skips when the
// cache is empty, which is why CI never runs it; `make check-models`
// runs the whole cache instead of the small end of it.
func TestModelInvariants(t *testing.T) {
	root := CacheRoot()
	ggufs, dirs := findModels(t, root)
	if len(ggufs)+len(dirs) == 0 {
		t.Skipf("no models under %s", root)
	}
	all := os.Getenv("TENSAI_ALL_MODELS") != ""
	// A pass loads the model twice, so the default run stays on the
	// small end of the cache and the make target takes the rest.
	const smallEnough = 1500 << 20

	for _, path := range ggufs {
		st, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if !all && st.Size() > smallEnough {
			continue
		}
		t.Run("gguf/"+filepath.Base(path), func(t *testing.T) {
			m, _, err := loadGGUF(path, 4, true, false, io.Discard)
			if err != nil {
				t.Skip(err)
			}
			checkPrefillMatchesDecode(t, m, 4)
			// A float32 load has to stay float32. Q6_K tensors used to
			// take the direct int4 repack whatever the caller asked for,
			// which is only visible from the weights themselves.
			if st.Size() < 700<<20 || all {
				plain, _, err := loadGGUF(path, 0, true, false, io.Discard)
				if err != nil {
					t.Fatalf("float32 load: %v", err)
				}
				for i := range plain.blocks {
					b := &plain.blocks[i]
					if b.qQKV != nil || b.qo != nil || b.qGU != nil || b.qDown != nil {
						t.Fatalf("layer %d holds quantized weights after a float32 load", i)
					}
				}
				checkPrefillMatchesDecode(t, plain, 0)
			}
			// The repack cache stores the repacked bytes, so a cached
			// load holds the same weights and must answer identically.
			cached, _, err := loadGGUF(path, 4, true, true, io.Discard)
			if err != nil {
				t.Fatalf("cached load: %v", err)
			}
			a, b := logitsFor(m, []int{1, 2, 3}), logitsFor(cached, []int{1, 2, 3})
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("logit %d differs with the repack cache: %v vs %v", i, a[i], b[i])
				}
			}
		})
	}
	for _, dir := range dirs {
		t.Run("safetensors/"+filepath.Base(dir), func(t *testing.T) {
			weights := filepath.Join(dir, "model.safetensors")
			if !exists(weights) {
				weights = filepath.Join(dir, "model.safetensors.index.json")
				if !exists(weights) {
					t.Skip("no weights in " + dir)
				}
			}
			if st, err := os.Stat(weights); err == nil && !all && st.Size() > smallEnough {
				t.Skip("large; TENSAI_ALL_MODELS=1 to include it")
			}
			m, err := loadQwen(filepath.Join(dir, "config.json"), weights, 4)
			if err != nil {
				t.Skip(err) // an architecture this loader does not speak
			}
			checkPrefillMatchesDecode(t, m, 4)
			if q8, err := loadQwen(filepath.Join(dir, "config.json"), weights, 8); err == nil {
				checkPrefillMatchesDecode(t, q8, 8)
			}
		})
	}
}

// checkPrefillMatchesDecode is the invariant that catches a KV cache
// that is not being written, shared with the wrong layer, or read at the
// wrong width: a batch and the same tokens one at a time have to end up
// in the same place. It is what a first message does, and what nothing
// covered when a zero-valued block claimed to share layer 0's cache.
func checkPrefillMatchesDecode(t *testing.T, m *qwen, bits int) {
	t.Helper()
	tokens := []int{1, 2, 3, 4, 5}
	batch := logitsFor(m, tokens)

	m.reset()
	var one []float32
	for i, tok := range tokens {
		one = m.step(tok, i)
	}
	if len(batch) != len(one) {
		t.Fatalf("%d logits prefilled, %d decoded", len(batch), len(one))
	}
	if argmax(batch) != argmax(one) {
		t.Fatalf("prefill picks token %d, decode picks %d", argmax(batch), argmax(one))
	}
	var worst float64
	for i := range batch {
		if d := math.Abs(float64(batch[i] - one[i])); d > worst {
			worst = d
		}
	}
	// In float32 and int8 the two paths are the same arithmetic in a
	// different order and agree to rounding. int4 quantizes the
	// activations, and the batch and the single row round differently --
	// measured against a float32 run, neither is the better of the two,
	// so the bar there is only that they still answer the same token.
	limit := 0.001
	if bits == 4 {
		limit = 0.25
	}
	if span := logitSpan(batch); worst > limit*span {
		t.Errorf("int%d prefill and decode differ by %v, %.1f%% of the logit span (limit %.1f%%)",
			bits, worst, 100*worst/span, 100*limit)
	}
}

func logitsFor(m *qwen, tokens []int) []float32 {
	m.reset()
	return m.prefill(tokens, 0)
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func logitSpan(v []float32) float64 {
	lo, hi := v[0], v[0]
	for _, x := range v {
		lo, hi = min(lo, x), max(hi, x)
	}
	return float64(hi - lo)
}

// findModels lists the gguf files and the checkpoint directories in the
// cache, newest layout first: a directory counts when it holds a
// config.json, which is what the safetensors loader needs.
func findModels(t *testing.T, root string) (ggufs, dirs []string) {
	t.Helper()
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, nil
	}
	for _, e := range entries {
		p := filepath.Join(root, e.Name())
		switch {
		case e.IsDir():
			if exists(filepath.Join(p, "config.json")) {
				dirs = append(dirs, p)
			}
		case strings.HasSuffix(e.Name(), ".gguf"):
			ggufs = append(ggufs, p)
		}
	}
	sort.Strings(ggufs)
	sort.Strings(dirs)
	fmt.Printf("%d gguf files and %d checkpoints under %s\n", len(ggufs), len(dirs), root)
	return ggufs, dirs
}

package qwenimage

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/tokenizer"
)

func checkpointDir(t *testing.T) ModelDir {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	dir := ModelDir(filepath.Join(home, ".cache", "tensai", "Qwen-Image-2.1"))
	if _, err := os.Stat(filepath.Join(dir.TextEncoder(), "model.safetensors.index.json")); err != nil {
		t.Skipf("no text encoder under %s", dir.TextEncoder())
	}
	return dir
}

// TestPromptTokens checks that the template tokenizes the way the
// reference's processor does, down to the preamble the pipeline drops.
func TestPromptTokens(t *testing.T) {
	dir := checkpointDir(t)
	tk, err := tokenizer.Load(dir.Tokenizer())
	if err != nil {
		t.Fatal(err)
	}
	ids := PromptTokens(tk, "a red cube on a white table")
	want := readI32(t, filepath.Join("testdata", "text_ids_4.i32"), len(ids))
	for i, w := range want {
		if ids[i] != w {
			t.Fatalf("token %d is %d, reference %d", i, ids[i], w)
		}
	}
	if got := len(tk.Encode(promptPreamble)); got != PromptDrop {
		t.Errorf("the preamble is %d tokens, PromptDrop says %d", got, PromptDrop)
	}
}

// TestTextEncoderMatchesReference runs the prompt encoder over the
// tokens testdata/text.py handed to transformers. The reference is cut
// to a few layers so it fits in memory; every layer is the same, so
// what this pins is the wiring — the embedding, the norms on each
// head's queries and keys, the rotation, grouped-query attention and
// the feed-forward.
func TestTextEncoderMatchesReference(t *testing.T) {
	dir := checkpointDir(t)
	const layers = 4
	tag := fmt.Sprintf("%d", layers)
	ids := readI32(t, filepath.Join("testdata", "text_ids_"+tag+".i32"), 29)
	want := readF32(t, filepath.Join("testdata", "text_hidden_"+tag+".f32"), len(ids)*teDim)

	e, err := loadTextEncoder(dir.TextEncoder(), 0, layers)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	got, err := e.Encode(ids)
	if err != nil {
		t.Fatal(err)
	}
	var worst float64
	var at int
	for i, v := range want {
		if d := math.Abs(float64(got.Data[i] - v)); d > worst {
			worst, at = d, i
		}
	}
	t.Logf("largest difference %.4g at element %d (rms %.4f)", worst, at, rmsOf(want))
	if worst > 2e-3 {
		t.Errorf("element %d is %g, reference %g", at, got.Data[at], want[at])
	}
}

func rmsOf(v []tensai.Float) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s / float64(len(v)))
}

package qwenaudio

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// checkpoint is where the tests find Qwen2-Audio's weights:
// QWENAUDIO_MODEL, or the cache a -model Qwen/Qwen2-Audio-7B-Instruct
// download fills.
func checkpoint(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("QWENAUDIO_MODEL")
	if dir == "" {
		home, _ := os.UserCacheDir()
		dir = filepath.Join(home, "tensai", "Qwen", "Qwen2-Audio-7B-Instruct")
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(index); err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	return index
}

func TestEncoderMatchesReference(t *testing.T) {
	const layers = 2 // what testdata/ref.py was run with
	want := readF32(t, "audio_2.f32")
	samples, err := ReadWAV(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		t.Skip(err)
	}
	e, err := LoadEncoder(checkpoint(t), layers)
	if err != nil {
		t.Fatal(err)
	}
	got := e.Encode(LogMel(samples))
	if got.Rows*got.Cols != len(want) || got.Cols != lmDim {
		t.Fatalf("encoded %dx%d, want %d values", got.Rows, got.Cols, len(want))
	}
	var num, den float64
	for i, w := range want {
		d := float64(got.Data[i]) - float64(w)
		num += d * d
		den += float64(w) * float64(w)
	}
	if rel := math.Sqrt(num / den); rel > 1e-4 {
		t.Errorf("relative error %g (max abs %g)", rel, maxDiff(got.Data, want))
	} else {
		t.Logf("relative error %g", rel)
	}
}

package qwenimage

import (
	"encoding/binary"
	"fmt"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/mattn/tensai"
)

func readF64(t *testing.T, path string, want int) []float64 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference dump: %v", err)
	}
	if len(b) != 8*want {
		t.Fatalf("%s: %d bytes, want %d values", path, len(b), want)
	}
	out := make([]float64, want)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return out
}

// TestScheduleMatchesReference checks the noise levels against what the
// checkpoint's own scheduler walks. Needs no weights.
func TestScheduleMatchesReference(t *testing.T) {
	const (
		steps  = 20
		tokens = 1024
	)
	want := readF64(t, filepath.Join("testdata", fmt.Sprintf("sigmas_%d_%d.f64", steps, tokens)), steps+1)
	got := NewSchedule(steps, tokens)
	var worst float64
	for i, w := range want {
		worst = math.Max(worst, math.Abs(got.Sigmas[i]-w))
	}
	t.Logf("largest difference %.3g; first %.5f last step %.5f", worst, got.Sigmas[0], got.Sigmas[steps-1])
	if worst > 1e-6 {
		t.Errorf("noise levels differ by %g", worst)
	}
}

// TestGenerateWritesPNG runs the whole pipeline at full depth and
// writes what it lands on. The prompt's hidden states are left at zero
// because the text encoder is not here yet, so the picture means
// nothing; what this checks is that 32 quantized blocks, the schedule
// and the decoder hold together at full size, and what they cost.
func TestGenerateWritesPNG(t *testing.T) {
	out := os.Getenv("TENSAI_IMAGE_PNG")
	if out == "" {
		t.Skip("set TENSAI_IMAGE_PNG to a path to write a generated image")
	}
	dir := ModelDir(filepath.Join(os.Getenv("HOME"), ".cache", "tensai", "Qwen-Image-2.1"))
	if _, err := os.Stat(dir.VAE()); err != nil {
		t.Skipf("no checkpoint under %s", string(dir))
	}
	steps := envInt("TENSAI_IMAGE_STEPS", 20)
	side := envInt("TENSAI_IMAGE_LATENT", 16) // 16 latent rows is a 256x256 image

	start := time.Now()
	m, err := LoadTransformer(dir.Transformer(), 8)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("transformer loaded in %v", time.Since(start).Round(time.Second))

	const textLen = 16
	l := NewLayout(textLen, side, side)
	text := tensai.NewMatrix(textLen, ditDim)
	latents := Noise(rand.New(rand.NewPCG(5, 0)), side, side)
	sched := NewSchedule(steps, side*side)

	start = time.Now()
	err = Generate(m, latents, text, l, sched, func(i int) {
		t.Logf("step %d/%d at %v", i+1, steps, time.Since(start).Round(time.Second))
	})
	if err != nil {
		t.Fatal(err)
	}
	per := time.Since(start) / time.Duration(steps)
	t.Logf("%d steps of %dx%d in %v (%v a step)", steps, side*16, side*16, time.Since(start).Round(time.Second), per.Round(time.Millisecond))

	for i, v := range latents.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("latent %d is %v after %d steps", i, v, steps)
		}
	}
	stats, err := LoadStats(dir.VAEConfig())
	if err != nil {
		t.Fatal(err)
	}
	d, err := LoadDecoder(dir.VAE())
	if err != nil {
		t.Fatal(err)
	}
	px, err := Decode(d, stats.Denormalize(latents, side, side))
	if err != nil {
		t.Fatal(err)
	}
	img, err := Image(px)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

func envInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

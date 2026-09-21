package qwenimage

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/mattn/tensai"
)

// Generation is flow matching: the latent starts as pure noise and the
// transformer says, at each point along the way, which direction carries
// it towards an image. An Euler step follows that direction as far as
// the schedule's next stop, and the schedule is what decides where the
// stops are.
//
// The stops are not evenly spaced. They are bent towards the noisy end
// by an amount that grows with the image, because a larger image has
// more to decide early, and then stretched so the last step lands just
// short of zero rather than on it.

// Schedule holds the noise levels a run passes through, from 1 down to
// its terminal value, with a trailing zero so every step has a next.
type Schedule struct {
	Sigmas []float64
}

// NewSchedule builds the schedule for an image of tokens latent
// positions over the given number of steps.
func NewSchedule(steps, tokens int) *Schedule {
	// The bend grows linearly with the token count between the two
	// anchors the checkpoint's scheduler carries.
	const (
		baseLen, maxLen     = 256, 8192
		baseShift, maxShift = 0.5, 0.9
		terminal            = 0.02
	)
	m := (maxShift - baseShift) / (maxLen - baseLen)
	mu := math.Exp(float64(tokens)*m + baseShift - m*baseLen)

	s := &Schedule{Sigmas: make([]float64, steps+1)}
	for i := 0; i < steps; i++ {
		t := 1 - float64(i)*(1-1/float64(steps))/float64(steps-1)
		s.Sigmas[i] = mu / (mu + (1/t - 1))
	}
	// Stretch so the run ends at the terminal noise level instead of
	// wherever the bend left it.
	scale := (1 - s.Sigmas[steps-1]) / (1 - terminal)
	for i := 0; i < steps; i++ {
		s.Sigmas[i] = 1 - (1-s.Sigmas[i])/scale
	}
	return s
}

// Steps is how many denoising steps the schedule runs.
func (s *Schedule) Steps() int { return len(s.Sigmas) - 1 }

// Stats are the per-channel mean and deviation the VAE's latent space
// was normalized by; generation works in the normalized space and the
// decoder wants the raw one.
type Stats struct {
	Mean, Std []float64
}

// LoadStats reads the statistics out of a VAE config.
func LoadStats(path string) (*Stats, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Mean []float64 `json:"latents_mean"`
		Std  []float64 `json:"latents_std"`
		Dim  int       `json:"z_dim"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, err
	}
	if len(cfg.Mean) != cfg.Dim || len(cfg.Std) != cfg.Dim {
		return nil, fmt.Errorf("qwenimage: %s: %d means and %d deviations for %d channels",
			path, len(cfg.Mean), len(cfg.Std), cfg.Dim)
	}
	return &Stats{Mean: cfg.Mean, Std: cfg.Std}, nil
}

// Denormalize turns a generated latent, which is (tokens, 64) in the
// normalized space, into the (64, height, width) tensor the decoder
// reads.
func (s *Stats) Denormalize(latents *tensai.Matrix, height, width int) *tensai.Tensor {
	out := tensai.NewTensor(ditLatent, height, width)
	n := height * width
	for c := 0; c < ditLatent; c++ {
		mean, std := tensai.Float(s.Mean[c]), tensai.Float(s.Std[c])
		for p := 0; p < n; p++ {
			out.Data[c*n+p] = latents.Data[p*ditLatent+c]*std + mean
		}
	}
	return out
}

// Noise draws a starting latent: the schedule begins at a noise level of
// one, so the first sample is noise and nothing else.
func Noise(rng *rand.Rand, height, width int) *tensai.Matrix {
	m := tensai.NewMatrix(height*width, ditLatent)
	for i := range m.Data {
		m.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return m
}

// Generate runs the denoising loop and returns the latent it lands on.
// progress, when set, is called after each step with its index.
func Generate(m *Transformer, latents, text *tensai.Matrix, l *Layout, s *Schedule, progress func(int)) error {
	scratch := NewScratch(l.Tokens())
	for i := 0; i < s.Steps(); i++ {
		v, err := m.Velocity(latents, text, s.Sigmas[i], l, scratch)
		if err != nil {
			return err
		}
		dt := tensai.Float(s.Sigmas[i+1] - s.Sigmas[i])
		for j, d := range v.Data {
			latents.Data[j] += dt * d
		}
		if progress != nil {
			progress(i)
		}
	}
	return nil
}

// ModelDir is where a downloaded checkpoint's components sit.
type ModelDir string

// VAE returns the path to the VAE's weights.
func (d ModelDir) VAE() string {
	return filepath.Join(string(d), "vae", "diffusion_pytorch_model.safetensors")
}

// VAEConfig returns the path to the VAE's config.
func (d ModelDir) VAEConfig() string { return filepath.Join(string(d), "vae", "config.json") }

// Transformer returns the directory holding the denoising transformer.
func (d ModelDir) Transformer() string { return filepath.Join(string(d), "transformer") }

package qwenimage

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
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

// NewTurboSchedule is the Viggle six-step schedule. It uses the same
// resolution shift as the base model, but deliberately no terminal stretch.
func NewTurboSchedule(tokens int) *Schedule {
	const slope = (0.9 - 0.5) / (8192.0 - 256.0)
	mu := math.Exp(float64(tokens)*slope + 0.5 - slope*256)
	s := &Schedule{Sigmas: []float64{1, 0.9375, 0.875, 0.75, 0.5, 0.25, 0}}
	for i := 0; i < 6; i++ {
		t := s.Sigmas[i]
		s.Sigmas[i] = mu / (mu + (1/t - 1))
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

// Guidance steers a run away from a second prompt: each step asks the
// transformer twice, once for what the prompt wants and once for what
// the other one does, and follows the difference past the first. It
// doubles what a step costs, which is why it is off unless asked for.
type Guidance struct {
	Text  *tensai.Matrix // the prompt to steer away from
	Scale float64        // how far past the wanted direction to go; 1 is no guidance
}

// On returns whether the guidance does anything.
func (g *Guidance) On() bool { return g != nil && g.Text != nil && g.Scale > 1 }

// Generate runs the denoising loop and returns the latent it lands on.
// progress, when set, is called after each step with its index.
func Generate(m *Transformer, latents, text *tensai.Matrix, l *Layout, s *Schedule, g *Guidance, progress func(int)) error {
	tokens := l.Tokens()
	var away *Layout
	if g.On() {
		// The two prompts rarely tokenize to the same length, so the
		// second gets its own layout and the buffers fit the longer.
		away = NewLayout(g.Text.Rows, l.Height, l.Width)
		tokens = max(tokens, away.Tokens())
	}
	scratch := NewScratch(tokens)
	for i := 0; i < s.Steps(); i++ {
		v, err := m.Velocity(latents, text, s.Sigmas[i], l, scratch)
		if err != nil {
			return err
		}
		if away != nil {
			u, err := m.Velocity(latents, g.Text, s.Sigmas[i], away, scratch)
			if err != nil {
				return err
			}
			scale := tensai.Float(g.Scale)
			kernels.SubSlices(v.Data, v.Data, u.Data)
			kernels.ScaleSlice(v.Data, scale)
			kernels.AddSlice(v.Data, u.Data)
		}
		dt := tensai.Float(s.Sigmas[i+1] - s.Sigmas[i])
		kernels.Axpy(dt, v.Data, latents.Data)
		if progress != nil {
			progress(i)
		}
	}
	return nil
}

// ModelDir is where a downloaded checkpoint's components sit. Two
// layouts are understood: the diffusers one, a directory per component,
// and ComfyUI's, a file per component under diffusion_models,
// text_encoders and vae (with the diffusers vae/config.json and
// processor/tokenizer.json beside them, which ComfyUI does not ship).
type ModelDir string

// Comfy reports whether the directory holds ComfyUI's layout.
func (d ModelDir) Comfy() bool {
	fi, err := os.Stat(filepath.Join(string(d), "diffusion_models"))
	return err == nil && fi.IsDir()
}

// comfyFile picks a component's file out of a ComfyUI directory: the
// int8 one when it is there, else the bfloat16 one. Either path is a
// name the loaders reject clearly when the file is missing.
func (d ModelDir) comfyFile(sub, stem string) string {
	int8 := filepath.Join(string(d), sub, stem+"_int8_convrot.safetensors")
	if _, err := os.Stat(int8); err == nil {
		return int8
	}
	bf16 := filepath.Join(string(d), sub, stem+"_bf16.safetensors")
	if _, err := os.Stat(bf16); err == nil {
		return bf16
	}
	return int8
}

// VAE returns the path to the VAE's weights.
func (d ModelDir) VAE() string {
	if d.Comfy() {
		return d.comfyFile("vae", "qwen_image_2.1_vae")
	}
	return filepath.Join(string(d), "vae", "diffusion_pytorch_model.safetensors")
}

// VAEConfig returns the path to the VAE's config.
func (d ModelDir) VAEConfig() string { return filepath.Join(string(d), "vae", "config.json") }

// Transformer returns where the denoising transformer is: a directory of
// shards, or one ComfyUI file.
func (d ModelDir) Transformer() string {
	if d.Comfy() {
		return d.comfyFile("diffusion_models", "qwen_image_2.1")
	}
	return filepath.Join(string(d), "transformer")
}

// TextEncoder returns where the prompt encoder is: a directory of
// shards, or one ComfyUI file.
func (d ModelDir) TextEncoder() string {
	if d.Comfy() {
		return d.comfyFile("text_encoders", "qwen3vl_8b")
	}
	return filepath.Join(string(d), "text_encoder")
}

// Tokenizer returns the path to the processor's tokenizer.
func (d ModelDir) Tokenizer() string {
	return filepath.Join(string(d), "processor", "tokenizer.json")
}

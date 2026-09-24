//go:build wgpu || wgpu24

package qwenimage

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
)

func TestGPUDecoderConvolution(t *testing.T) {
	g, err := gpu.Open(gpu.HighPerformance)
	if err != nil {
		t.Skip(err)
	}
	defer g.Close()
	for _, k := range []int{1, 3} {
		c := &conv{in: 7, out: 11, ksz: k, pad: k / 2, w: tensai.NewMatrix(11, k*k*7), b: make([]tensai.Float, 11)}
		for i := range c.w.Data {
			c.w.Data[i] = tensai.Float(math.Sin(float64(i))) * 0.05
		}
		for i := range c.b {
			c.b[i] = tensai.Float(i) * 0.1
		}
		x := tensai.NewMatrix(9*13, 7)
		for i := range x.Data {
			x.Data[i] = tensai.Float(math.Cos(float64(i)))
		}
		want, err := c.apply(x, 9, 13, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := c.apply(x, 9, 13, g)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got.Data {
			if math.IsNaN(float64(v)) || math.Abs(float64(v-want.Data[i])) > 1e-5 {
				t.Fatalf("kernel %d pixel %d: got %g want %g", k, i, v, want.Data[i])
			}
		}
	}
}

// Run the complete decoder, including residuals and all upsampling stages.
func TestGPUDecoderMatchesCPU(t *testing.T) {
	g, err := gpu.Open(gpu.HighPerformance)
	if err != nil {
		t.Skip(err)
	}
	defer g.Close()
	d, err := LoadDecoder(filepath.Join(vaeDir(t), "diffusion_pytorch_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	z := tensai.NewTensor(64, 4, 4)
	for i := range z.Data {
		z.Data[i] = tensai.Float(math.Sin(float64(i) * 0.13))
	}
	want, err := Decode(d, z)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decode(d, z, g)
	if err != nil {
		t.Fatal(err)
	}
	var maxError float64
	for i, v := range got.Data {
		e := math.Abs(float64(v - want.Data[i]))
		if math.IsNaN(e) {
			t.Fatal("NaN output")
		}
		maxError = max(maxError, e)
	}
	t.Logf("maximum decoder error: %g", maxError)
	if maxError > 1e-3 {
		t.Fatalf("decoder error %g", maxError)
	}
}

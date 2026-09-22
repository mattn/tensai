//go:build wgpu || wgpu24

package qwenimage

import (
	"math"
	"testing"

	"github.com/mattn/tensai"
)

// TestGPUMatchesCPU runs the same inputs through a few blocks three
// ways -- in float, with the feed-forward quantized on the host, and
// with it quantized on the device -- and asks that the device is no
// further from the float answer than the host is.
//
// Device and host are not expected to agree with each other: both
// quantize the activations of every product and they do it differently,
// which a residual stream carries forward. What matters is that neither
// drifts from what the weights actually say.
func TestGPUMatchesCPU(t *testing.T) {
	dir := transformerDir(t)
	const (
		textLen = 16
		side    = 16
		layers  = 4
	)
	l := NewLayout(textLen, side, side)
	text := tensai.NewMatrix(textLen, ditDim)
	for i := range text.Data {
		text.Data[i] = tensai.Float(math.Sin(float64(i)*0.13)) * 0.5
	}
	lat := tensai.NewMatrix(side*side, ditLatent)
	for i := range lat.Data {
		lat.Data[i] = tensai.Float(math.Cos(float64(i) * 0.07))
	}
	run := func(bits int, onGPU bool) []tensai.Float {
		m, err := loadTransformer(dir, bits, layers)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if onGPU {
			if _, _, err := UseGPU(m, 4<<30); err != nil {
				t.Skipf("no usable device: %v", err)
			}
		}
		v, err := m.Velocity(lat, text, 0.7, l, NewScratch(l.Tokens()))
		if err != nil {
			t.Fatal(err)
		}
		return append([]tensai.Float(nil), v.Data...)
	}
	exact := run(0, false)
	host := run(8, false)
	dev := run(8, true)

	rel := func(a []tensai.Float) float64 {
		var sq, ref float64
		for i := range exact {
			d := float64(a[i] - exact[i])
			sq += d * d
			ref += float64(exact[i]) * float64(exact[i])
		}
		return math.Sqrt(sq / ref)
	}
	rh, rd := rel(host), rel(dev)
	t.Logf("against float: host %.4f, device %.4f", rh, rd)
	// A wiring fault shows up as the device being much the worse of the
	// two, not as the two disagreeing.
	if rd > rh*1.5+0.01 {
		t.Errorf("the device is %.4f from float where the host is %.4f", rd, rh)
	}
}

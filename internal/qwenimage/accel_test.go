//go:build wgpu || wgpu24

package qwenimage

import (
	"fmt"
	"math"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/quant"
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

func TestGPUAttentionMatchesCPU(t *testing.T) {
	g, err := gpu.Open(gpu.HighPerformance)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer g.Close()
	b := &Block{g: g}
	for _, tc := range []struct{ text, side int }{{0, 4}, {1, 4}, {11, 4}, {33, 4}, {11, 20}, {12, 32}} {
		t.Run(fmt.Sprintf("text%d_side%d", tc.text, tc.side), func(t *testing.T) {
			l := NewLayout(tc.text, tc.side, tc.side)
			n := l.Tokens()
			q, k, v := tensai.NewMatrix(n, ditDim), tensai.NewMatrix(n, ditDim), tensai.NewMatrix(n, ditDim)
			for i := range q.Data {
				q.Data[i] = tensai.Float(math.Sin(float64(i) * 0.13))
				k.Data[i] = tensai.Float(math.Cos(float64(i) * 0.17))
				v.Data[i] = tensai.Float(math.Sin(float64(i) * 0.07))
			}
			want, got := tensai.NewMatrix(n, ditDim), tensai.NewMatrix(n, ditDim)
			s := NewScratch(n)
			check := func() {
				t.Helper()
				if err := attention(want, q, k, v, l.KeyLimit, s); err != nil {
					t.Fatal(err)
				}
				if err := b.attentionOnDevice(got, q, k, v, l, s); err != nil {
					t.Fatal(err)
				}
				for i, w := range want.Data {
					if d := math.Abs(float64(got.Data[i] - w)); math.IsNaN(d) || d > 0.002 {
						t.Fatalf("element %d: got %g want %g", i, got.Data[i], w)
					}
				}
			}
			check()
			if tc.side == 32 {
				// A later block/step must upload its own K/V, while all
				// tiles within this call share those updated values.
				for i := range k.Data {
					k.Data[i] *= -0.5
					v.Data[i] += 0.25
				}
				check()
			}
			l.KeyLimit[n-1] = 1 // nonstandard masks retain the host implementation
			check()
		})
	}
}

func TestGPUCloseReleasesWeights(t *testing.T) {
	for _, bits := range []int{8, 4} {
		t.Run(fmt.Sprint(bits), func(t *testing.T) {
			l := &linear{}
			w := tensai.NewMatrix(32, 32)
			if bits == 8 {
				l.q = quant.Quantize(w)
			} else {
				var err error
				l.q4, err = quant.Quantize4(w)
				if err != nil {
					t.Fatal(err)
				}
			}
			block := &Block{mlpProj: l, mlpGate: l, mlpOut: l}
			m := &Transformer{blocks: []*Block{block}}
			if _, _, err := UseGPU(m, 1<<20); err != nil {
				t.Skipf("no usable device: %v", err)
			}
			defer m.Close()
			releases := 0
			m.release = func() error { releases++; return nil }
			if err := m.Close(); err != nil {
				t.Fatal(err)
			}
			if err := m.Close(); err != nil {
				t.Fatal(err)
			}
			if releases != 1 || m.dev != nil || block.dev != nil || block.g != nil {
				t.Fatal("close did not clear ownership exactly once")
			}
		})
	}
}

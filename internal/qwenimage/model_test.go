package qwenimage

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
)

// TestVelocityMatchesReference runs the whole transformer over the
// inputs testdata/model.py handed to diffusers. The reference is cut to
// two blocks so it fits in memory; everything around them — the two
// projections, the timestep, the modulation, the rotary layout, the
// block-causal mask and the final norm — is what this checks, and none
// of it depends on how many blocks sit in the middle.
func TestVelocityMatchesReference(t *testing.T) {
	dir := transformerDir(t)
	const (
		layers  = 2
		textLen = 13
		height  = 6
		width   = 4
	)
	tag := fmt.Sprintf("%d_%d_%d_%d", layers, textLen, height, width)
	read := func(kind string, n int) []tensai.Float {
		return readF32(t, filepath.Join("testdata", "model_"+kind+"_"+tag+".f32"), n)
	}
	latents := &tensai.Matrix{Rows: height * width, Cols: ditLatent, Data: read("lat", height*width*ditLatent)}
	text := &tensai.Matrix{Rows: textLen, Cols: ditDim, Data: read("txt", textLen*ditDim)}
	want := read("vel", height*width*ditLatent)

	m, err := loadTransformer(dir, 0, layers)
	if err != nil {
		t.Fatal(err)
	}
	l := NewLayout(textLen, height, width)
	got, err := m.Velocity(latents, text, 0.7, l, NewScratch(l.Tokens()))
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
	t.Logf("largest difference %.4g at element %d", worst, at)
	if worst > 2e-3 {
		t.Errorf("element %d is %g, reference %g", at, got.Data[at], want[at])
	}
}

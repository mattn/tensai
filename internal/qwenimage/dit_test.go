package qwenimage

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
)

func transformerDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	dir := filepath.Join(home, ".cache", "tensai", "Qwen-Image-2.1", "transformer")
	if _, err := os.Stat(filepath.Join(dir, "diffusion_pytorch_model.safetensors.index.json")); err != nil {
		t.Skipf("no transformer checkpoint under %s", dir)
	}
	return dir
}

// TestBlockMatchesReference runs one transformer block over the inputs
// testdata/block.py handed to diffusers and compares the result. The
// rotary angles are drawn at random rather than built from the position
// scheme: a block only ever sees the angles, so this pins the block's
// own arithmetic and leaves the layout to its own test.
func TestBlockMatchesReference(t *testing.T) {
	dir := transformerDir(t)
	const seq = 48
	read := func(name string, n int) []tensai.Float {
		return readF32(t, filepath.Join("testdata", name), n)
	}
	x := &tensai.Matrix{Rows: seq, Cols: ditDim, Data: read("block_x_48.f32", seq*ditDim)}
	modRows := read("block_mod_48.f32", 2*4*ditDim)
	want := read("block_y_48.f32", seq*ditDim)
	rope := &Rope{
		Cos: &tensai.Matrix{Rows: seq, Cols: ropePairs, Data: read("block_cos_48.f32", seq*ropePairs)},
		Sin: &tensai.Matrix{Rows: seq, Cols: ropePairs, Data: read("block_sin_48.f32", seq*ropePairs)},
	}

	// The reference lays a row out as scale1, gate1, scale2, gate2.
	part := func(row, i int) []tensai.Float {
		return modRows[row*4*ditDim+i*ditDim:][:ditDim]
	}
	m := &Modulation{}
	for row := 0; row < 2; row++ {
		m.Scale1 = append(m.Scale1, part(row, 0))
		m.Gate1 = append(m.Gate1, part(row, 1))
		m.Scale2 = append(m.Scale2, part(row, 2))
		m.Gate2 = append(m.Gate2, part(row, 3))
	}
	// block.py marks the second half as target tokens, which read the
	// sampled timestep's row, and passes no attention mask, so every
	// query reads every key.
	l := &Layout{KeyLimit: make([]int, seq), Row: make([]int, seq)}
	for i := range l.Row {
		l.KeyLimit[i] = seq
		if i < seq/2 {
			l.Row[i] = 1
		}
	}

	w, err := OpenTransformer(filepath.Join(dir, "diffusion_pytorch_model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	b, err := LoadBlock(w, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Forward(x, m, l, rope, NewScratch(seq)); err != nil {
		t.Fatal(err)
	}

	// The reference runs in float32 but its weights came from bf16, and
	// a 4096-wide accumulation orders differently than torch's; the bar
	// is well under what one block's output means to the next.
	var worst float64
	var at int
	for i, v := range want {
		if d := math.Abs(float64(x.Data[i] - v)); d > worst {
			worst, at = d, i
		}
	}
	t.Logf("largest difference %.6g at element %d", worst, at)
	if worst > 1e-3 {
		t.Errorf("element %d is %g, reference %g", at, x.Data[at], want[at])
	}
}

// TestBlockInt8Error measures what quantizing a block's weights costs,
// on the same inputs the float test uses. The model does not fit in
// memory as floats, so this number decides whether it can run at all;
// it is reported rather than bounded tightly, since what matters is the
// picture at the end of 32 blocks and a denoising loop.
func TestBlockInt8Error(t *testing.T) {
	dir := transformerDir(t)
	const seq = 48
	read := func(name string, n int) []tensai.Float {
		return readF32(t, filepath.Join("testdata", name), n)
	}
	x := &tensai.Matrix{Rows: seq, Cols: ditDim, Data: read("block_x_48.f32", seq*ditDim)}
	modRows := read("block_mod_48.f32", 2*4*ditDim)
	want := read("block_y_48.f32", seq*ditDim)
	rope := &Rope{
		Cos: &tensai.Matrix{Rows: seq, Cols: ropePairs, Data: read("block_cos_48.f32", seq*ropePairs)},
		Sin: &tensai.Matrix{Rows: seq, Cols: ropePairs, Data: read("block_sin_48.f32", seq*ropePairs)},
	}
	part := func(row, i int) []tensai.Float {
		return modRows[row*4*ditDim+i*ditDim:][:ditDim]
	}
	m := &Modulation{}
	for row := 0; row < 2; row++ {
		m.Scale1 = append(m.Scale1, part(row, 0))
		m.Gate1 = append(m.Gate1, part(row, 1))
		m.Scale2 = append(m.Scale2, part(row, 2))
		m.Gate2 = append(m.Gate2, part(row, 3))
	}
	l := &Layout{KeyLimit: make([]int, seq), Row: make([]int, seq)}
	for i := range l.Row {
		l.KeyLimit[i] = seq
		if i < seq/2 {
			l.Row[i] = 1
		}
	}

	w, err := OpenTransformer(filepath.Join(dir, "diffusion_pytorch_model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	// Rotating the quantized weights and their inputs by a Hadamard
	// matrix spreads outlier columns across their group; the two rows
	// per width say what that buys.
	in := append([]tensai.Float(nil), x.Data...)
	defer func() { noRotate = false }()
	for _, bits := range []int{8, 4} {
		for _, off := range []bool{true, false} {
			noRotate = off
			b, err := LoadBlock(w, 0, bits)
			if err != nil {
				t.Fatal(err)
			}
			copy(x.Data, in)
			if err := b.Forward(x, m, l, rope, NewScratch(seq)); err != nil {
				t.Fatal(err)
			}
			var sq, ref float64
			for i, v := range want {
				d := float64(x.Data[i] - v)
				sq += d * d
				ref += float64(v) * float64(v)
			}
			how := "rotated"
			if off {
				how = "unrotated"
			}
			t.Logf("int%d %-9s relative error %.4f%% (rms %.4g against the block's %.4g)", bits, how,
				100*math.Sqrt(sq/ref), math.Sqrt(sq/float64(len(want))), math.Sqrt(ref/float64(len(want))))
		}
	}
}

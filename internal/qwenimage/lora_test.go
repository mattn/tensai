package qwenimage

import (
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
	"github.com/mattn/tensai/quant"
)

func testAdapterModel() *Transformer {
	layer := func() *linear { return &linear{f: tensai.NewMatrix(3, 4)} }
	return &Transformer{timeIn: layer(), timeUp: layer(), modulation: layer()}
}

func writeTestAdapter(t *testing.T, m *Transformer, missing bool) string {
	t.Helper()
	ts := map[string]*tensai.Tensor{}
	for name := range m.turboLayers() {
		a, b := tensai.NewTensor(2, 4), tensai.NewTensor(3, 2)
		for i := range a.Data {
			a.Data[i] = tensai.Float(i+1) * 0.02
		}
		for i := range b.Data {
			b.Data[i] = tensai.Float(i+1) * 0.03
		}
		ts["transformer."+name+".lora_A.weight"] = a
		ts["transformer."+name+".lora_B.weight"] = b
	}
	if missing {
		delete(ts, "transformer.modulation.1.lora_B.weight")
	}
	meta, _ := json.Marshal(map[string]any{"transformer.r": 2, "transformer.lora_alpha": 1, "transformer.bias": "none"})
	p := filepath.Join(t.TempDir(), "adapter.safetensors")
	if err := safetensors.SaveFile(p, ts, map[string]string{"lora_adapter_metadata": string(meta)}); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTurboAdapterLoad(t *testing.T) {
	m := testAdapterModel()
	m.timeIn.rot = 4 // the loader must rotate A to match the base input
	if err := LoadTurboLoRA(m, writeTestAdapter(t, m, true)); err == nil {
		t.Fatal("accepted incomplete adapter")
	}
	for _, l := range m.turboLayers() {
		if l.lora != nil {
			t.Fatal("partial attachment after failure")
		}
	}
	if err := LoadTurboLoRA(m, writeTestAdapter(t, m, false)); err != nil {
		t.Fatal(err)
	}
	x := tensai.NewMatrix(1, 4)
	copy(x.Data, []tensai.Float{1, 2, 3, 4})
	out := tensai.NewMatrix(1, 3)
	if err := m.timeIn.apply(out, x); err != nil {
		t.Fatal(err)
	}
	// A*x = [0.6, 1.4], alpha/r = 0.5. F32 tensors must
	// remain valid after LoadTurboLoRA closes the backing file.
	for i, want := range []float64{0.051, 0.111, 0.171} {
		if math.Abs(float64(out.Data[i])-want) > 1e-6 {
			t.Fatalf("%v", out.Data)
		}
	}
	if err := LoadTurboLoRA(m, writeTestAdapter(t, m, false)); err == nil {
		t.Fatal("accepted second adapter")
	}
}

func TestLoRARotatedInput(t *testing.T) {
	const in, out, rank, rows = 32, 16, 3, 5
	w := tensai.NewMatrix(in, out)
	for i := range w.Data {
		w.Data[i] = tensai.Float(math.Sin(float64(i))) * 0.1
	}
	q := quant.Quantize(w)
	l := &linear{q: q, rot: 16}
	a, b := tensai.NewMatrix(rank, in), tensai.NewMatrix(out, rank)
	for i := range a.Data {
		a.Data[i] = tensai.Float(math.Cos(float64(i))) * 0.1
	}
	for i := range b.Data {
		b.Data[i] = tensai.Float(math.Sin(float64(i))) * 0.1
	}
	x := tensai.NewMatrix(rows, in)
	for i := range x.Data {
		x.Data[i] = tensai.Float(math.Sin(float64(i) * 0.3))
	}
	want, got := tensai.NewMatrix(rows, out), tensai.NewMatrix(rows, out)
	if err := l.apply(want, x); err != nil {
		t.Fatal(err)
	}
	unrotated := &lowRank{a: a, b: b, scale: 0.4}
	if err := unrotated.add(want, x); err != nil {
		t.Fatal(err)
	}
	ra := clone(a)
	rotateRows(ra, l.rot)
	l.lora = &lowRank{a: ra, b: b, scale: 0.4}
	if err := l.apply(got, x); err != nil {
		t.Fatal(err)
	}
	for i, v := range got.Data {
		if math.Abs(float64(v-want.Data[i])) > 1e-6 {
			t.Fatalf("%d: %g != %g", i, v, want.Data[i])
		}
	}
}

func TestTurboSchedule(t *testing.T) {
	for _, tokens := range []int{256, 1024, 8192} {
		s := NewTurboSchedule(tokens)
		if s.Steps() != 6 || s.Sigmas[0] != 1 || s.Sigmas[6] != 0 {
			t.Fatal(s)
		}
		mu := math.Exp(0.5 + float64(tokens-256)*0.4/7936)
		for i, raw := range []float64{1, .9375, .875, .75, .5, .25} {
			want := mu * raw / (1 + (mu-1)*raw)
			if math.Abs(s.Sigmas[i]-want) > 1e-12 {
				t.Fatalf("%d/%d: %g != %g", tokens, i, s.Sigmas[i], want)
			}
		}
		if s.Sigmas[5] <= 0.25 {
			t.Fatal("unexpected terminal stretch", s)
		}
	}
}

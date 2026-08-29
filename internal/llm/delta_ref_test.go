package llm

import (
	"testing"

	"github.com/mattn/tensai"
)

// lcg is a generator simple enough to reproduce exactly in the reference
// script, so both sides see the same weights without shipping a fixture.
type lcg struct{ x uint32 }

func (r *lcg) next() float32 {
	r.x = r.x*1103515245 + 12345
	return float32(r.x>>1)/float32(1<<31) - 0.5
}

func (r *lcg) vec(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.next()
	}
	return v
}

// mat fills an [in, out] matrix, the orientation mvInto wants and the one
// the loader leaves behind after transposing the checkpoint's.
func (r *lcg) mat(in, out int) *tensai.Matrix {
	m := tensai.NewMatrix(in, out)
	for i := range m.Data {
		m.Data[i] = r.next()
	}
	return m
}

// wantDeltaStep is the third step of the layer below, computed
// independently in float64 from the formula in transformers'
// Qwen3_5GatedDeltaNet. It pins the arithmetic this file gets right by
// construction and nothing else would notice getting wrong: a decay, a
// write, and a read against a state that carries across tokens.
var wantDeltaStep = []float32{
	-0.032711055, -0.011559756, -0.002720115, 0.084442301,
	0.019588967, 0.068860133, -0.040629612, 0.040702248,
}

// TestDeltaAgainstReference runs one small layer for three tokens, so the
// convolution window and the recurrent state both have history by the
// last one, and compares against the reference.
func TestDeltaAgainstReference(t *testing.T) {
	const hidden, heads, kd, vd, convK = 8, 2, 4, 4, 4
	r := &lcg{x: 12345}
	d := &deltaWeights{heads: heads, kDim: kd, vDim: vd, convK: convK}
	d.convDim = kd*heads*2 + vd*heads
	d.wQKV = r.mat(hidden, d.convDim)
	d.wZ = r.mat(hidden, vd*heads)
	d.wA = r.mat(hidden, heads)
	d.wB = r.mat(hidden, heads)
	d.wOut = r.mat(vd*heads, hidden)
	d.conv = r.vec(d.convDim * convK)
	d.aLog = r.vec(heads)
	d.dtBias = r.vec(heads)
	d.norm = r.vec(vd)
	if err := d.check(); err != nil {
		t.Fatal(err)
	}

	st := d.newState()
	scratch := newDeltaScratch(d, hidden)
	var got []float32
	for i := 0; i < 3; i++ {
		got = append([]float32(nil), d.step(st, r.vec(hidden), scratch)...)
	}
	if len(got) != len(wantDeltaStep) {
		t.Fatalf("step returned %d values, want %d", len(got), len(wantDeltaStep))
	}
	for i, want := range wantDeltaStep {
		if diff := got[i] - want; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("out[%d] = %.9f, want %.9f", i, got[i], want)
		}
	}
}

// A layer that carried no state would give the same answer every step,
// which is the failure the reference values alone would not catch.
func TestDeltaCarriesState(t *testing.T) {
	const hidden, heads, kd, vd, convK = 8, 2, 4, 4, 4
	r := &lcg{x: 999}
	d := &deltaWeights{heads: heads, kDim: kd, vDim: vd, convK: convK}
	d.convDim = kd*heads*2 + vd*heads
	d.wQKV, d.wZ = r.mat(hidden, d.convDim), r.mat(hidden, vd*heads)
	d.wA, d.wB = r.mat(hidden, heads), r.mat(hidden, heads)
	d.wOut = r.mat(vd*heads, hidden)
	d.conv, d.aLog, d.dtBias, d.norm = r.vec(d.convDim*convK), r.vec(heads), r.vec(heads), r.vec(vd)

	x := r.vec(hidden)
	st, scratch := d.newState(), newDeltaScratch(d, hidden)
	first := append([]float32(nil), d.step(st, x, scratch)...)
	second := append([]float32(nil), d.step(st, x, scratch)...)
	same := true
	for i := range first {
		if first[i] != second[i] {
			same = false
		}
	}
	if same {
		t.Error("the same token twice gave the same answer: the state is not carrying")
	}
	// A fresh state must reproduce the first answer exactly.
	again := d.step(d.newState(), x, newDeltaScratch(d, hidden))
	for i := range first {
		if again[i] != first[i] {
			t.Fatalf("a fresh state gave %v, want %v", again[i], first[i])
		}
	}
}

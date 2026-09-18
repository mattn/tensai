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
	d := &deltaWeights{heads: heads, kHeads: heads, kDim: kd, vDim: vd, convK: convK}
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
	d.fuse()
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
	d := &deltaWeights{heads: heads, kHeads: heads, kDim: kd, vDim: vd, convK: convK}
	d.convDim = kd*heads*2 + vd*heads
	d.wQKV, d.wZ = r.mat(hidden, d.convDim), r.mat(hidden, vd*heads)
	d.wA, d.wB = r.mat(hidden, heads), r.mat(hidden, heads)
	d.wOut = r.mat(vd*heads, hidden)
	d.conv, d.aLog, d.dtBias, d.norm = r.vec(d.convDim*convK), r.vec(heads), r.vec(heads), r.vec(vd)
	d.fuse()

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

// permCols reorders blocks of width w in a matrix's columns: output
// block i is input block perm[i].
func permCols(m *tensai.Matrix, w int, perm []int) *tensai.Matrix {
	out := tensai.NewMatrix(m.Rows, m.Cols)
	for r := 0; r < m.Rows; r++ {
		for i, p := range perm {
			copy(out.Data[r*m.Cols+i*w:r*m.Cols+(i+1)*w], m.Data[r*m.Cols+p*w:r*m.Cols+(p+1)*w])
		}
	}
	return out
}

// permRows does the same to blocks of rows.
func permRows(m *tensai.Matrix, w int, perm []int) *tensai.Matrix {
	out := tensai.NewMatrix(m.Rows, m.Cols)
	for i, p := range perm {
		copy(out.Data[i*w*m.Cols:(i+1)*w*m.Cols], m.Data[p*w*m.Cols:(p+1)*w*m.Cols])
	}
	return out
}

// permVec reorders blocks of a vector.
func permVec(v []float32, w int, perm []int) []float32 {
	out := make([]float32, len(v))
	for i, p := range perm {
		copy(out[i*w:(i+1)*w], v[p*w:(p+1)*w])
	}
	return out
}

// Fewer key heads than value heads: value heads share a key head, and
// the layer must read the same key whichever order the value heads come
// in. A layer in HF's grouped order and the same layer with its value
// heads tiled the way a gguf converter leaves them give the same answer.
func TestDeltaKeyHeadGrouping(t *testing.T) {
	const hidden, kHeads, heads, kd, vd, convK = 8, 2, 4, 4, 4, 4
	r := &lcg{x: 4242}
	g := &deltaWeights{heads: heads, kHeads: kHeads, kDim: kd, vDim: vd, convK: convK}
	g.convDim = kd*kHeads*2 + vd*heads
	g.wQKV, g.wZ = r.mat(hidden, g.convDim), r.mat(hidden, vd*heads)
	g.wA, g.wB = r.mat(hidden, heads), r.mat(hidden, heads)
	g.wOut = r.mat(vd*heads, hidden)
	g.conv, g.aLog, g.dtBias, g.norm = r.vec(g.convDim*convK), r.vec(heads), r.vec(heads), r.vec(vd)
	g.fuse()
	if err := g.check(); err != nil {
		t.Fatal(err)
	}
	// Tiled order lists value head (key head k, replica j) at j*kHeads+k:
	// with two key heads and two replicas, grouped heads 0,2,1,3.
	perm := []int{0, 2, 1, 3}
	tl := &deltaWeights{heads: heads, kHeads: kHeads, tiled: true, kDim: kd, vDim: vd, convK: convK, convDim: g.convDim}
	keyDim := kd * kHeads
	tl.wQKV = tensai.NewMatrix(hidden, g.convDim)
	for row := 0; row < hidden; row++ {
		src := g.wQKV.Data[row*g.convDim : (row+1)*g.convDim]
		dst := tl.wQKV.Data[row*g.convDim : (row+1)*g.convDim]
		copy(dst, src[:2*keyDim])
		copy(dst[2*keyDim:], permVec(src[2*keyDim:], vd, perm))
	}
	tl.wZ = permCols(g.wZ, vd, perm)
	tl.wA, tl.wB = permCols(g.wA, 1, perm), permCols(g.wB, 1, perm)
	tl.wOut = permRows(g.wOut, vd, perm)
	tl.conv = append(append([]float32(nil), g.conv[:2*keyDim*convK]...), permVec(g.conv[2*keyDim*convK:], vd*convK, perm)...)
	tl.aLog, tl.dtBias, tl.norm = permVec(g.aLog, 1, perm), permVec(g.dtBias, 1, perm), g.norm
	tl.fuse()

	gs, gsc := g.newState(), newDeltaScratch(g, hidden)
	ts, tsc := tl.newState(), newDeltaScratch(tl, hidden)
	for i := 0; i < 3; i++ {
		x := r.vec(hidden)
		want := append([]float32(nil), g.step(gs, x, gsc)...)
		got := tl.step(ts, x, tsc)
		for j := range want {
			if diff := got[j] - want[j]; diff > 1e-5 || diff < -1e-5 {
				t.Fatalf("step %d out[%d] = %.7f tiled, %.7f grouped", i, j, got[j], want[j])
			}
		}
	}
	// And the grouped layer is not secretly reading one key head for
	// every value head: a key head lookup that ignored the value head
	// would give a different answer.
	wrong := *g
	wrong.kHeads = 1
	wrong.heads = 4
	ws, wsc := wrong.newState(), newDeltaScratch(&wrong, hidden)
	gs, gsc = g.newState(), newDeltaScratch(g, hidden)
	x := r.vec(hidden)
	want := append([]float32(nil), g.step(gs, x, gsc)...)
	got := wrong.step(ws, x, wsc)
	same := true
	for j := range want {
		if got[j] != want[j] {
			same = false
		}
	}
	if same {
		t.Fatal("key head grouping made no difference")
	}
}

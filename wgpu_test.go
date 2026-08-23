//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func openTestGPU(t *testing.T) *GPU {
	t.Helper()
	g, err := OpenGPU()
	if err != nil {
		t.Skipf("wgpu unavailable: %v", err)
	}
	return g
}

func TestGPUMatMul(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(7))
	cases := [][2][]int{
		{{3, 4}, {4, 5}},
		{{2, 3, 4}, {2, 4, 5}},
		{{3, 4}, {6, 4, 5}},
		{{6, 3, 4}, {4, 5}},
		{{2, 1, 3, 4}, {1, 5, 4, 2}},
		{{1, 17, 9}, {4, 9, 33}}, // not multiples of the 8x8 workgroup
	}
	for _, c := range cases {
		a, b := randTensor(rng, c[0]...), randTensor(rng, c[1]...)
		got, err := g.MatMul(a, b)
		if err != nil {
			t.Fatalf("gpu matmul %v*%v: %v", c[0], c[1], err)
		}
		want, err := MatMul(a, b)
		if err != nil {
			t.Fatalf("cpu matmul: %v", err)
		}
		if !sameDims(got.Shape, want.Shape) {
			t.Fatalf("shape: got %v want %v", got.Shape, want.Shape)
		}
		for i := range want.Data {
			if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
				t.Fatalf("%v*%v element %d: gpu=%v cpu=%v", c[0], c[1], i, got.Data[i], want.Data[i])
			}
		}
	}

	if _, err := g.MatMul(randTensor(rng, 2, 3), randTensor(rng, 4, 5)); err == nil {
		t.Fatal("expected shape mismatch error")
	}
	if _, err := g.MatMul(randTensor(rng, 4), randTensor(rng, 4, 5)); err == nil {
		t.Fatal("expected error for 1-D operand")
	}
}

func TestGPUAdapterSelection(t *testing.T) {
	// Power preference is a hint, so with any adapter set this must still
	// succeed and report a name.
	g, err := OpenGPU(GPULowPower)
	if err != nil {
		t.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	if g.Name() == "" {
		t.Fatal("adapter name is empty")
	}
	t.Logf("adapter: %s", g.Name())

	x, err := NewTensorFromSlice([]Float{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.MatMul(x, x)
	if err != nil {
		t.Fatalf("gpu matmul: %v", err)
	}
	want := []Float{7, 10, 15, 22}
	for i, v := range want {
		if got.Data[i] != v {
			t.Fatalf("element %d: got %v want %v", i, got.Data[i], v)
		}
	}
}

func TestGPUTensorResident(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(11))

	// A weight uploaded once serves several MatMuls without re-upload.
	w := randTensor(rng, 64, 32)
	gw, err := g.Upload(w)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	defer gw.Free()
	if !sameDims(gw.Shape(), []int{64, 32}) || gw.Size() != 64*32 {
		t.Fatalf("shape/size: %v %d", gw.Shape(), gw.Size())
	}
	for i := 0; i < 3; i++ {
		x := randTensor(rng, 4, 5, 64)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatalf("upload x: %v", err)
		}
		gy, err := gx.MatMul(gw)
		if err != nil {
			t.Fatalf("resident matmul: %v", err)
		}
		got, err := gy.Download()
		if err != nil {
			t.Fatalf("download: %v", err)
		}
		want, err := MatMul(x, w)
		if err != nil {
			t.Fatalf("cpu matmul: %v", err)
		}
		if !sameDims(got.Shape, want.Shape) {
			t.Fatalf("shape: got %v want %v", got.Shape, want.Shape)
		}
		for j := range want.Data {
			if diff := math.Abs(float64(got.Data[j] - want.Data[j])); diff > 1e-4 {
				t.Fatalf("round %d element %d: gpu=%v cpu=%v", i, j, got.Data[j], want.Data[j])
			}
		}
		gy.Free()
		gx.Free()
	}
}

func TestGPUTensorChain(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(12))

	// (a @ b) @ c entirely on the GPU: the intermediate never touches the
	// host.
	a, b, c := randTensor(rng, 2, 4, 8), randTensor(rng, 2, 8, 3), randTensor(rng, 2, 3, 6)
	ga, err := g.Upload(a)
	if err != nil {
		t.Fatal(err)
	}
	defer ga.Free()
	gb, err := g.Upload(b)
	if err != nil {
		t.Fatal(err)
	}
	defer gb.Free()
	gc, err := g.Upload(c)
	if err != nil {
		t.Fatal(err)
	}
	defer gc.Free()

	ab, err := ga.MatMul(gb)
	if err != nil {
		t.Fatalf("a@b: %v", err)
	}
	defer ab.Free()
	abc, err := ab.MatMul(gc)
	if err != nil {
		t.Fatalf("(a@b)@c: %v", err)
	}
	got, err := abc.Download()
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	wantAB, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want, err := MatMul(wantAB, c)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}

	if _, err := ab.MatMul(ab); err == nil {
		t.Fatal("expected shape mismatch error for (2,4,3)@(2,4,3)")
	}
	abc.Free()
	if _, err := abc.Download(); err == nil {
		t.Fatal("expected error downloading a freed tensor")
	}
	if _, err := abc.MatMul(ga); err == nil {
		t.Fatal("expected error using a freed tensor")
	}
	abc.Free() // second Free is a no-op
}

// cpuSoftmaxLast applies softmax along the last axis, in place.
func cpuSoftmaxLast(x *Tensor) {
	n := x.Shape[len(x.Shape)-1]
	for pos := 0; pos < len(x.Data); pos += n {
		row := x.Data[pos : pos+n]
		maxv := row[0]
		for _, v := range row[1:] {
			if v > maxv {
				maxv = v
			}
		}
		var sum Float
		for i, v := range row {
			row[i] = expF(v - maxv)
			sum += row[i]
		}
		for i := range row {
			row[i] /= sum
		}
	}
}

func TestGPUKernels(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(15))

	// MatMulT against MatMul on a materialized transpose, batched.
	a, b := randTensor(rng, 3, 5, 8), randTensor(rng, 3, 7, 8)
	ga, _ := g.Upload(a)
	gb, _ := g.Upload(b)
	defer ga.Free()
	defer gb.Free()
	gt, err := ga.MatMulT(gb)
	if err != nil {
		t.Fatalf("matmul-t: %v", err)
	}
	defer gt.Free()
	got, err := gt.Download()
	if err != nil {
		t.Fatal(err)
	}
	bt, err := b.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	want, err := MatMul(a, bt)
	if err != nil {
		t.Fatal(err)
	}
	if !sameDims(got.Shape, want.Shape) {
		t.Fatalf("matmul-t shape: got %v want %v", got.Shape, want.Shape)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("matmul-t element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}
	if _, err := ga.MatMulT(gt); err == nil { // (3,5,8) @ (3,5,7)^T: k mismatch
		t.Fatal("expected matmul-t shape mismatch error")
	}

	// Scale in place.
	x := randTensor(rng, 4, 33)
	gx, _ := g.Upload(x)
	defer gx.Free()
	if err := gx.Scale(0.5); err != nil {
		t.Fatalf("scale: %v", err)
	}
	sGot, err := gx.Download()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range x.Data {
		if diff := math.Abs(float64(sGot.Data[i] - v*0.5)); diff > 1e-6 {
			t.Fatalf("scale element %d: gpu=%v want=%v", i, sGot.Data[i], v*0.5)
		}
	}

	// Softmax over the last axis; 300 columns exercise the strided loops
	// beyond one 256-lane workgroup pass.
	for _, shape := range [][]int{{5, 30}, {2, 3, 300}} {
		y := randTensor(rng, shape...)
		gy, _ := g.Upload(y)
		sm, err := gy.Softmax()
		if err != nil {
			t.Fatalf("softmax %v: %v", shape, err)
		}
		smGot, err := sm.Download()
		if err != nil {
			t.Fatal(err)
		}
		cpuSoftmaxLast(y)
		for i := range y.Data {
			if diff := math.Abs(float64(smGot.Data[i] - y.Data[i])); diff > 1e-5 {
				t.Fatalf("softmax %v element %d: gpu=%v cpu=%v", shape, i, smGot.Data[i], y.Data[i])
			}
		}
		sm.Free()
		gy.Free()
	}
}

func TestGPUAttention(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(16))

	// Batched single-head attention entirely on the GPU vs the same math
	// on the CPU tensor ops.
	q, k, v := randTensor(rng, 2, 6, 8), randTensor(rng, 2, 6, 8), randTensor(rng, 2, 6, 8)
	gq, _ := g.Upload(q)
	gk, _ := g.Upload(k)
	gv, _ := g.Upload(v)
	defer gq.Free()
	defer gk.Free()
	defer gv.Free()

	out, err := gq.Attention(gk, gv)
	if err != nil {
		t.Fatalf("attention: %v", err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}

	kt, err := k.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	scores, err := MatMul(q, kt)
	if err != nil {
		t.Fatal(err)
	}
	scores.Scale(1 / sqrtF(8))
	cpuSoftmaxLast(scores)
	want, err := MatMul(scores, v)
	if err != nil {
		t.Fatal(err)
	}
	if !sameDims(got.Shape, want.Shape) {
		t.Fatalf("shape: got %v want %v", got.Shape, want.Shape)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}
}

// cpuAttention is the reference softmax(q*k^T/sqrt(d))*v on CPU tensors.
func cpuAttention(t *testing.T, q, k, v *Tensor) *Tensor {
	t.Helper()
	kt, err := k.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	s, err := MatMul(q, kt)
	if err != nil {
		t.Fatal(err)
	}
	s.Scale(1 / sqrtF(Float(q.Shape[len(q.Shape)-1])))
	cpuSoftmaxLast(s)
	out, err := MatMul(s, v)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestGPUMultiHeadAttention(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(17))

	const batch, seq, seqKV, heads, dh = 2, 5, 7, 3, 4
	const d = heads * dh
	q := randTensor(rng, batch, seq, d)
	k := randTensor(rng, batch, seqKV, d)
	v := randTensor(rng, batch, seqKV, d)
	gq, _ := g.Upload(q)
	gk, _ := g.Upload(k)
	gv, _ := g.Upload(v)
	defer gq.Free()
	defer gk.Free()
	defer gv.Free()

	out, err := gq.MultiHeadAttention(gk, gv, heads)
	if err != nil {
		t.Fatalf("multi-head attention: %v", err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	if !sameDims(got.Shape, []int{batch, seq, d}) {
		t.Fatalf("shape: got %v", got.Shape)
	}

	// Reference: slice each head out of the packed layout, run single-head
	// attention on the CPU, and scatter the result back.
	want := NewTensor(batch, seq, d)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			slice := func(x *Tensor, rows int) *Tensor {
				s := NewTensor(rows, dh)
				for r := 0; r < rows; r++ {
					for c := 0; c < dh; c++ {
						s.Data[r*dh+c] = x.Data[(b*rows+r)*d+h*dh+c]
					}
				}
				return s
			}
			ho := cpuAttention(t, slice(q, seq), slice(k, seqKV), slice(v, seqKV))
			for r := 0; r < seq; r++ {
				for c := 0; c < dh; c++ {
					want.Data[(b*seq+r)*d+h*dh+c] = ho.Data[r*dh+c]
				}
			}
		}
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}

	// heads=1 must agree with the single-head Attention path.
	qs := randTensor(rng, 2, 6, 8)
	ks := randTensor(rng, 2, 6, 8)
	vs := randTensor(rng, 2, 6, 8)
	gqs, _ := g.Upload(qs)
	gks, _ := g.Upload(ks)
	gvs, _ := g.Upload(vs)
	defer gqs.Free()
	defer gks.Free()
	defer gvs.Free()
	mh, err := gqs.MultiHeadAttention(gks, gvs, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer mh.Free()
	sh, err := gqs.Attention(gks, gvs)
	if err != nil {
		t.Fatal(err)
	}
	defer sh.Free()
	mhT, _ := mh.Download()
	shT, _ := sh.Download()
	for i := range shT.Data {
		if diff := math.Abs(float64(mhT.Data[i] - shT.Data[i])); diff > 1e-5 {
			t.Fatalf("heads=1 mismatch at %d: %v vs %v", i, mhT.Data[i], shT.Data[i])
		}
	}

	if _, err := gq.MultiHeadAttention(gk, gv, 5); err == nil {
		t.Fatal("expected error: 5 heads do not divide d=12")
	}
	if _, err := gq.MultiHeadAttention(gqs, gqs, 2); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

// cpuCausalAttention is single-head causal attention on the CPU: query i
// attends to key positions 0..i+(seqKV-seqQ).
func cpuCausalAttention(t *testing.T, q, k, v *Tensor) *Tensor {
	t.Helper()
	seq, d := q.Shape[0], q.Shape[1]
	seqKV := k.Shape[0]
	out := NewTensor(seq, d)
	scores := make([]float64, seqKV)
	for i := 0; i < seq; i++ {
		limit := i + seqKV - seq + 1
		maxs := math.Inf(-1)
		for j := 0; j < limit; j++ {
			var s float64
			for c := 0; c < d; c++ {
				s += float64(q.Data[i*d+c]) * float64(k.Data[j*d+c])
			}
			s /= math.Sqrt(float64(d))
			scores[j] = s
			if s > maxs {
				maxs = s
			}
		}
		var sum float64
		for j := 0; j < limit; j++ {
			scores[j] = math.Exp(scores[j] - maxs)
			sum += scores[j]
		}
		for j := 0; j < limit; j++ {
			p := float32(scores[j] / sum)
			for c := 0; c < d; c++ {
				out.Data[i*d+c] += p * v.Data[j*d+c]
			}
		}
	}
	return out
}

func TestGPUCausalAttention(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(19))

	// Prefill shape: seqQ == seqKV, single head.
	q, k, v := randTensor(rng, 6, 8), randTensor(rng, 6, 8), randTensor(rng, 6, 8)
	gq, _ := g.Upload(q)
	gk, _ := g.Upload(k)
	gv, _ := g.Upload(v)
	defer gq.Free()
	defer gk.Free()
	defer gv.Free()
	out, err := gq.CausalAttention(gk, gv)
	if err != nil {
		t.Fatalf("causal attention: %v", err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	want := cpuCausalAttention(t, q, k, v)
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}

	// Chunked decode: 3 fresh queries against 7 cached positions.
	q2, k2, v2 := randTensor(rng, 3, 8), randTensor(rng, 7, 8), randTensor(rng, 7, 8)
	gq2, _ := g.Upload(q2)
	gk2, _ := g.Upload(k2)
	gv2, _ := g.Upload(v2)
	defer gq2.Free()
	defer gk2.Free()
	defer gv2.Free()
	out2, err := gq2.CausalAttention(gk2, gv2)
	if err != nil {
		t.Fatalf("causal attention kv>q: %v", err)
	}
	defer out2.Free()
	got2, err := out2.Download()
	if err != nil {
		t.Fatal(err)
	}
	want2 := cpuCausalAttention(t, q2, k2, v2)
	for i := range want2.Data {
		if diff := math.Abs(float64(got2.Data[i] - want2.Data[i])); diff > 1e-4 {
			t.Fatalf("kv>q element %d: gpu=%v cpu=%v", i, got2.Data[i], want2.Data[i])
		}
	}

	if _, err := gq2.CausalAttention(gq2, gq2); err != nil {
		t.Fatalf("seqKV == seqQ must be accepted: %v", err)
	}
	if _, err := gk2.CausalAttention(gq2, gq2); err == nil {
		t.Fatal("expected error for seqKV < seqQ")
	}
}

func TestGPUCausalMultiHead(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(20))

	const seq, seqKV, heads, dh = 5, 9, 3, 4
	const d = heads * dh
	q, k, v := randTensor(rng, seq, d), randTensor(rng, seqKV, d), randTensor(rng, seqKV, d)
	gq, _ := g.Upload(q)
	gk, _ := g.Upload(k)
	gv, _ := g.Upload(v)
	defer gq.Free()
	defer gk.Free()
	defer gv.Free()
	out, err := gq.CausalMultiHeadAttention(gk, gv, heads)
	if err != nil {
		t.Fatalf("causal mha: %v", err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}

	// Reference: slice heads out, run single-head causal attention.
	want := NewTensor(seq, d)
	for h := 0; h < heads; h++ {
		slice := func(x *Tensor, rows int) *Tensor {
			s := NewTensor(rows, dh)
			for r := 0; r < rows; r++ {
				copy(s.Data[r*dh:(r+1)*dh], x.Data[r*d+h*dh:r*d+(h+1)*dh])
			}
			return s
		}
		ho := cpuCausalAttention(t, slice(q, seq), slice(k, seqKV), slice(v, seqKV))
		for r := 0; r < seq; r++ {
			copy(want.Data[r*d+h*dh:r*d+(h+1)*dh], ho.Data[r*dh:(r+1)*dh])
		}
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-4 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestGPUDispatch2D(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(18))

	// 70000 rows exceed the 65535 single-axis dispatch limit, exercising
	// the 2-D workgroup grid in softmax.
	x := randTensor(rng, 70000, 8)
	gx, err := g.Upload(x)
	if err != nil {
		t.Fatal(err)
	}
	defer gx.Free()
	sm, err := gx.Softmax()
	if err != nil {
		t.Fatalf("softmax: %v", err)
	}
	defer sm.Free()
	got, err := sm.Download()
	if err != nil {
		t.Fatal(err)
	}
	cpuSoftmaxLast(x)
	for i := range x.Data {
		if diff := math.Abs(float64(got.Data[i] - x.Data[i])); diff > 1e-5 {
			t.Fatalf("softmax element %d: gpu=%v cpu=%v", i, got.Data[i], x.Data[i])
		}
	}

	// 17.4M elements exceed 65535 workgroups of 256 lanes for scale.
	y := randTensor(rng, 68000, 256)
	gy, err := g.Upload(y)
	if err != nil {
		t.Fatal(err)
	}
	defer gy.Free()
	if err := gy.Scale(2); err != nil {
		t.Fatalf("scale: %v", err)
	}
	sGot, err := gy.Download()
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{0, 12345, len(y.Data) / 2, len(y.Data) - 1} {
		if diff := math.Abs(float64(sGot.Data[i] - y.Data[i]*2)); diff > 1e-5 {
			t.Fatalf("scale element %d: gpu=%v want=%v", i, sGot.Data[i], y.Data[i]*2)
		}
	}
}

func TestGPUMatMulLarge(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(8))
	a, b := randTensor(rng, 8, 64, 96), randTensor(rng, 8, 96, 80)
	got, err := g.MatMul(a, b)
	if err != nil {
		t.Fatalf("gpu matmul: %v", err)
	}
	want, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("cpu matmul: %v", err)
	}
	for i := range want.Data {
		// f32 accumulation order differs between CPU and GPU.
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > 1e-2 {
			t.Fatalf("element %d: gpu=%v cpu=%v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestGPUQ8MatMul(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(61))
	for _, c := range []struct{ m, rows, cols int }{
		{1, 256, 512},
		{3, 64, 33}, // cols not a multiple of 4: guarded tail
		{5, 33, 7},  // odd rows: interleave pad must not leak
	} {
		w := RandomMatrix(c.rows, c.cols, rng)
		q := QuantizeMatrix(w)
		gq, err := g.UploadQ8(q)
		if err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		x := randTensor(rng, c.m, c.rows)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatal(err)
		}
		got, err := gq.MatMul(gx)
		if err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		out, err := got.Download()
		if err != nil {
			t.Fatal(err)
		}

		// Reference: the same dequantized weights on the CPU. The GPU
		// multiplies f32 activations (no 7-bit activation step), so the
		// comparison is float tolerance, not bit equality.
		for r := 0; r < c.m; r++ {
			for j := 0; j < c.cols; j++ {
				var want float64
				for i := 0; i < c.rows; i++ {
					wq := float64(q.Q[(i/2)*2*c.cols+2*j+i%2]) * float64(q.Scale[j])
					want += float64(x.Data[r*c.rows+i]) * wq
				}
				gotv := float64(out.Data[r*c.cols+j])
				if diff := math.Abs(gotv - want); diff > 1e-3*(1+math.Abs(want)) {
					t.Fatalf("%v row %d col %d: gpu=%v cpu=%v", c, r, j, gotv, want)
				}
			}
		}
		got.Free()
		gx.Free()
		gq.Free()
	}

	q := QuantizeMatrix(RandomMatrix(8, 8, rng))
	gq, err := g.UploadQ8(q)
	if err != nil {
		t.Fatal(err)
	}
	defer gq.Free()
	gx, err := g.Upload(randTensor(rng, 2, 7))
	if err != nil {
		t.Fatal(err)
	}
	defer gx.Free()
	if _, err := gq.MatMul(gx); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

// BenchmarkGPUQ8MatVec measures the decode shape — one activation row
// against a large resident weight — for the int8 kernel next to the f32
// matmul it replaces. 2048x8192 keeps the f32 twin (64MB) under modest
// device storage limits (dzn stops at 128MB).
func BenchmarkGPUQ8MatVec(b *testing.B) {
	g, err := OpenGPU()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(62))
	w := RandomMatrix(2048, 8192, rng)
	gq, err := g.UploadQ8(QuantizeMatrix(w))
	if err != nil {
		b.Fatal(err)
	}
	defer gq.Free()
	gx, err := g.Upload(randTensor(rng, 1, 2048))
	if err != nil {
		b.Fatal(err)
	}
	defer gx.Free()
	b.SetBytes(2048 * 8192)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := gq.MatMul(gx)
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
}

func BenchmarkGPUF32MatVec(b *testing.B) {
	g, err := OpenGPU()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(62))
	wt := randTensor(rng, 2048, 8192)
	gw, err := g.Upload(wt)
	if err != nil {
		b.Fatal(err)
	}
	defer gw.Free()
	gx, err := g.Upload(randTensor(rng, 1, 2048))
	if err != nil {
		b.Fatal(err)
	}
	defer gx.Free()
	b.SetBytes(2048 * 8192 * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := gx.MatMul(gw)
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
}

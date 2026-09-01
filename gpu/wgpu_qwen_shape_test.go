//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

// TestGPUGroupedCausalAttentionQwenShape runs the grouped kernel at the
// Qwen2.5-0.5B shape (group 7, dh 64) against the scalar reference.
func TestGPUGroupedCausalAttentionQwenShape(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(7))

	const heads, kvHeads, dh = 14, 2, 64
	const d, kvDim = heads * dh, kvHeads * dh
	const seq, seqKV = 70, 70
	group := heads / kvHeads
	k := randTensor(rng, seqKV, kvDim)
	v := randTensor(rng, seqKV, kvDim)
	gk, err := g.Upload(k)
	if err != nil {
		t.Fatal(err)
	}
	defer gk.Free()
	gv, err := g.Upload(v)
	if err != nil {
		t.Fatal(err)
	}
	defer gv.Free()
	q := randTensor(rng, seq, d)
	gq, err := g.Upload(q)
	if err != nil {
		t.Fatal(err)
	}
	defer gq.Free()
	got, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Free()
	out, err := got.Download()
	if err != nil {
		t.Fatal(err)
	}

	scale := 1 / math.Sqrt(dh)
	for qi := 0; qi < seq; qi++ {
		limit := qi + seqKV - seq + 1
		for h := 0; h < heads; h++ {
			kvOff := (h / group) * dh
			scores := make([]float64, limit)
			maxs := math.Inf(-1)
			for j := 0; j < limit; j++ {
				var s float64
				for c := 0; c < dh; c++ {
					s += float64(q.Data[qi*d+h*dh+c]) * float64(k.Data[j*kvDim+kvOff+c])
				}
				scores[j] = s * scale
				maxs = math.Max(maxs, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - maxs)
				sum += scores[j]
			}
			for c := 0; c < dh; c++ {
				var want float64
				for j := 0; j < limit; j++ {
					want += scores[j] / sum * float64(v.Data[j*kvDim+kvOff+c])
				}
				gotv := float64(out.Data[qi*d+h*dh+c])
				if diff := math.Abs(gotv - want); diff > 1e-3 {
					t.Fatalf("query %d head %d chan %d: got %v want %v", qi, h, c, gotv, want)
				}
			}
		}
	}
}

// TestGPUGroupedCausalAttentionF16 runs the same shape against the
// half-precision cache path: rows convert in through CopyRowsInto, and
// the scores tolerate f16's coarser K/V.
func TestGPUGroupedCausalAttentionF16(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	if !g.HasF16() {
		t.Skip("device has no shader-f16")
	}
	rng := rand.New(rand.NewSource(16))

	const heads, kvHeads, dh = 14, 2, 64
	const d, kvDim = heads * dh, kvHeads * dh
	const seq, seqKV = 70, 70
	group := heads / kvHeads
	k := randTensor(rng, seqKV, kvDim)
	v := randTensor(rng, seqKV, kvDim)
	upload16 := func(src *tensai.Tensor) *Tensor {
		t.Helper()
		f32, err := g.Upload(src)
		if err != nil {
			t.Fatal(err)
		}
		defer f32.Free()
		c, err := g.NewF16Tensor(seqKV, kvDim)
		if err != nil {
			t.Fatal(err)
		}
		if err := f32.CopyRowsInto(c, 0); err != nil {
			t.Fatal(err)
		}
		return c
	}
	gk := upload16(k)
	defer gk.Free()
	gv := upload16(v)
	defer gv.Free()
	q := randTensor(rng, seq, d)
	gq, err := g.Upload(q)
	if err != nil {
		t.Fatal(err)
	}
	defer gq.Free()
	got, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Free()
	out, err := got.Download()
	if err != nil {
		t.Fatal(err)
	}

	// Reference in f64 over f16-rounded K/V, so the tolerance covers the
	// kernel, not the storage narrowing itself.
	h16 := func(x float32) float64 {
		return float64(float16round(x))
	}
	scale := 1 / math.Sqrt(dh)
	for qi := 0; qi < seq; qi++ {
		limit := qi + seqKV - seq + 1
		for h := 0; h < heads; h++ {
			kvOff := (h / group) * dh
			scores := make([]float64, limit)
			maxs := math.Inf(-1)
			for j := 0; j < limit; j++ {
				var s float64
				for c := 0; c < dh; c++ {
					s += float64(q.Data[qi*d+h*dh+c]) * h16(k.Data[j*kvDim+kvOff+c])
				}
				scores[j] = s * scale
				maxs = math.Max(maxs, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - maxs)
				sum += scores[j]
			}
			for c := 0; c < dh; c++ {
				var want float64
				for j := 0; j < limit; j++ {
					want += scores[j] / sum * h16(v.Data[j*kvDim+kvOff+c])
				}
				gotv := float64(out.Data[qi*d+h*dh+c])
				if diff := math.Abs(gotv - want); diff > 2e-3*(1+math.Abs(want)) {
					t.Fatalf("query %d head %d chan %d: got %v want %v", qi, h, c, gotv, want)
				}
			}
		}
	}
}

// float16round rounds a float32 through IEEE half precision.
func float16round(x float32) float32 {
	b := math.Float32bits(x)
	sign := b >> 31 << 15
	exp := int32(b>>23&0xff) - 127 + 15
	man := b >> 13 & 0x3ff
	// Round to nearest even on the dropped mantissa bits.
	if b&0x1fff > 0x1000 || (b&0x1fff == 0x1000 && man&1 == 1) {
		man++
		if man == 0x400 {
			man = 0
			exp++
		}
	}
	var h uint16
	switch {
	case int32(b>>23&0xff) == 0xff:
		h = uint16(sign | 0x7c00)
	case exp <= 0:
		h = uint16(sign) // flush tiny values to zero
	case exp >= 31:
		h = uint16(sign | 0x7c00)
	default:
		h = uint16(sign | uint32(exp)<<10 | man)
	}
	// Expand back.
	he := uint32(h>>10) & 0x1f
	hm := uint32(h) & 0x3ff
	hs := uint32(h>>15) << 31
	switch {
	case he == 0 && hm == 0:
		return math.Float32frombits(hs)
	case he == 0x1f:
		return math.Float32frombits(hs | 0x7f800000)
	case he == 0:
		return math.Float32frombits(hs) // subnormals flushed above
	default:
		return math.Float32frombits(hs | (he-15+127)<<23 | hm<<13)
	}
}

// BenchmarkGPUGroupedAttnPrefill measures grouped causal attention at the
// Qwen2.5-0.5B prefill shape: a 512-token chunk against a 625-position
// cache, through the f16 cache path when the device grants shader-f16.
func BenchmarkGPUGroupedAttnPrefill(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(9))
	const heads, kvHeads, dh = 14, 2, 64
	const d, kvDim = heads * dh, kvHeads * dh
	const seqQ, seqKV = 512, 625
	upload := func(f *tensai.Tensor) *Tensor {
		if g.HasF16() {
			c, err := g.NewF16Tensor(seqKV, kvDim)
			if err != nil {
				b.Fatal(err)
			}
			src, err := g.Upload(f)
			if err != nil {
				b.Fatal(err)
			}
			if err := src.CopyRowsInto(c, 0); err != nil {
				b.Fatal(err)
			}
			src.Free()
			return c
		}
		gt, err := g.Upload(f)
		if err != nil {
			b.Fatal(err)
		}
		return gt
	}
	gk := upload(randTensor(rng, seqKV, kvDim))
	defer gk.Free()
	gv := upload(randTensor(rng, seqKV, kvDim))
	defer gv.Free()
	gq, err := g.Upload(randTensor(rng, seqQ, d))
	if err != nil {
		b.Fatal(err)
	}
	defer gq.Free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
}

// TestGPUGroupedCausalAttentionF16Split covers the split decode path: one
// query row over a context long enough (>= 256) to take attn_split_gh +
// attn_reduce_g, checked against the f64 reference over f16-rounded K/V.
func TestGPUGroupedCausalAttentionF16Split(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	if !g.HasF16() {
		t.Skip("device has no shader-f16")
	}
	rng := rand.New(rand.NewSource(17))

	const heads, kvHeads, dh = 14, 2, 64
	const d, kvDim = heads * dh, kvHeads * dh
	group := heads / kvHeads
	// 400 is the decode-bench context; 257 exercises a ragged final tile
	// right past the split threshold.
	for _, seqKV := range []int{257, 400} {
		k := randTensor(rng, seqKV, kvDim)
		v := randTensor(rng, seqKV, kvDim)
		upload16 := func(src *tensai.Tensor) *Tensor {
			t.Helper()
			f32, err := g.Upload(src)
			if err != nil {
				t.Fatal(err)
			}
			defer f32.Free()
			c, err := g.NewF16Tensor(seqKV, kvDim)
			if err != nil {
				t.Fatal(err)
			}
			if err := f32.CopyRowsInto(c, 0); err != nil {
				t.Fatal(err)
			}
			return c
		}
		gk := upload16(k)
		defer gk.Free()
		gv := upload16(v)
		defer gv.Free()
		q := randTensor(rng, 1, d)
		gq, err := g.Upload(q)
		if err != nil {
			t.Fatal(err)
		}
		defer gq.Free()
		got, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer got.Free()
		out, err := got.Download()
		if err != nil {
			t.Fatal(err)
		}

		h16 := func(x float32) float64 {
			return float64(float16round(x))
		}
		scale := 1 / math.Sqrt(dh)
		for h := 0; h < heads; h++ {
			kvOff := (h / group) * dh
			scores := make([]float64, seqKV)
			maxs := math.Inf(-1)
			for j := 0; j < seqKV; j++ {
				var s float64
				for c := 0; c < dh; c++ {
					s += float64(q.Data[h*dh+c]) * h16(k.Data[j*kvDim+kvOff+c])
				}
				scores[j] = s * scale
				maxs = math.Max(maxs, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - maxs)
				sum += scores[j]
			}
			for c := 0; c < dh; c++ {
				var want float64
				for j := 0; j < seqKV; j++ {
					want += scores[j] / sum * h16(v.Data[j*kvDim+kvOff+c])
				}
				gotv := float64(out.Data[h*dh+c])
				if diff := math.Abs(gotv - want); diff > 2e-3*(1+math.Abs(want)) {
					t.Fatalf("seqKV %d head %d chan %d: got %v want %v", seqKV, h, c, gotv, want)
				}
			}
		}
	}
}

// TestGPUCausalAttentionWideHead runs Gemma 4's global-layer shape:
// eight query heads sharing one KV head, 512 wide. A whole group of
// those does not fit the grouped kernel's workgroup memory, so the
// per-head kernel takes it -- which is the widest head it stages.
func TestGPUCausalAttentionWideHead(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewSource(11))

	const heads, kvHeads, dh = 8, 1, 512
	const d, kvDim = heads * dh, kvHeads * dh
	const seq, seqKV = 6, 20
	group := heads / kvHeads
	k, v := randTensor(rng, seqKV, kvDim), randTensor(rng, seqKV, kvDim)
	gk, err := g.Upload(k)
	if err != nil {
		t.Fatal(err)
	}
	defer gk.Free()
	gv, err := g.Upload(v)
	if err != nil {
		t.Fatal(err)
	}
	defer gv.Free()
	q := randTensor(rng, seq, d)
	gq, err := g.Upload(q)
	if err != nil {
		t.Fatal(err)
	}
	defer gq.Free()
	got, err := gq.GroupedCausalAttention(gk, gv, heads, kvHeads, seqKV, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Free()
	out, err := got.Download()
	if err != nil {
		t.Fatal(err)
	}

	scale := 1 / math.Sqrt(dh)
	for qi := 0; qi < seq; qi++ {
		limit := qi + seqKV - seq + 1
		for h := 0; h < heads; h++ {
			kvOff := (h / group) * dh
			scores := make([]float64, limit)
			maxs := math.Inf(-1)
			for j := 0; j < limit; j++ {
				var s float64
				for c := 0; c < dh; c++ {
					s += float64(q.Data[qi*d+h*dh+c]) * float64(k.Data[j*kvDim+kvOff+c])
				}
				s *= scale
				scores[j] = s
				maxs = math.Max(maxs, s)
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - maxs)
				sum += scores[j]
			}
			for c := 0; c < dh; c++ {
				var acc float64
				for j := 0; j < limit; j++ {
					acc += scores[j] / sum * float64(v.Data[j*kvDim+kvOff+c])
				}
				if diff := math.Abs(acc - float64(out.Data[qi*d+h*dh+c])); diff > 2e-4 {
					t.Fatalf("row %d head %d col %d: %v on the device, %v here", qi, h, c, out.Data[qi*d+h*dh+c], acc)
				}
			}
		}
	}
}

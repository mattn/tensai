//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package tensai

import (
	"math"
	"math/rand"
	"testing"
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

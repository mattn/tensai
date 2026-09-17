package llm

import (
	"math"
	"testing"
)

// A grouped RMSNorm is the plain one per group with the matching weight
// slice, and a single group is the plain one over the row.
func TestRMSNormGroups(t *testing.T) {
	x := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	w := []float32{1, 1, 1, 1, 2, 2, 2, 2}
	const eps = 1e-6
	m := &qwen{cfg: config{RMSEps: eps, NormGroups: 2}}
	got := make([]float32, len(x))
	m.rmsnorm(got, x, w)
	// Each half normalized by its own RMS: the second half's values are
	// ten times the first's, so both halves normalize to the same shape
	// and the weight alone tells them apart.
	rms1 := math.Sqrt((1+4+9+16)/4.0 + eps)
	rms2 := math.Sqrt((100+400+900+1600)/4.0 + eps)
	for i := 0; i < 4; i++ {
		want1 := float32(float64(x[i])/rms1) * w[i]
		want2 := float32(float64(x[4+i])/rms2) * w[4+i]
		if got[i] != want1 || got[4+i] != want2 {
			t.Fatalf("lane %d: got %v/%v, want %v/%v", i, got[i], got[4+i], want1, want2)
		}
	}
	// One group is the whole-row norm, whatever NormGroups says short of 2.
	for _, g := range []int{0, 1} {
		m.cfg.NormGroups = g
		whole := make([]float32, len(x))
		m.rmsnorm(whole, x, w)
		ref := make([]float32, len(x))
		rmsnormInto(ref, x, w, eps)
		for i := range ref {
			if whole[i] != ref[i] {
				t.Fatalf("groups=%d lane %d: %v != %v", g, i, whole[i], ref[i])
			}
		}
	}
}

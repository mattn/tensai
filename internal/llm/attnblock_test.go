package llm

import (
	"math"
	"math/rand"
	"testing"
)

// The blocked prefill kernel has to answer exactly what the row-at-a-time
// one does, or a prompt's own tokens see a different model depending on
// where the batch boundary fell. Both the head width and the logit scale
// belong to the layer: gemma4 alternates two widths and leaves its
// logits unscaled, and reading either off the model instead gave its
// global layers one attention for the first rows of a block and another
// for the last seven.
func TestAttendGroupBlockMatchesRows(t *testing.T) {
	const heads, kvHeads, steps0, qb = 8, 1, 11, 8
	group := heads / kvHeads
	rng := rand.New(rand.NewSource(3))
	for _, tt := range []struct {
		name   string
		head   int
		unitQK bool
	}{
		{"scaled logits", 64, false},
		{"gemma4 unscaled logits", 64, true},
		{"gemma4 wide head", 128, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d := tt.head
			m := &qwen{
				cfg:    config{Heads: heads, KVHeads: kvHeads},
				headSz: 32, // deliberately not the layer's width
				blocks: make([]qblock, 1),
			}
			b := &m.blocks[0]
			b.headSz, b.unitQK, b.kvFrom = d, tt.unitQK, -1
			for j := 0; j < steps0+qb; j++ {
				k := make([]float32, kvHeads*d)
				v := make([]float32, kvHeads*d)
				for i := range k {
					k[i] = float32(rng.NormFloat64())
					v[i] = float32(rng.NormFloat64())
				}
				b.kc = append(b.kc, k)
				b.vc = append(b.vc, v)
			}
			qrs := make([][]float32, qb)
			rows := make([][]float32, qb)
			blocked := make([][]float32, qb)
			for r := range qrs {
				q := make([]float32, heads*d)
				for i := range q {
					q[i] = float32(rng.NormFloat64())
				}
				qrs[r] = q
				rows[r] = make([]float32, heads*d)
				blocked[r] = make([]float32, heads*d)
			}

			scores := make([]float32, group*(steps0+qb))
			ws := make([]float32, group*(steps0+qb))
			for r := 0; r < qb; r++ {
				for kh := 0; kh < kvHeads; kh++ {
					m.attendGroup(b, qrs[r], rows[r], kh, group, steps0+r, scores, ws)
				}
			}
			nq := qb * group
			for kh := 0; kh < kvHeads; kh++ {
				qs := make([][]float32, qb)
				as := make([][]float32, qb)
				for r := 0; r < qb; r++ {
					qs[r] = qrs[r][kh*group*d : (kh+1)*group*d]
					as[r] = blocked[r][kh*group*d : (kh+1)*group*d]
				}
				m.attendGroupBlock(b, qs, as, kh, group, steps0,
					make([]float32, (steps0+qb)*nq), make([]float32, steps0+qb),
					make([]float32, nq*d), make([]float32, nq*d))
			}
			for r := 0; r < qb; r++ {
				for i := range rows[r] {
					if math.Abs(float64(rows[r][i]-blocked[r][i])) > 1e-5 {
						t.Fatalf("row %d value %d: %v by rows, %v in a block",
							r, i, rows[r][i], blocked[r][i])
					}
				}
			}
		})
	}
}

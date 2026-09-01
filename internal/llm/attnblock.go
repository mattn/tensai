package llm

import (
	"math"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// attendGroupBlock is attendGroup over a block of consecutive query
// rows. attendGroup streams each cached K and V row once per call, so a
// prefill at group 2 reloads every row for every second dot — the 8-wide
// dot and axpy kernels sit idle. Blocking qb rows lifts the fan-out to
// nq = qb*group dots per K load, which is where the long-context prefill
// spent nine tenths of its time. Row r of the block attends its own
// steps0+r positions: the shared prefix runs through the wide kernels
// and the ragged causal tail runs per row, in the same per-element
// order as attendGroup, so the output stays bit-identical.
//
// qrs[r] is row r's query slice (group heads of this KV head already
// offset out), ars[r] its output slice. ws needs (steps0+qb)*nq floats,
// si steps0+qb, packO nq*headSz; sliding-window layers take the per-row
// path instead.
func (m *qwen) attendGroupBlock(b *qblock, qrs, ars [][]float32, kh, group, steps0 int, ws, si, packQ, packO []float32) {
	// The head width and the logit scale belong to the layer, not to the
	// model: gemma4 alternates two widths and leaves its logits unscaled.
	d := m.headSize(b)
	qb := len(qrs)
	nq := qb * group
	kvOff := kh * d
	scale := float32(m.qkScale(b, d))

	// Shared prefix: every row of the block sees positions [0, steps0).
	for r, qr := range qrs {
		copy(packQ[r*group*d:(r+1)*group*d], qr[:group*d])
	}
	for j := 0; j < steps0; j++ {
		tensai.DotVecs(packQ, b.kc[j][kvOff:kvOff+d], ws[j*nq:(j+1)*nq])
	}
	// The causal tail: row r alone sees positions [steps0, steps0+r].
	for r := 1; r < qb; r++ {
		for j := steps0; j < steps0+r; j++ {
			tensai.DotVecs(packQ[r*group*d:(r+1)*group*d], b.kc[j][kvOff:kvOff+d],
				ws[j*nq+r*group:j*nq+(r+1)*group])
		}
	}

	for r := 0; r < qb; r++ {
		steps := steps0 + r
		for i := 0; i < group; i++ {
			// Same gather-scale-exp-normalize as attendGroup, one
			// (row, head) at a time over that row's own length.
			maxs := float32(math.Inf(-1))
			for j := 0; j < steps; j++ {
				s := ws[j*nq+r*group+i] * scale
				si[j] = s
				if s > maxs {
					maxs = s
				}
			}
			h := kh*group + i
			if b.sinks != nil && b.sinks[h] > maxs {
				maxs = b.sinks[h]
			}
			kernels.ExpShift(si[:steps], si[:steps], maxs)
			var sum float32
			for _, v := range si[:steps] {
				sum += v
			}
			if b.sinks != nil {
				sum += kernels.ExpF(b.sinks[h] - maxs)
			}
			inv := 1 / sum
			for j := 0; j < steps; j++ {
				ws[j*nq+r*group+i] = si[j] * inv
			}
		}
	}

	// Value accumulation, one V load for the whole block on the shared
	// prefix. The tail entries of ws for rows past their own limit were
	// never written, and are never read either: the tail loops below run
	// exactly each row's own positions.
	for i := range packO[:nq*d] {
		packO[i] = 0
	}
	for j := 0; j < steps0; j++ {
		tensai.Axpys(ws[j*nq:(j+1)*nq], b.vc[j][kvOff:kvOff+d], packO[:nq*d])
	}
	for r := 1; r < qb; r++ {
		for j := steps0; j < steps0+r; j++ {
			tensai.Axpys(ws[j*nq+r*group:j*nq+(r+1)*group], b.vc[j][kvOff:kvOff+d],
				packO[r*group*d:(r+1)*group*d])
		}
	}
	for r, ar := range ars {
		copy(ar[:group*d], packO[r*group*d:(r+1)*group*d])
	}
}

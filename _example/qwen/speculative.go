package main

import (
	"math"
	"math/rand"
	"sort"
)

type specStats struct {
	accepted int
	proposed int
}

// sampleDistribution returns the exact temperature/top-p distribution used by
// speculative rejection sampling. Keeping the zeroed tail is intentional: it
// makes max(0, p-q) cheap on a rejected proposal.
func sampleDistribution(logits []float32, temp, topP float64) []float64 {
	p := make([]float64, len(logits))
	buckets := make([]uint16, len(logits))
	maxl := logits[0]
	for _, v := range logits[1:] {
		if v > maxl {
			maxl = v
		}
	}
	var sum float64
	const nb = 1100
	var bsum [nb]float64
	for i, v := range logits {
		p[i] = math.Exp(float64(v-maxl) / temp)
		sum += p[i]
		if topP < 1 && p[i] > 0 {
			b := -math.Ilogb(p[i])
			if b < 0 {
				b = 0
			} else if b >= nb {
				b = nb - 1
			}
			buckets[i] = uint16(b)
			bsum[b] += p[i]
		}
	}
	if topP < 1 {
		target := topP * sum
		var mass float64
		cut := 0
		for cut < nb-1 && mass+bsum[cut] < target {
			mass += bsum[cut]
			cut++
		}
		ids := make([]int, 0, len(p)/100)
		for i, b := range buckets {
			if int(b) <= cut && p[i] > 0 {
				ids = append(ids, i)
			}
		}
		sort.Slice(ids, func(i, j int) bool { return p[ids[i]] > p[ids[j]] })
		mass = 0
		n := 0
		for n < len(ids) && (n == 0 || mass < target) {
			mass += p[ids[n]]
			n++
		}
		for i := range p {
			p[i] = 0
		}
		for _, id := range ids[:n] {
			p[id] = math.Exp(float64(logits[id]-maxl) / temp)
		}
		sum = mass
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

func drawDistribution(p []float64, rng *rand.Rand) int {
	r := rng.Float64()
	for i, v := range p {
		r -= v
		if r <= 0 {
			return i
		}
	}
	return len(p) - 1
}

// generateSpeculative implements greedy verification at temp <= 0 and the
// standard exact speculative-sampling rejection rule otherwise. emit returns
// false to abort (for example when an HTTP client disconnects).
func generateSpeculative(target, draft *qwen, logits []float32, steps, limit, nCtx, specK int,
	temp, topP float64, stop func(int) bool, rng *rand.Rand, emit func(int) bool,
) ([]float32, int, string, specStats) {
	stats := specStats{}
	generated := 0
	for generated < limit && steps < nCtx-1 {
		if specK < 1 || steps >= nCtx-1-specK {
			next := sample(logits, temp, topP, rng)
			if stop(next) {
				return logits, steps, "stop", stats
			}
			if !emit(next) {
				return logits, steps, "abort", stats
			}
			logits = target.step(next, steps)
			draft.step(next, steps)
			steps++
			generated++
			continue
		}

		first := sample(logits, temp, topP, rng)
		if stop(first) {
			return logits, steps, "stop", stats
		}
		if !emit(first) {
			return logits, steps, "abort", stats
		}
		generated++
		props := make([]int, 0, specK)
		qdists := make([][]float64, 0, specK)
		dl := draft.step(first, steps)
		for i := 0; i < specK; i++ {
			if temp <= 0 {
				props = append(props, sample(dl, 0, 1, rng))
			} else {
				q := sampleDistribution(dl, temp, topP)
				qdists = append(qdists, q)
				props = append(props, drawDistribution(q, rng))
			}
			dl = draft.step(props[i], steps+1+i)
		}
		lm := target.prefillLogits(append([]int{first}, props...), steps)
		accepted := 0
		for accepted < len(props) && generated < limit {
			row := lm.Data[accepted*lm.Cols : (accepted+1)*lm.Cols]
			if temp <= 0 {
				if sample(row, 0, 1, rng) != props[accepted] {
					break
				}
			} else {
				p := sampleDistribution(row, temp, topP)
				q := qdists[accepted]
				ratio := 1.0
				if q[props[accepted]] > 0 {
					ratio = p[props[accepted]] / q[props[accepted]]
				}
				if rng.Float64() > math.Min(1, ratio) {
					for i := range p {
						p[i] = math.Max(0, p[i]-q[i])
					}
					var mass float64
					for _, v := range p {
						mass += v
					}
					for i := range p {
						p[i] /= mass
					}
					correction := drawDistribution(p, rng)
					target.truncate(steps + 1 + accepted)
					draft.truncate(steps + 1 + accepted)
					steps += 1 + accepted
					stats.accepted += accepted
					stats.proposed += len(props)
					if stop(correction) {
						return logits, steps, "stop", stats
					}
					if !emit(correction) {
						return logits, steps, "abort", stats
					}
					generated++
					logits = target.step(correction, steps)
					draft.step(correction, steps)
					steps++
					accepted = -1
					break
				}
			}
			if stop(props[accepted]) {
				accepted++
				target.truncate(steps + 1 + accepted)
				draft.truncate(steps + 1 + accepted)
				steps += 1 + accepted
				stats.accepted += accepted
				stats.proposed += len(props)
				return logits, steps, "stop", stats
			}
			if !emit(props[accepted]) {
				return logits, steps, "abort", stats
			}
			generated++
			accepted++
		}
		if accepted < 0 {
			continue
		}
		stats.accepted += accepted
		stats.proposed += len(props)
		target.truncate(steps + 1 + accepted)
		draft.truncate(steps + 1 + accepted)
		steps += 1 + accepted
		logits = lm.Data[accepted*lm.Cols : (accepted+1)*lm.Cols]
	}
	return logits, steps, "length", stats
}

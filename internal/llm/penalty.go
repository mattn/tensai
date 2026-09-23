package llm

// penalty is the repetition control a run carries, in the two
// conventions callers know. Repeat is llama.cpp's: every token seen in
// the last LastN positions, prompt included, has its logit divided by
// Repeat when positive and multiplied by it when negative, so 1.1 is a
// nudge and 1.0 is off. Presence and Frequency are OpenAI's: a token
// generated in this completion loses Presence once and Frequency per
// occurrence, so either can also be negative to invite repetition.
//
// Both act on the logits the sampler sees and nothing else: greedy
// decoding is steered by them too, which is where a small model looping
// on one phrase most needs it.
type penalty struct {
	Repeat    float64
	LastN     int
	Presence  float64
	Frequency float64
}

func (p penalty) active() bool {
	return (p.Repeat != 0 && p.Repeat != 1) || p.Presence != 0 || p.Frequency != 0
}

// penaltyState follows a generation: the recent window for the repeat
// penalty and the per-token counts for presence and frequency. The
// window is fed the prompt too, the counts only what was generated.
type penaltyState struct {
	p      penalty
	window []int
	counts map[int]int
}

func newPenaltyState(p penalty, prompt []int) *penaltyState {
	if !p.active() {
		return nil
	}
	if p.LastN <= 0 {
		p.LastN = 64
	}
	s := &penaltyState{p: p, counts: map[int]int{}}
	s.push(prompt, false)
	return s
}

// push records tokens; generated says whether they count for presence
// and frequency as well as for the window.
func (s *penaltyState) push(ids []int, generated bool) {
	if s == nil {
		return
	}
	for _, id := range ids {
		if generated {
			s.counts[id]++
		}
	}
	s.window = append(s.window, ids...)
	if n := len(s.window) - s.p.LastN; n > 0 {
		s.window = s.window[n:]
	}
}

// apply adjusts logits in place for the next sample.
func (s *penaltyState) apply(logits []float32) {
	if s == nil {
		return
	}
	if r := s.p.Repeat; r != 0 && r != 1 {
		f := float32(r)
		for _, id := range s.window {
			if id < 0 || id >= len(logits) {
				continue
			}
			if logits[id] > 0 {
				logits[id] /= f
			} else {
				logits[id] *= f
			}
		}
	}
	if s.p.Presence != 0 || s.p.Frequency != 0 {
		for id, n := range s.counts {
			if id < 0 || id >= len(logits) {
				continue
			}
			logits[id] -= float32(s.p.Frequency)*float32(n) + float32(s.p.Presence)
		}
	}
}

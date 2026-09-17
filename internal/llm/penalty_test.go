package llm

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestPenaltyState(t *testing.T) {
	if newPenaltyState(penalty{Repeat: 1}, nil) != nil {
		t.Fatal("a repeat of 1 is off and should cost nothing")
	}
	// The window holds the last N of prompt and answer alike; the counts
	// only what was generated.
	s := newPenaltyState(penalty{Repeat: 2, LastN: 3, Presence: 1, Frequency: 0.5}, []int{7, 8, 9, 1})
	s.push([]int{2}, true)
	s.push([]int{2}, true)
	if got := s.window; len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 2 {
		t.Fatalf("window = %v, want [1 2 2]", got)
	}
	if s.counts[2] != 2 || s.counts[1] != 0 {
		t.Fatalf("counts = %v", s.counts)
	}
	// Repeat divides a positive logit and multiplies a negative one, once
	// per position in the window; presence and frequency subtract by count.
	logits := []float32{4, 4, -4, 4}
	s.apply(logits)
	// id 1: in the window once, never generated: 4/2.
	// id 2: in the window twice, generated twice: -4*2*2 - (0.5*2 + 1).
	// id 3: untouched. id 0: untouched.
	want := []float32{4, 2, -16 - 2, 4}
	for i := range want {
		if logits[i] != want[i] {
			t.Fatalf("logits = %v, want %v", logits, want)
		}
	}
	// A nil state is a no-op everywhere.
	var none *penaltyState
	none.push([]int{1}, true)
	none.apply(logits)
}

// A request's frequency_penalty reaches the sampler: a model that would
// greedily repeat one token forever alternates once repeating it costs.
func TestServeFrequencyPenalty(t *testing.T) {
	tok := scriptedTok{pieces: []string{"a", "b"}}
	s := &server{tok: tok, nCtx: 4096, imEnd: 2, eot: 2, tm: templateFor("qwen2", false), reset: func() {}}
	fixed := func() []float32 { return []float32{2, 1.5, -100} }
	s.prefill = func([]int, int) []float32 { return fixed() }
	s.step = func(int, int) []float32 { return fixed() }
	content := func(body string) string {
		w := post(t, s, body)
		if w.Code != http.StatusOK {
			t.Fatalf("status %d: %s", w.Code, w.Body.String())
		}
		var r struct {
			Choices []struct {
				Message struct{ Content string } `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		return r.Choices[0].Message.Content
	}
	if got := content(`{"messages":[{"role":"user","content":"x"}],"max_tokens":4}`); got != "aaaa" {
		t.Fatalf("without a penalty: %q, want aaaa", got)
	}
	if got := content(`{"messages":[{"role":"user","content":"x"}],"max_tokens":4,"frequency_penalty":1.0}`); got != "abab" {
		t.Fatalf("with frequency_penalty 1: %q, want abab", got)
	}
	// repetition_penalty looks back over the prompt as well, and the
	// scripted prompt is token 0: so "a" is halved before the first
	// sample and "b" leads, then they alternate.
	if got := content(`{"messages":[{"role":"user","content":"x"}],"max_tokens":4,"repetition_penalty":2.0}`); got != "baba" {
		t.Fatalf("with repetition_penalty 2: %q, want baba", got)
	}
}

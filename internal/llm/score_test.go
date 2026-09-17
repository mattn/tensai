package llm

import (
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLogProbSoftmax(t *testing.T) {
	logits := []float32{1, 2, 3, 4}
	var sum float64
	for id := range logits {
		p := math.Exp(logProb(logits, id))
		if p <= 0 || p > 1 {
			t.Fatalf("id %d: probability %v", id, p)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("probabilities sum to %v", sum)
	}
	// The largest logit gets the largest probability, and an id outside
	// the vocabulary gets none.
	if logProb(logits, 3) <= logProb(logits, 2) {
		t.Fatal("logProb is not monotone in the logit")
	}
	if !math.IsInf(logProb(logits, 4), -1) {
		t.Fatal("out-of-range id did not score -Inf")
	}
	// softmax64 is shift-invariant and normalizes.
	a := softmax64([]float64{0, 1, 2})
	b := softmax64([]float64{100, 101, 102})
	var total float64
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-12 {
			t.Fatalf("softmax64 not shift-invariant at %d: %v vs %v", i, a[i], b[i])
		}
		total += a[i]
	}
	if math.Abs(total-1) > 1e-12 {
		t.Fatalf("softmax64 sums to %v", total)
	}
}

// Score against a real model: for single-token options its answer is the
// model's own next-token distribution restricted to those tokens, it
// does not depend on the order the options come in, and the rollback
// between options leaves nothing behind, so asking twice agrees exactly.
// Skips without the small Qwen the other model tests use.
func TestScoreAgainstModel(t *testing.T) {
	dir := filepath.Join(CacheRoot(), "Qwen2.5-0.5B-Instruct")
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no %s", dir)
	}
	e, err := Open(Options{Data: dir, Bits: 8, Log: io.Discard})
	if err != nil {
		t.Skip(err)
	}
	defer e.Close()
	const q = "What color is a clear daytime sky? Answer with one word."
	opts := []string{"blue", "green", "red"}
	probs, err := e.Score(q, opts)
	if err != nil {
		t.Fatal(err)
	}
	if probs[0] < 0.5 {
		t.Fatalf("blue scored %.3f, expected the majority", probs[0])
	}
	// Reversed options give the same numbers back in reversed places.
	rev, err := e.Score(q, []string{"red", "green", "blue"})
	if err != nil {
		t.Fatal(err)
	}
	for i := range opts {
		if math.Abs(probs[i]-rev[len(opts)-1-i]) > 1e-9 {
			t.Fatalf("option order changed the answer: %v vs %v", probs, rev)
		}
	}
	// And the restricted next-token distribution says the same thing:
	// score the prompt's logits directly.
	text := e.tm.bos + e.systemTurn() + e.tm.userOpen + q + e.tm.userClose + e.tm.asstOpen + e.tm.asstPrefill
	e.Reset()
	logits := e.prefill(e.tok.Encode(text), 0)
	ll := make([]float64, len(opts))
	for i, o := range opts {
		ids := e.tok.Encode(o)
		if len(ids) != 1 {
			t.Skipf("%q is %d tokens here; the direct check wants one", o, len(ids))
		}
		ll[i] = logProb(logits, ids[0])
	}
	direct := softmax64(ll)
	for i := range opts {
		if math.Abs(probs[i]-direct[i]) > 1e-6 {
			t.Fatalf("Score %v differs from the next-token distribution %v", probs, direct)
		}
	}
}

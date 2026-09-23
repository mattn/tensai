package llm

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
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

func TestScorePrefill(t *testing.T) {
	// No thinking block: the prefill as it is.
	if got := scorePrefill(tmpl{asstPrefill: "x"}, false); got != "x" {
		t.Fatalf("plain template: %q", got)
	}
	// A gemma4 opens and closes its channel before the answer.
	g4 := tmpl{reasonOpen: "<|channel>thought\n", reasonClose: "<channel|>"}
	if got := scorePrefill(g4, false); got != "<|channel>thought\n<channel|>" {
		t.Fatalf("gemma4: %q", got)
	}
	// Asked to think, the block is the model's to fill.
	if got := scorePrefill(g4, true); got != "" {
		t.Fatalf("gemma4 thinking: %q", got)
	}
	// A template that already closes an empty block is left alone.
	q3 := tmpl{reasonOpen: "<think>", reasonClose: "</think>", asstPrefill: "<think>\n\n</think>\n\n"}
	if got := scorePrefill(q3, false); got != q3.asstPrefill {
		t.Fatalf("qwen3: %q", got)
	}
}

func TestRenderLabels(t *testing.T) {
	got := renderLabels([]string{"blue", "green"})
	want := "\nA. blue\nB. green\nAnswer with the letter only."
	if got != want {
		t.Fatalf("renderLabels = %q, want %q", got, want)
	}
	if len(labels) != 26 {
		t.Fatalf("%d labels", len(labels))
	}
}

// ScoreMany against a real model: questions sharing a state prefill it
// once and extend it, and each answer is what Score gives the same
// question asked alone with the state ahead of it. Skips without the
// small Qwen.
func TestScoreManyAgainstModel(t *testing.T) {
	dir := filepath.Join(CacheRoot(), "Qwen2.5-0.5B-Instruct")
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no %s", dir)
	}
	var log bytes.Buffer
	e, err := Open(Options{Data: dir, Bits: 8, Log: &log})
	if err != nil {
		t.Skip(err)
	}
	defer e.Close()
	const state = "It is a clear afternoon in July and the sun is out."
	qs := []Question{
		{Text: "What color is the sky?", Options: []string{"blue", "green", "red"}},
		{Text: "Is it raining? Answer yes or no.", Options: []string{"yes", "no"}},
	}
	log.Reset()
	res, err := e.ScoreMany(state, qs, false)
	if err != nil {
		t.Fatal(err)
	}
	many := res.Probs
	// The second question reused the state: fewer tokens prefilled
	// than it has.
	var total, done int
	if _, err := fmt.Sscanf(lastLine(log.String(), "question 2:"), "question 2: %d tokens, %d prefilled", &total, &done); err != nil {
		t.Fatalf("no question 2 line in %q", log.String())
	}
	if done >= total {
		t.Fatalf("question 2 prefilled %d of %d tokens; the state was not reused", done, total)
	}
	for i, q := range qs {
		alone, err := e.Score(state+"\n\n"+q.Text, q.Options)
		if err != nil {
			t.Fatal(err)
		}
		for j := range alone {
			if math.Abs(many[i][j]-alone[j]) > 1e-6 {
				t.Fatalf("question %d: batched %v, alone %v", i, many[i], alone)
			}
		}
	}
	if many[0][0] < 0.5 || many[1][1] < 0.5 {
		t.Fatalf("unexpected answers %v", many)
	}
	// Labeled, the letter of the right color wins. (The yes/no question
	// is left out: a 0.5B leans on A whatever the question, which is
	// the model's habit and not the mechanism's.)
	lres, err := e.ScoreMany(state, qs, true)
	if err != nil {
		t.Fatal(err)
	}
	if labeled := lres.Probs; labeled[0][0] < 0.5 {
		t.Fatalf("unexpected labeled answer %v", labeled[0])
	}
	// Five labels were scored, one token each, and the prompt tokens
	// are what the log said was prefilled.
	if lres.OptionTokens != 5 {
		t.Fatalf("labeled run scored %d option tokens, want 5", lres.OptionTokens)
	}
	if lres.PromptTokens <= total {
		t.Fatalf("prompt tokens %d, but question 2 alone is %d", lres.PromptTokens, total)
	}
}

// lastLine is the last line of s starting with prefix, or "".
func lastLine(s, prefix string) string {
	var out string
	for _, l := range strings.Split(s, "\n") {
		if strings.HasPrefix(l, prefix) {
			out = l
		}
	}
	return out
}

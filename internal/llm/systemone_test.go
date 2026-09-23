package llm

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestPlanQuestion(t *testing.T) {
	q := func(s string) SystemOneQuestion {
		var out SystemOneQuestion
		if err := json.Unmarshal([]byte(s), &out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	// A noul lists yes and no, described when the criteria say how.
	p, err := planQuestion(q(`{"type":"noul","instructions":"Urgent?","criteria":{"true":"time-sensitive","false":"no rush"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.question.Text != "Urgent?" || len(p.question.Options) != 2 ||
		p.question.Options[0] != "yes: time-sensitive" || p.question.Options[1] != "no: no rush" {
		t.Fatalf("noul planned as %+v", p.question)
	}
	// Without criteria the names stand alone.
	p, err = planQuestion(q(`{"type":"noul","instructions":"Urgent?"}`))
	if err != nil || p.question.Options[0] != "yes" || p.question.Options[1] != "no" {
		t.Fatalf("bare noul planned as %+v, %v", p.question, err)
	}
	// A choice's options come in name order, whatever the JSON's, and
	// structured criteria show as JSON.
	p, err = planQuestion(q(`{"type":"choice","instructions":["Which team?","Pick one."],"criteria":{"technical":{"covers":"bugs"},"billing":"payments"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.question.Text != `["Which team?","Pick one."]` {
		t.Fatalf("structured instructions rendered as %q", p.question.Text)
	}
	if p.names[0] != "billing" || p.names[1] != "technical" ||
		p.question.Options[0] != "billing: payments" || p.question.Options[1] != `technical: {"covers":"bugs"}` {
		t.Fatalf("choice planned as %v / %v", p.names, p.question.Options)
	}
	// A score numbers its levels from zero and remembers the legend.
	p, err = planQuestion(q(`{"type":"score","instructions":"How bad?","criteria":["fine","bad","awful"]}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.question.Options[2] != "2: awful" || p.legend["1"] != "bad" || p.names[0] != "0" {
		t.Fatalf("score planned as %+v", p)
	}
	// What is refused.
	for _, bad := range []string{
		`{"type":"noul"}`,
		`{"type":"choice","instructions":"?","criteria":{"one":""}}`,
		`{"type":"score","instructions":"?","criteria":["one"]}`,
		`{"type":"score","instructions":"?","criteria":["a","b","c","d","e","f","g","h","i","j","k"]}`,
		`{"type":"rank","instructions":"?"}`,
	} {
		if _, err := planQuestion(q(bad)); err == nil {
			t.Errorf("%s was accepted", bad)
		}
	}
}

func TestSystemOneAnswer(t *testing.T) {
	p := systemOnePlan{typ: "noul", names: []string{"yes", "no"}}
	a := p.answer([]float64{0.9, 0.1})
	if a.Noul == nil || *a.Noul != 0.9 || a.Confidence != nil || a.Probabilities != nil {
		t.Fatalf("noul answered %+v", a)
	}
	p = systemOnePlan{typ: "choice", names: []string{"billing", "sales", "technical"}}
	a = p.answer([]float64{0.84, 0.001, 0.159})
	if a.Choice != "billing" || a.Probabilities["technical"] != 0.159 {
		t.Fatalf("choice answered %+v", a)
	}
	// Jev's published example: these probabilities, confidence 0.596.
	if math.Abs(*a.Confidence-0.596) > 0.005 {
		t.Fatalf("confidence %.3f, Jev says 0.596", *a.Confidence)
	}
	p = systemOnePlan{typ: "score", names: []string{"0", "1", "2"}, legend: map[string]string{"0": "calm", "1": "cross", "2": "angry"}}
	a = p.answer([]float64{0.1, 0.8, 0.1})
	if a.Score == nil || math.Abs(*a.Score-1) > 1e-12 || a.Legend["2"] != "angry" || a.Probabilities["1"] != 0.8 {
		t.Fatalf("score answered %+v", a)
	}
	// All the mass on one option is full confidence; an even spread
	// is none.
	if c := confidence([]float64{1, 0, 0}); c != 1 {
		t.Fatalf("certain answer has confidence %v", c)
	}
	if c := confidence([]float64{0.25, 0.25, 0.25, 0.25}); c != 0 {
		t.Fatalf("even spread has confidence %v", c)
	}
}

// The endpoint against a real model: the Jev quickstart request comes
// back in Jev's shape, with every answer typed as asked, and the chat
// endpoint still works after it. Skips without the small Qwen.
func TestSystemOneEndpoint(t *testing.T) {
	dir := filepath.Join(CacheRoot(), "Qwen2.5-0.5B-Instruct")
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no %s", dir)
	}
	e, err := Open(Options{Data: dir, Bits: 8, Log: io.Discard})
	if err != nil {
		t.Skip(err)
	}
	defer e.Close()
	s := &server{
		engine: e, model: e.model, tok: e.tok, system: e.system, nCtx: e.nCtx,
		imEnd: e.imEnd, eot: e.eot, tm: e.tm, prefill: e.prefill, step: e.step, reset: e.reset,
	}
	s.cache.enabled = true
	const body = `{
		"state": "Help! My payouts have been failing for 3 days.",
		"questions": {
			"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
			"department": {"type": "choice", "instructions": "Which team should handle this?",
				"criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}},
			"frustration": {"type": "score", "instructions": "How frustrated is the customer?",
				"criteria": ["Calm", "Frustrated", "Very angry"]}
		}
	}`
	rec := httptest.NewRecorder()
	s.systemOne(rec, httptest.NewRequest(http.MethodPost, "/v1/systemone", bytes.NewBufferString(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var resp SystemOneResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != "tensai" || len(resp.Answers) != 3 || resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens != 8 {
		t.Fatalf("response %+v", resp)
	}
	if a := resp.Answers["is_urgent"]; a.Type != "noul" || a.Noul == nil || *a.Noul < 0.5 {
		t.Fatalf("is_urgent answered %+v", a)
	}
	if a := resp.Answers["department"]; a.Type != "choice" || a.Choice == "" || len(a.Probabilities) != 3 || a.Confidence == nil {
		t.Fatalf("department answered %+v", a)
	}
	if a := resp.Answers["frustration"]; a.Type != "score" || a.Score == nil || *a.Score < 0 || *a.Score > 2 || len(a.Legend) != 3 {
		t.Fatalf("frustration answered %+v", a)
	}
	// A request with nothing to answer is refused, not served.
	rec = httptest.NewRecorder()
	s.systemOne(rec, httptest.NewRequest(http.MethodPost, "/v1/systemone", bytes.NewBufferString(`{"state": "x"}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("empty request got %d", rec.Code)
	}
	// The chat endpoint finds no cache to trust and starts clean.
	if s.cache.live != nil || s.cache.ckpt != nil {
		t.Fatal("prompt cache survived a systemone call")
	}
	rec = httptest.NewRecorder()
	s.chatCompletions(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":4}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("chat after systemone: %d %s", rec.Code, rec.Body)
	}
}

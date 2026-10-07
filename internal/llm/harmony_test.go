package llm

import (
	"strings"
	"testing"
	"time"
)

// gpt-oss keeps the system block tensai always sent it; a harmony
// template that names its own identity (LLM-jp-4) gets that identity,
// its cutoff and the date its template writes.
func TestHarmonySystem(t *testing.T) {
	now := time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)
	const gptOSS = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."
	for _, tpl := range []string{
		"",
		`{%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %} conversation_start_date`,
	} {
		if got := harmonySystem(tpl, now); got != gptOSS {
			t.Errorf("gpt-oss block changed:\n%q", got)
		}
	}
	llmjp := `{%- set model_identity = "You are LLM-jp-4, a large language model trained by LLM-jp." %}
{%- set knowledge_cutoff = "2025-12" %} {%- set conversation_start_date = strftime_now("%Y-%m-%d") %}`
	want := "You are LLM-jp-4, a large language model trained by LLM-jp.\nKnowledge cutoff: 2025-12\nCurrent date: 2026-10-07\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."
	if got := harmonySystem(llmjp, now); got != want {
		t.Errorf("LLM-jp block:\n%q\nwant\n%q", got, want)
	}
}

// The harmony analysis channel is reasoning: run shows only the final
// channel's text, whether the model reasoned first or answered straight
// away, however the stream happens to be cut into tokens.
func TestHarmonyHidesAnalysis(t *testing.T) {
	tm := templateFor("gpt-oss", false)
	for _, c := range []struct {
		pieces []string
		reason string
	}{
		{[]string{"<|channel|>", "analysis", "<|message|>", "Need answer.", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "東京", "です。"}, "Need answer."},
		{[]string{"<|channel|>", "final", "<|message|>", "東京", "です。"}, ""},
		{[]string{"<|channel|>final<|message|>東京です。"}, ""},
	} {
		var out strings.Builder
		f := &thoughtFilter{w: &out, open: tm.reasonOpen, close: tm.reasonClose, answer: tm.answerOpen}
		for _, p := range c.pieces {
			f.Write([]byte(p))
		}
		f.flush()
		if out.String() != "東京です。" {
			t.Errorf("%q streamed as %q", c.pieces, out.String())
		}
		reason, rest := tm.split(strings.Join(c.pieces, ""))
		if reason != c.reason || rest != "東京です。" {
			t.Errorf("%q split into %q and %q", c.pieces, reason, rest)
		}
	}
	// A family without an answer header (Qwen3's <think>) passes the
	// text after its block through untouched.
	var out strings.Builder
	f := &thoughtFilter{w: &out, open: "<think>", close: "</think>"}
	f.Write([]byte("<think>hmm</think><|channel|>final<|message|>x"))
	f.flush()
	if out.String() != "<|channel|>final<|message|>x" {
		t.Errorf("qwen3 text changed: %q", out.String())
	}
}

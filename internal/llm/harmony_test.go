package llm

import (
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

package llm

import "testing"

// LLM-jp-3's instruction template, rendered by Hugging Face for a system
// message, a finished exchange and a new question: the system sentence
// is fixed, the markers are plain text, and an answer closes with </s>.
func TestLLMJP3Render(t *testing.T) {
	msgs := []chatMessage{
		{Role: "system", Content: llmjp3System},
		{Role: "user", Content: "日本の首都は？"},
		{Role: "assistant", Content: "東京です。"},
		{Role: "user", Content: "人口は？"},
	}
	want := "<s>以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。" +
		"\n\n### 指示:\n日本の首都は？\n\n### 応答:\n東京です。</s>" +
		"\n\n### 指示:\n人口は？\n\n### 応答:\n"
	if got := render(templateFor("llm-jp-3", false), msgs, llmjp3System, nil); got != want {
		t.Errorf("got\n%q\nwant\n%q", got, want)
	}
}

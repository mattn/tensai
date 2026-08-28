package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// scriptedTok is a tokenizer over a fixed script: token i decodes to
// piece i, and the last id is the end-of-turn the server stops on.
type scriptedTok struct{ pieces []string }

func (t scriptedTok) Encode(s string) []int { return []int{0} }
func (t scriptedTok) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.pieces) {
			b.WriteString(t.pieces[id])
		}
	}
	return b.String()
}

// scriptedServer replays pieces one token at a time, which is what
// exercises the split: a marker can arrive in fragments.
func scriptedServer(t *testing.T, tm tmpl, pieces []string) *server {
	t.Helper()
	tok := scriptedTok{pieces: pieces}
	end := len(pieces)
	next := 0
	oneHot := func(id int) []float32 {
		l := make([]float32, end+1)
		l[id] = 1
		return l
	}
	s := &server{
		tok: tok, nCtx: 4096, imEnd: end, eot: end, tm: tm,
		reset: func() { next = 0 },
	}
	s.prefill = func([]int, int) []float32 {
		id := next
		next++
		return oneHot(id)
	}
	s.step = func(int, int) []float32 {
		if next >= end {
			return oneHot(end)
		}
		id := next
		next++
		return oneHot(id)
	}
	return s
}

func post(t *testing.T, s *server, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	s.chatCompletions(w, r)
	return w
}

func thinkingTmpl() tmpl { return templateFor("qwen3", true) }

func TestReasoningSplitNonStreaming(t *testing.T) {
	// The markers arrive split across tokens, as a byte-level BPE gives them.
	s := scriptedServer(t, thinkingTmpl(), []string{
		"<th", "ink>", "\n17*3", " is 51.\n", "</think", ">", "\n\nIt is ", "51.",
	})
	w := post(t, s, `{"messages":[{"role":"user","content":"17*3?"}]}`)
	var got struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("%v: %s", err, w.Body)
	}
	m := got.Choices[0].Message
	if m.Reasoning != "17*3 is 51." {
		t.Errorf("reasoning_content = %q, want the thinking only", m.Reasoning)
	}
	if m.Content != "It is 51." {
		t.Errorf("content = %q, want the answer only", m.Content)
	}
	if strings.Contains(m.Content, "<think") {
		t.Errorf("the marker leaked into content: %q", m.Content)
	}
}

// Deltas must carry reasoning and answer in their own fields, and no
// fragment of a marker may reach the client in either.
func TestReasoningSplitStreaming(t *testing.T) {
	s := scriptedServer(t, thinkingTmpl(), []string{
		"<th", "ink>", "weighing", " it", "</thi", "nk>", "done", ".",
	})
	w := post(t, s, `{"stream":true,"messages":[{"role":"user","content":"?"}]}`)
	var reason, content strings.Builder
	for _, line := range strings.Split(w.Body.String(), "\n") {
		line = strings.TrimPrefix(line, "data: ")
		if line == "" || line == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			t.Fatalf("%v: %s", err, line)
		}
		reason.WriteString(chunk.Choices[0].Delta.Reasoning)
		content.WriteString(chunk.Choices[0].Delta.Content)
	}
	if reason.String() != "weighing it" {
		t.Errorf("streamed reasoning = %q", reason.String())
	}
	if content.String() != "done." {
		t.Errorf("streamed content = %q", content.String())
	}
	for _, bad := range []string{"<th", "think", "</th"} {
		if strings.Contains(content.String(), bad) || strings.Contains(reason.String(), bad) {
			t.Errorf("a marker fragment %q leaked into the stream", bad)
		}
	}
}

// A model that never opens a block, or a family that does not think,
// must stream exactly what it said.
func TestReasoningAbsent(t *testing.T) {
	for _, tt := range []struct {
		name string
		tm   tmpl
	}{
		{"thinking family, no block", thinkingTmpl()},
		{"non-thinking family", templateFor("qwen2", false)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := scriptedServer(t, tt.tm, []string{"just", " an ", "answer"})
			w := post(t, s, `{"messages":[{"role":"user","content":"?"}]}`)
			var got map[string]any
			json.Unmarshal(w.Body.Bytes(), &got)
			msg := got["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
			if msg["content"] != "just an answer" {
				t.Errorf("content = %v", msg["content"])
			}
			if _, ok := msg["reasoning_content"]; ok {
				t.Errorf("a turn with no thinking grew a reasoning_content field")
			}
		})
	}
}

func TestSplitReasoning(t *testing.T) {
	for _, tt := range []struct{ in, reason, rest string }{
		{"<think>a</think>b", "a", "b"},
		{"no block here", "", "no block here"},
		// Cut short mid-block: it is all reasoning, and saying so beats
		// presenting half a thought as the answer.
		{"<think>unfinished", "unfinished", ""},
		{"lead <think>a</think> tail", "a", "lead  tail"},
	} {
		reason, rest := splitReasoning(tt.in, "<think>", "</think>")
		if reason != tt.reason || rest != tt.rest {
			t.Errorf("splitReasoning(%q) = (%q, %q), want (%q, %q)", tt.in, reason, rest, tt.reason, tt.rest)
		}
	}
}

// Reasoning never goes back into the prompt: the model's own template
// drops it from history.
func TestRenderDropsReasoningFromHistory(t *testing.T) {
	got := render(thinkingTmpl(), []chatMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "<think>pondering</think>hello"},
		{Role: "user", Content: "again"},
	}, "sys", nil)
	if strings.Contains(got, "pondering") || strings.Contains(got, "<think>") {
		t.Errorf("replayed history carried the thinking back in:\n%s", got)
	}
	if !strings.Contains(got, "hello") {
		t.Errorf("the answer was dropped along with the thinking:\n%s", got)
	}
}

package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func chatml() tmpl { return templateFor("qwen2", false) }

func weatherTool() toolDef {
	return toolDef{Type: "function", Function: toolFunc{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
	}}
}

func TestParseToolCalls(t *testing.T) {
	for _, tt := range []struct {
		name  string
		in    string
		text  string
		calls []string // "name(arguments)"
	}{
		{"plain text", "The capital is Paris.", "The capital is Paris.", nil},
		{"one call",
			"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>",
			"", []string{`get_weather({"city": "Tokyo"})`}},
		{"text then call",
			"Let me look.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
			"Let me look.", []string{"f({})"}},
		{"two calls",
			"<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>",
			"", []string{`a({"x": 1})`, "b({})"}},
		{"missing arguments become an empty object",
			`<tool_call>{"name": "a"}</tool_call>`, "", []string{"a({})"}},
		// Running out of tokens mid-block must not drop the call.
		{"unterminated", `<tool_call>{"name": "a", "arguments": {}}`, "", []string{"a({})"}},
		// A reply that merely mentions the tag is not a call.
		{"tag with no json", "see <tool_call>nonsense</tool_call> there",
			"see <tool_call>nonsense there", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			text, calls := parseToolCalls(tt.in)
			if text != tt.text {
				t.Errorf("text = %q, want %q", text, tt.text)
			}
			if len(calls) != len(tt.calls) {
				t.Fatalf("got %d calls, want %d: %+v", len(calls), len(tt.calls), calls)
			}
			for i, want := range tt.calls {
				got := calls[i].Function.Name + "(" + calls[i].Function.Arguments + ")"
				if got != want {
					t.Errorf("call %d = %s, want %s", i, got, want)
				}
				// yagi and other clients accumulate deltas by index, so
				// the indices must be 0..n-1 in order.
				if calls[i].Index == nil || *calls[i].Index != i {
					t.Errorf("call %d has index %v, want %d", i, calls[i].Index, i)
				}
				if calls[i].ID == "" || calls[i].Type != "function" {
					t.Errorf("call %d = %+v, want an id and type function", i, calls[i])
				}
			}
		})
	}
}

func TestPartialMarker(t *testing.T) {
	const m = "<tool_call>"
	for _, tt := range []struct {
		in   string
		want int
	}{
		{"hello", 0},
		{"hello<", 1},
		{"hello<tool", 5},
		{"<tool_cal", 9},
		{"a<b", 0},
		{"", 0},
		// A complete marker is the caller's business, not a partial one.
		{"x<tool_call>", 0},
	} {
		if got := partialMarker(tt.in, m); got != tt.want {
			t.Errorf("partialMarker(%q) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestRenderOffersTools(t *testing.T) {
	got := render(chatml(), []chatMessage{{Role: "user", Content: "weather?"}}, "sys", []toolDef{weatherTool()})
	for _, want := range []string{
		"<tools>", "</tools>", `"name":"get_weather"`, "<tool_call>",
		"<|im_start|>system\nsys\n\n# Tools",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("rendered prompt is missing %q:\n%s", want, got)
		}
	}
	// Without tools the prompt must be exactly what it was before.
	plain := render(chatml(), []chatMessage{{Role: "user", Content: "weather?"}}, "sys", nil)
	if strings.Contains(plain, "# Tools") {
		t.Errorf("a request with no tools grew a tools block:\n%s", plain)
	}
}

func TestRenderToolRoundTrip(t *testing.T) {
	idx := 0
	msgs := []chatMessage{
		{Role: "user", Content: "weather in Tokyo?"},
		{Role: "assistant", ToolCalls: []toolCall{{
			Index: &idx, ID: "call_0", Type: "function",
			Function: callFunc{Name: "get_weather", Arguments: `{"city":"Tokyo"}`},
		}}},
		{Role: "tool", ToolCallID: "call_0", Content: `{"c":22}`},
		{Role: "tool", ToolCallID: "call_1", Content: `{"c":19}`},
		{Role: "user", Content: "and tomorrow?"},
	}
	got := render(chatml(), msgs, "sys", []toolDef{weatherTool()})
	want := "<|im_start|>assistant\n\n<tool_call>\n" +
		`{"name": "get_weather", "arguments": {"city":"Tokyo"}}` + "\n</tool_call><|im_end|>\n"
	if !strings.Contains(got, want) {
		t.Errorf("call replay missing:\n%s", got)
	}
	// Consecutive results share one user turn, as the trained template does.
	want = "<|im_start|>user\n<tool_response>\n{\"c\":22}\n</tool_response>" +
		"<tool_response>\n{\"c\":19}\n</tool_response><|im_end|>\n"
	if !strings.Contains(got, want) {
		t.Errorf("tool results not grouped into one turn:\n%s", got)
	}
	if n := strings.Count(got, "<|im_start|>user"); n != 3 {
		t.Errorf("got %d user turns, want 3 (question, results, follow-up):\n%s", n, got)
	}
	if !strings.HasSuffix(got, "<|im_start|>assistant\n") {
		t.Errorf("prompt does not end with an open assistant turn:\n%s", got)
	}
}

// A family with no calling convention must say so rather than quietly
// dropping the tools the caller offered.
func TestFamiliesWithoutToolConvention(t *testing.T) {
	for _, style := range []string{"gemma3", "phi3", "mistral", "deepseek", "gpt-oss"} {
		if tm := templateFor(style, false); tm.toolCalls != "" {
			t.Errorf("templateFor(%q).toolCalls = %q, want empty", style, tm.toolCalls)
		}
	}
	for _, style := range []string{"qwen2", "qwen3", "llama", "qwen3moe"} {
		if tm := templateFor(style, false); tm.toolCalls != "hermes" {
			t.Errorf("templateFor(%q).toolCalls = %q, want hermes", style, tm.toolCalls)
		}
	}
}

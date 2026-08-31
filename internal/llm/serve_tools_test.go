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
	// Consecutive results share one user turn, each opening its own
	// line, as the trained template writes them.
	want = "<|im_start|>user\n<tool_response>\n{\"c\":22}\n</tool_response>\n" +
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
	// Qwen3.5 speaks a convention of its own, not the one its
	// predecessors were trained on.
	if tm := templateFor("qwen3_5", false); tm.toolCalls != "qwen3xml" {
		t.Errorf("templateFor(qwen3_5).toolCalls = %q, want qwen3xml", tm.toolCalls)
	}
}

// Excerpts of the real templates these checkpoints ship, which is the
// only thing that says whether they were prepared to be offered tools.
const (
	qwenTemplate = `{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- for tool in tools %}
        {{- "\n" }}{{- tool | tojson }}
    {%- endfor %}`
	llamaTemplate = `{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}`
	// SmolLM2 ships 368 characters that never mention the variable.
	smolTemplate = `{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}` +
		`{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM<|im_end|>\n' }}{% endif %}` +
		`{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}`
	// Prose about tools is not a branch on them.
	proseTemplate = `{% for m in messages %}{{ "You have no tools available." + m['content'] }}{% endfor %}`
)

func TestTemplateTakesTools(t *testing.T) {
	for _, tt := range []struct {
		name string
		tpl  string
		want bool
	}{
		{"qwen", qwenTemplate, true},
		{"llama 3.1", llamaTemplate, true},
		{"smollm2", smolTemplate, false},
		{"prose only", proseTemplate, false},
		{"empty", "", false},
		{"odd spacing", "{%-   if   tools   %}", true},
	} {
		if got := templateTakesTools(tt.tpl); got != tt.want {
			t.Errorf("templateTakesTools(%s) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// Qwen3.5 does not speak the Hermes convention: its signatures lead the
// system turn and a call is a <function> element, not JSON. Every string
// checked here comes from the template the checkpoint ships.
func qwen35() tmpl { return templateFor("qwen3_5", false) }

func TestRenderQwen35OffersTools(t *testing.T) {
	got := render(qwen35(), []chatMessage{{Role: "user", Content: "weather?"}}, "sys", []toolDef{weatherTool()})
	for _, want := range []string{
		"<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
		`{"type": "function", "function": {"name": "get_weather", "description": "Look up the weather"`,
		"If you choose to call a function ONLY reply in the following format with NO suffix:",
		"<function=example_function_name>",
		// The caller's own system text follows the block, not the
		// other way round.
		"</IMPORTANT>\n\nsys<|im_end|>",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("rendered prompt is missing %q:\n%s", want, got)
		}
	}
	// The Hermes preamble must not leak into a family that never saw it.
	if strings.Contains(got, "return a json object with function name") {
		t.Errorf("qwen3.5 prompt carries the hermes preamble:\n%s", got)
	}
}

func TestRenderQwen35ReplaysCalls(t *testing.T) {
	idx := 0
	msgs := []chatMessage{
		{Role: "user", Content: "weather in Tokyo?"},
		{Role: "assistant", Content: "Let me look.", ToolCalls: []toolCall{{
			Index: &idx, ID: "call_0", Type: "function",
			Function: callFunc{Name: "get_weather", Arguments: `{"city":"Tokyo","days":2}`},
		}}},
		{Role: "tool", ToolCallID: "call_0", Content: `{"c":22}`},
	}
	got := render(qwen35(), msgs, "sys", []toolDef{weatherTool()})
	// The turn belongs to the query still being answered, so it keeps
	// the think block its template writes there.
	want := "<|im_start|>assistant\n<think>\n\n</think>\n\nLet me look.\n\n" +
		"<tool_call>\n<function=get_weather>\n" +
		"<parameter=city>\nTokyo\n</parameter>\n<parameter=days>\n2\n</parameter>\n" +
		"</function>\n</tool_call><|im_end|>\n"
	if !strings.Contains(got, want) {
		t.Errorf("call replay missing:\n%s", got)
	}
	if !strings.HasSuffix(got, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Errorf("prompt does not end with an open thinking turn:\n%s", got)
	}
}

// An assistant turn older than the last user message loses the empty
// think block, which is what the template does with it.
func TestRenderQwen35DropsStaleThinkBlock(t *testing.T) {
	msgs := []chatMessage{
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c"},
	}
	got := render(qwen35(), msgs, "sys", nil)
	if !strings.Contains(got, "<|im_start|>assistant\nb<|im_end|>") {
		t.Errorf("replayed answer grew a think block:\n%s", got)
	}
	if n := strings.Count(got, "<think>"); n != 1 {
		t.Errorf("got %d think blocks, want only the open turn's:\n%s", n, got)
	}
}

func TestParseXMLToolCalls(t *testing.T) {
	tools := []toolDef{{Type: "function", Function: toolFunc{
		Name: "f",
		Parameters: json.RawMessage(`{"type":"object","properties":{` +
			`"s":{"type":"string"},"n":{"type":"integer"},"b":{"type":"boolean"},` +
			`"xs":{"type":"array"},"o":{"type":"object"}}}`),
	}}}
	for _, tt := range []struct {
		name  string
		in    string
		text  string
		calls []string
	}{
		{"plain text", "The capital is Paris.", "The capital is Paris.", nil},
		{"one string argument",
			"<tool_call>\n<function=f>\n<parameter=s>\nTokyo\n</parameter>\n</function>\n</tool_call>",
			"", []string{`f({"s": "Tokyo"})`}},
		// The schema is the only thing that says what a value means: a
		// declared string stays text however numeric it looks.
		{"types come from the schema",
			"<tool_call>\n<function=f>\n<parameter=n>\n3\n</parameter>\n" +
				"<parameter=b>\ntrue\n</parameter>\n<parameter=s>\n42\n</parameter>\n" +
				"<parameter=xs>\n[1, 2]\n</parameter>\n</function>\n</tool_call>",
			"", []string{`f({"n": 3, "b": true, "s": "42", "xs": [1, 2]})`}},
		{"text before the call",
			"I will look.\n<tool_call>\n<function=f>\n</function>\n</tool_call>",
			"I will look.", []string{"f({})"}},
		{"two calls",
			"<tool_call>\n<function=f>\n<parameter=s>\na\n</parameter>\n</function>\n</tool_call>\n" +
				"<tool_call>\n<function=f>\n<parameter=s>\nb\n</parameter>\n</function>\n</tool_call>",
			"", []string{`f({"s": "a"})`, `f({"s": "b"})`}},
		// A value may span lines, and only the newlines the format adds
		// come off.
		{"multi-line value",
			"<tool_call>\n<function=f>\n<parameter=s>\nfirst\n\nlast\n</parameter>\n</function>\n</tool_call>",
			"", []string{"f({\"s\": \"first\\n\\nlast\"})"}},
		// Running out of tokens mid-block must not drop what arrived.
		{"unterminated",
			"<tool_call>\n<function=f>\n<parameter=s>\nTokyo\n</parameter>",
			"", []string{`f({"s": "Tokyo"})`}},
		// An argument the schema never mentions is text unless it is
		// bracketed like JSON.
		{"undeclared argument",
			"<tool_call>\n<function=f>\n<parameter=zz>\n7\n</parameter>\n</function>\n</tool_call>",
			"", []string{`f({"zz": "7"})`}},
		// A reply that merely mentions the tag is not a call.
		{"tag with no function", "see <tool_call>nonsense</tool_call> there",
			"see <tool_call>nonsense there", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			text, calls := parseXMLToolCalls(tt.in, tools)
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

// What a call parses to must be what replaying it writes back, or an
// agent loop teaches the model a format it did not emit.
func TestQwen35CallRoundTrip(t *testing.T) {
	const block = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n" +
		"</function>\n</tool_call>"
	_, calls := parseXMLToolCalls(block, []toolDef{weatherTool()})
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	msgs := []chatMessage{{Role: "user", Content: "weather?"}, {Role: "assistant", ToolCalls: calls}}
	if got := render(qwen35(), msgs, "sys", []toolDef{weatherTool()}); !strings.Contains(got, block) {
		t.Errorf("replay does not reproduce the call:\n%s", got)
	}
}

// Go escapes <, > and & and packs JSON tight; the template the model was
// trained on does neither.
func TestToolSignaturesMatchPythonJSON(t *testing.T) {
	tool := toolDef{Type: "function", Function: toolFunc{
		Name: "f", Description: "a < b & c > d",
		Parameters: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
	}}
	got := xmlToolPreamble([]toolDef{tool})
	want := `{"type": "function", "function": {"name": "f", "description": "a < b & c > d", ` +
		`"parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}`
	if !strings.Contains(got, want) {
		t.Errorf("signature is not written the way the template writes it:\ngot:  %s\nwant: %s", got, want)
	}
}

func TestParseLooseToolCalls(t *testing.T) {
	tools := []toolDef{weatherTool()}
	for _, tt := range []struct {
		name, in  string
		wantText  string
		wantCalls int
		wantFn    string
		wantArgs  string
	}{
		{"fenced json call", "```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Osaka\"}}\n```", "", 1, "get_weather", `{"city": "Osaka"}`},
		{"fence without lang", "```\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Osaka\"}}\n```", "", 1, "get_weather", `{"city": "Osaka"}`},
		{"unterminated fence", "```json\n{\"name\": \"get_weather\", \"arguments\": {}}", "", 1, "get_weather", "{}"},
		{"bare object turn", `{"name": "get_weather", "arguments": {"city": "Osaka"}}`, "", 1, "get_weather", `{"city": "Osaka"}`},
		{"text around call", "調べます。\n```json\n{\"name\": \"get_weather\", \"arguments\": {}}\n```\n以上", "調べます。\n\n以上", 1, "get_weather", "{}"},
		{"unknown tool stays text", "```json\n{\"name\": \"rm_rf\", \"arguments\": {}}\n```", "```json\n{\"name\": \"rm_rf\", \"arguments\": {}}\n```", 0, "", ""},
		{"ordinary code block", "例です:\n```go\nfmt.Println(1)\n```", "例です:\n```go\nfmt.Println(1)\n```", 0, "", ""},
		{"plain text", "こんにちは", "こんにちは", 0, "", ""},
	} {
		text, calls := parseLooseToolCalls(tt.in, tools)
		if len(calls) != tt.wantCalls {
			t.Fatalf("%s: got %d calls, want %d", tt.name, len(calls), tt.wantCalls)
		}
		if strings.TrimSpace(text) != strings.TrimSpace(tt.wantText) {
			t.Errorf("%s: text %q, want %q", tt.name, text, tt.wantText)
		}
		if tt.wantCalls == 1 {
			if calls[0].Function.Name != tt.wantFn {
				t.Errorf("%s: name %q", tt.name, calls[0].Function.Name)
			}
			if calls[0].Function.Arguments != tt.wantArgs {
				t.Errorf("%s: args %q, want %q", tt.name, calls[0].Function.Arguments, tt.wantArgs)
			}
		}
	}
}

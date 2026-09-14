package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// K2-Horizon (IFM) differs from Qwen3-family in three ways:
//  1. Turn markers: <|ifm|im_start|> / <|ifm|im_end|> (pipe-delimited).
//  2. Reasoning:    <ifm|think> / </ifm|think> (IFM signature).
//  3. Tool calls:   qwen3xml nested-XML (same wire format as Qwen3.5).

// k2h returns the K2-Horizon template with thinking enabled.
func k2h() tmpl { return templateFor("k2-horizon", true) }

// --- 1. Turn markers ---

func TestK2HorizonTurnMarkers(t *testing.T) {
	tm := k2h()
	if tm.sysOpen != "<|ifm|im_start|>system\n" {
		t.Errorf("sysOpen = %q", tm.sysOpen)
	}
	if tm.userOpen != "<|ifm|im_start|>user\n" {
		t.Errorf("userOpen = %q", tm.userOpen)
	}
	if tm.asstOpen != "<|ifm|im_start|>assistant\n" {
		t.Errorf("asstOpen = %q", tm.asstOpen)
	}
	if len(tm.stops) == 0 || tm.stops[0] != "<|ifm|im_end|>" {
		t.Errorf("stops = %v", tm.stops)
	}
	if strings.Contains(tm.sysOpen, "<im_start>") {
		t.Errorf("ChatML marker leaked into sysOpen: %q", tm.sysOpen)
	}
}

// --- 2. Reasoning markers ---

func TestK2HorizonReasoningMarkers(t *testing.T) {
	tm := k2h()
	if tm.reasonOpen != "<ifm|think>" {
		t.Errorf("reasonOpen = %q, want <ifm|think>", tm.reasonOpen)
	}
	if tm.reasonClose != "</ifm|think>" {
		t.Errorf("reasonClose = %q, want </ifm|think>", tm.reasonClose)
	}
}

func TestK2HorizonThinkingOff(t *testing.T) {
	tm := templateFor("k2-horizon", false)
	if tm.reasonOpen != "" {
		t.Errorf("thinking off: reasonOpen = %q, want empty", tm.reasonOpen)
	}
}

func TestK2HorizonThoughtFilter(t *testing.T) {
	for _, tt := range []struct {
		name  string
		parts []string
		want  string
	}{
		{"empty block",
			[]string{"<ifm|think>", "</ifm|think>", "hello"}, "hello"},
		{"block with reasoning",
			[]string{"<ifm|think>", "let me think", "</ifm|think>", "answer"}, "answer"},
		{"markers split across writes",
			[]string{"<ifm|thi", "nk>reason</ifm|thi", "nk>done"}, "done"},
		{"unclosed block is suppressed",
			[]string{"<ifm|think>", "still thinking"}, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var out strings.Builder
			f := &thoughtFilter{w: &out, open: "<ifm|think>", close: "</ifm|think>"}
			for _, p := range tt.parts {
				if _, err := f.Write([]byte(p)); err != nil {
					t.Fatal(err)
				}
			}
			f.flush()
			if out.String() != tt.want {
				t.Errorf("got %q, want %q", out.String(), tt.want)
			}
		})
	}
}

func TestK2HorizonSplitReasoning(t *testing.T) {
	for _, tt := range []struct{ in, reason, rest string }{
		{"<ifm|think>a</ifm|think>b", "a", "b"},
		{"no block", "", "no block"},
		{"<ifm|think>unfinished", "unfinished", ""},
		{"lead <ifm|think>a</ifm|think> tail", "a", "lead  tail"},
	} {
		reason, rest := splitReasoning(tt.in, "<ifm|think>", "</ifm|think>")
		if reason != tt.reason || rest != tt.rest {
			t.Errorf("splitReasoning(%q) = (%q, %q), want (%q, %q)",
				tt.in, reason, rest, tt.reason, tt.rest)
		}
	}
}

// --- 3. Tool calls (qwen3xml format) ---

func TestK2HorizonToolConvention(t *testing.T) {
	tm := k2h()
	if tm.toolCalls != "qwen3xml" {
		t.Errorf("toolCalls = %q, want qwen3xml", tm.toolCalls)
	}
}

func TestK2HorizonRenderOffersTools(t *testing.T) {
	got := render(k2h(),
		[]chatMessage{{Role: "user", Content: "weather?"}},
		"sys", []toolDef{weatherTool()})
	for _, want := range []string{
		"<|ifm|im_start|>system\n",
		"<tools>",
		`"name": "get_weather"`,
		"<tool_call>",
		"<|ifm|im_end|>",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("rendered prompt is missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "<im_start>") {
		t.Errorf("ChatML marker leaked into K2-Horizon prompt:\n%s", got)
	}
}
func TestK2HorizonCallRoundTrip(t *testing.T) {
	block := "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
	_, calls := parseXMLToolCalls(block, []toolDef{weatherTool()})
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	msgs := []chatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: calls},
	}
	got := render(k2h(), msgs, "sys", []toolDef{weatherTool()})
	if !strings.Contains(got, block) {
		t.Errorf("replay does not reproduce the call:\n%s", got)
	}
	if strings.Contains(got, "<im_start>") {
		t.Errorf("ChatML marker leaked in round-trip:\n%s", got)
	}
}

// --- 4. Arg-leak guard ---

func TestK2HorizonArgsNoXMLLeak(t *testing.T) {
	block := "<tool_call>\n<function=calc>\n<parameter=expr>\n12 * 34\n</parameter>\n</function>\n</tool_call>"
	tools := []toolDef{{
		Type: "function", Function: toolFunc{
			Name:       "calc",
			Parameters: []byte(`{"type":"object","properties":{"expr":{"type":"string"}}}`),
		},
	}}
	_, calls := parseXMLToolCalls(block, tools)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	args := calls[0].Function.Arguments
	for _, bad := range []string{"</ifm|", "<ifm|", "arg_value", "arg_key"} {
		if strings.Contains(args, bad) {
			t.Errorf("XML leak in arguments: %q contains %q", args, bad)
		}
	}
	if !strings.Contains(args, "12 * 34") {
		t.Errorf("arguments = %q, want expression", args)
	}
}

// --- 5. Inspect ---

func TestK2HorizonInspect(t *testing.T) {
	got := Inspect(modelDir(t, "k2-horizon", qwenTemplate)).String()
	if got != "tools think" {
		t.Errorf("Inspect(k2-horizon) = %q, want %q", got, "tools think")
	}
}

func TestK2HorizonInspectNoTemplate(t *testing.T) {
	got := Inspect(modelDir(t, "k2-horizon", "")).String()
	if got != "tools think" {
		t.Errorf("Inspect(k2-horizon, no template) = %q, want %q", got, "tools think")
	}
}

// --- 6. Japanese text (IFM is Japanese-native) ---

func TestK2HorizonJapaneseClip(t *testing.T) {
	got := clipRunes("  \u6771\u4eac\u306e\u5929\u6c17\u306f\uff1f  ", 10)
	if got != "\u6771\u4eac\u306e\u5929\u6c17\u306f\uff1f" {
		t.Errorf("clipRunes(japanese) = %q", got)
	}
}

func TestK2HorizonWikiLangJa(t *testing.T) {
	if got := wikiLang("\u6771\u4eac\u30bf\u30ef\u30fc"); got != "ja" {
		t.Errorf("wikiLang() = %s, want ja", got)
	}
}

func TestK2HorizonLooseToolCallJa(t *testing.T) {
	in := "\u8abf\u3079\u307e\u3059\u3002\n```json\n{\"name\": \"wikipedia\", \"arguments\": {\"query\": \"\u6771\u4eac\"}}\n```\n\u4ee5\u4e0a"
	text, calls := parseLooseToolCalls(in, []toolDef{weatherTool()})
	if len(calls) != 0 {
		t.Errorf("got %d calls, want 0 (wikipedia is not offered)", len(calls))
	}
	if !strings.Contains(text, "\u6771\u4eac") {
		t.Errorf("Japanese text lost in loose parse:\n%s", text)
	}
	if !strings.Contains(text, "\u4ee5\u4e0a") {
		t.Errorf("trailing Japanese lost:\n%s", text)
	}
}

// --- 7. Convention list ---

func TestK2HorizonIsInToolConventionList(t *testing.T) {
	if tm := templateFor("k2-horizon", false); tm.toolCalls != "qwen3xml" {
		t.Errorf("templateFor(k2-horizon).toolCalls = %q, want qwen3xml", tm.toolCalls)
	}
}

// --- 8. Thinking-off asstPrefill ---

func TestK2HorizonThinkingOffAsstPrefill(t *testing.T) {
	tm := templateFor("k2-horizon", false)
	if tm.asstPrefill == "" {
		t.Errorf("thinking off: asstPrefill is empty, want empty think block prefill")
	}
	if !strings.Contains(tm.asstPrefill, "<ifm|think>") {
		t.Errorf("thinking off: asstPrefill = %q, want <ifm|think>", tm.asstPrefill)
	}
	if !strings.Contains(tm.asstPrefill, "</ifm|think>") {
		t.Errorf("thinking off: asstPrefill = %q, want </ifm|think>", tm.asstPrefill)
	}
}

// --- 9. Reasoning dropped from history ---

func TestK2HorizonRenderDropsReasoningFromHistory(t *testing.T) {
	got := render(k2h(), []chatMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "<ifm|think>pondering</ifm|think>hello"},
		{Role: "user", Content: "again"},
	}, "sys", nil)
	if strings.Contains(got, "pondering") || strings.Contains(got, "<ifm|think>") {
		t.Errorf("replayed history carried the thinking back in:\n%s", got)
	}
	if !strings.Contains(got, "hello") {
		t.Errorf("the answer was dropped along with the thinking:\n%s", got)
	}
}

// --- 10. Stale think block not injected in history replay ---

func TestK2HorizonDropsStaleThinkBlock(t *testing.T) {
	msgs := []chatMessage{
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c"},
	}
	got := render(k2h(), msgs, "sys", nil)
	if !strings.Contains(got, "<|ifm|im_start|>assistant\nb<|ifm|im_end|>") {
		t.Errorf("replayed answer grew a think block:\n%s", got)
	}
	if n := strings.Count(got, "<ifm|think>"); n != 0 {
		t.Errorf("got %d <ifm|think> blocks in history, want 0:\n%s", n, got)
	}
}

// --- 11. Streaming reasoning with IFM markers ---

func k2hThinkingTmpl() tmpl { return templateFor("k2-horizon", true) }

func TestK2HorizonReasoningSplitNonStreaming(t *testing.T) {
	// Markers arrive split across tokens, as a byte-level BPE gives them.
	s := scriptedServer(t, k2hThinkingTmpl(), []string{
		"<ifm|thi", "nk>", "\n17*3", " is 51.\n", "</ifm|thi", "nk>", "\n\nIt is ", "51.",
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
	for _, bad := range []string{"<ifm|think", "ifm|think", "</ifm|think"} {
		if strings.Contains(m.Content, bad) {
			t.Errorf("the marker leaked into content: %q", m.Content)
		}
	}
}

func TestK2HorizonReasoningSplitStreaming(t *testing.T) {
	s := scriptedServer(t, k2hThinkingTmpl(), []string{
		"<ifm|thi", "nk>", "weighing", " it", "</ifm|thi", "nk>", "done", ".",
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
	for _, bad := range []string{"<ifm|thi", "ifm|think", "</ifm|thi"} {
		if strings.Contains(content.String(), bad) || strings.Contains(reason.String(), bad) {
			t.Errorf("a marker fragment %q leaked into the stream", bad)
		}
	}
}

// --- 12. Reasoning absent when think=false ---

func TestK2HorizonReasoningAbsent(t *testing.T) {
	s := scriptedServer(t, templateFor("k2-horizon", false), []string{"just", " an ", "answer"})
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
}

package llm

import (
	"encoding/json"
	"testing"
)

func gemma4Tmpl() tmpl { return templateFor("gemma4", false) }

func weatherToolG4() toolDef {
	return toolDef{Type: "function", Function: toolFunc{
		Name:        "get_weather",
		Description: "Get the current weather in a city",
		Parameters: json.RawMessage(`{"type":"object","properties":` +
			`{"city":{"type":"string","description":"City name"},` +
			`"days":{"type":"integer","description":"How many days"}},` +
			`"required":["city"]}`),
	}}
}

// The signatures, byte for byte as llama.cpp renders the model's own
// template: properties sorted, strings in <|"|>, types upper-cased, and
// the whole thing inside the system turn.
func TestGemma4ToolPrompt(t *testing.T) {
	got := render(gemma4Tmpl(), []chatMessage{{Role: "user", Content: "weather?"}},
		"You are a helpful assistant.", []toolDef{weatherToolG4()})
	want := `<bos><|turn>system
You are a helpful assistant.<|tool>declaration:get_weather{description:<|"|>Get the current weather in a city<|"|>,parameters:{properties:{city:{description:<|"|>City name<|"|>,type:<|"|>STRING<|"|>},days:{description:<|"|>How many days<|"|>,type:<|"|>INTEGER<|"|>}},required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
weather?<turn|>
<|turn>model
`
	if got != want {
		t.Errorf("render() =\n%q\nwant\n%q", got, want)
	}
	// Without tools the system turn carries nothing extra.
	plain := render(gemma4Tmpl(), []chatMessage{{Role: "user", Content: "hi"}}, "sys", nil)
	if plain != "<bos><|turn>system\nsys<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n" {
		t.Errorf("render() without tools = %q", plain)
	}
}

// A call and the result answering it stay inside one model turn, and the
// answer continues in that same turn rather than opening another.
func TestGemma4ToolRoundTrip(t *testing.T) {
	msgs := []chatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []toolCall{{
			ID: "call_0", Type: "function",
			Function: callFunc{Name: "get_weather", Arguments: `{"city":"Tokyo","days":2,"metric":true}`},
		}}},
		{Role: "tool", ToolCallID: "call_0", Content: `{"temp_c":22}`},
	}
	got := render(gemma4Tmpl(), msgs, "", []toolDef{weatherToolG4()})
	want := `<|turn>model
<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>,days:2,metric:true}<tool_call|>` +
		`<|tool_response>response:get_weather{value:<|"|>{"temp_c":22}<|"|>}<tool_response|>`
	if len(got) < len(want) || got[len(got)-len(want):] != want {
		t.Errorf("render() ends %q, want %q", got, want)
	}
	// A question after the result closes that turn and opens a new one.
	msgs = append(msgs, chatMessage{Role: "user", Content: "and Osaka?"})
	got = render(gemma4Tmpl(), msgs, "", []toolDef{weatherToolG4()})
	tail := "<tool_response|><turn|>\n<|turn>user\nand Osaka?<turn|>\n<|turn>model\n"
	if len(got) < len(tail) || got[len(got)-len(tail):] != tail {
		t.Errorf("render() ends %q, want it to end %q", got, tail)
	}
}

func TestParseGemma4ToolCalls(t *testing.T) {
	for _, tt := range []struct {
		name    string
		in      string
		content string
		calls   [][2]string // name, arguments
	}{
		{"a call and nothing else",
			`<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>`,
			"", [][2]string{{"get_weather", `{"city":"Tokyo"}`}}},
		{"every kind of value",
			`<|tool_call>call:f{n:2,x:1.5,b:true,z:null,l:[1,<|"|>a<|"|>],o:{k:<|"|>v<|"|>}}<tool_call|>`,
			// The order the model wrote them in is the order that comes
			// back; only what tensai writes is sorted.
			"", [][2]string{{"f", `{"n":2,"x":1.5,"b":true,"z":null,"l":[1,"a"],"o":{"k":"v"}}`}}},
		{"no arguments",
			`<|tool_call>call:now{}<tool_call|>`,
			"", [][2]string{{"now", `{}`}}},
		{"two calls, and what came before them",
			`Let me check. <|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:<|"|>2<|"|>}<tool_call|>`,
			"Let me check.", [][2]string{{"a", `{"x":1}`}, {"b", `{"y":"2"}`}}},
		{"the model asking for the result it expects",
			`<|tool_call>call:a{x:1}<tool_call|><|tool_response>`,
			"", [][2]string{{"a", `{"x":1}`}}},
		{"cut short by the token limit",
			`<|tool_call>call:get_weather{city:<|"|>Tok`,
			"", [][2]string{{"get_weather", `{"city":"Tok"}`}}},
		{"a reply that merely mentions the marker",
			"A call starts with <|tool_call> and I cannot make one.",
			"A call starts with <|tool_call> and I cannot make one.", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			content, calls := parseGemma4ToolCalls(tt.in)
			if content != tt.content {
				t.Errorf("content = %q, want %q", content, tt.content)
			}
			if len(calls) != len(tt.calls) {
				t.Fatalf("got %d calls, want %d: %+v", len(calls), len(tt.calls), calls)
			}
			for i, want := range tt.calls {
				if calls[i].Function.Name != want[0] || calls[i].Function.Arguments != want[1] {
					t.Errorf("call %d = %s%s, want %s%s", i,
						calls[i].Function.Name, calls[i].Function.Arguments, want[0], want[1])
				}
			}
		})
	}
}

// What the parser reads back has to be what the renderer would write, or
// an agent replaying its own history teaches the model a dialect it does
// not speak.
func TestGemma4CallRoundTrips(t *testing.T) {
	for _, args := range []string{
		`{"city":"Tokyo"}`,
		`{"a":1,"b":-2.5,"c":true,"d":null}`,
		`{"list":["x","y"],"nested":{"k":1}}`,
		`{}`,
	} {
		_, calls := parseGemma4ToolCalls(gemma4Call("f", args))
		if len(calls) != 1 {
			t.Fatalf("%s: got %d calls", args, len(calls))
		}
		if got := calls[0].Function.Arguments; got != args {
			t.Errorf("%s came back as %s", args, got)
		}
	}
}

func TestGemma4IsOfferedTools(t *testing.T) {
	if tm := templateFor("gemma4", false); tm.toolCalls != "gemma4" {
		t.Errorf("templateFor(gemma4).toolCalls = %q", tm.toolCalls)
	}
	if got := callMarkers("gemma4"); got[0] != "<|tool_call>" {
		t.Errorf("callMarkers(gemma4) = %v", got)
	}
	if got := callMarkers("hermes"); got[0] != "<tool_call>" {
		t.Errorf("callMarkers(hermes) = %v", got)
	}
}

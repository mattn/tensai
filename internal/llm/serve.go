package llm

// -serve turns the example into a minimal OpenAI-compatible endpoint:
// POST /v1/chat/completions accepts the familiar messages array (with
// optional streaming), renders it through the ChatML template, prefills
// the prompt through the batched path, and decodes with the same
// sampling as the CLI. One request holds the model at a time — the KV
// cache is rebuilt per request — so any OpenAI client pointed at the
// address works against a pure-Go model. Offered tools reach families
// trained on the ChatML calling convention, and the calls they emit come
// back as tool_calls, which is enough for an agent to drive a loop.

import (
	"bytes"
	"crypto/subtle"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// An assistant turn replayed from history carries the calls it made,
	// and each result comes back as its own "tool" message naming the
	// call it answers.
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// toolFunc is one function signature: the JSON Schema in Parameters is
// passed to the model verbatim, which is what the families trained on
// this convention expect to read.
type toolFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type toolDef struct {
	Type     string   `json:"type"`
	Function toolFunc `json:"function"`
}

type callFunc struct {
	Name string `json:"name"`
	// Arguments is a JSON object encoded as a string, as the API defines
	// it -- not an object -- so a caller can hand it to a decoder of its
	// own without a second round trip.
	Arguments string `json:"arguments"`
}

type toolCall struct {
	Index    *int     `json:"index,omitempty"`
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function callFunc `json:"function"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature *float64      `json:"temperature"`
	TopP        *float64      `json:"top_p"`
	MaxTokens   int           `json:"max_tokens"`
	Seed        *int64        `json:"seed"`
	Tools       []toolDef     `json:"tools,omitempty"`
	// ToolChoice is "none", "auto", "required", or an object naming one
	// function. Only "none" changes what the model sees here: without a
	// constrained sampler nothing can force a call, so the rest read as
	// "auto".
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
}

// reset drops the KV cache so the next request starts a fresh context.
func (m *qwen) reset() {
	for i := range m.blocks {
		m.blocks[i].kc = nil
		m.blocks[i].vc = nil
		// A delta layer's state is not indexed by position, so there is
		// nothing to truncate: a fresh context starts from a zeroed one.
		m.blocks[i].dstate = nil
	}
}

// toolPreamble is the block the ChatML families were trained to read:
// the signatures they may call, and the shape a call takes. It is
// appended to the system turn, which is where their own template puts it.
func toolPreamble(tools []toolDef) string {
	var b strings.Builder
	b.WriteString("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, t := range tools {
		if t.Type == "" {
			t.Type = "function"
		}
		raw, err := json.Marshal(t)
		if err != nil {
			continue
		}
		b.WriteString("\n")
		b.Write(raw)
	}
	b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and " +
		"arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
		`{"name": <function-name>, "arguments": <args-json-object>}` + "\n</tool_call>")
	return b.String()
}

// xmlToolPreamble is the block Qwen3.5 was trained to read, word for
// word from its own template: the same <tools> listing, but a call
// written as a <function> element with one <parameter> per argument.
// It leads the system turn rather than trailing it.
func xmlToolPreamble(tools []toolDef) string {
	var b strings.Builder
	b.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
	for _, t := range tools {
		if t.Type == "" {
			t.Type = "function"
		}
		raw, err := marshalTool(t)
		if err != nil {
			continue
		}
		b.WriteString("\n" + raw)
	}
	b.WriteString("\n</tools>\n\n" +
		"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n<function=example_function_name>\n" +
		"<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
		"<parameter=example_parameter_2>\nThis is the value for the second parameter\n" +
		"that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n" +
		"<IMPORTANT>\nReminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> " +
		"block must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE " +
		"the function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your " +
		"current knowledge and do not tell the user about function calls\n</IMPORTANT>")
	return b.String()
}

// marshalTool writes one signature the way Qwen3.5's own template does,
// which is Python's json.dumps: ", " between members, ": " after a key,
// and <, > and & left as they are rather than escaped. Go's encoder
// defaults the other way on both counts, and the difference is not
// cosmetic -- it is the tokens the model reads.
func marshalTool(t toolDef) (string, error) {
	raw, err := encodeJSON(t)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	writeSpaced(&b, raw)
	return b.String(), nil
}

// encodeJSON marshals without Go's HTML escaping.
func encodeJSON(v any) (json.RawMessage, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return json.RawMessage(bytes.TrimRight(buf.Bytes(), "\n")), nil
}

// writeSpaced re-emits a JSON value with Python's separators, keeping
// the member order it was given. Anything it cannot walk goes out as it
// came in.
func writeSpaced(b *strings.Builder, raw json.RawMessage) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		b.Write(bytes.TrimSpace(raw))
		return
	}
	delim, _ := tok.(json.Delim)
	switch delim {
	case '{':
		b.WriteString("{")
		for i := 0; dec.More(); i++ {
			key, err := dec.Token()
			if err != nil {
				break
			}
			var value json.RawMessage
			if err := dec.Decode(&value); err != nil {
				break
			}
			if i > 0 {
				b.WriteString(", ")
			}
			name, _ := key.(string)
			quoted, err := encodeJSON(name)
			if err != nil {
				break
			}
			b.Write(quoted)
			b.WriteString(": ")
			writeSpaced(b, value)
		}
		b.WriteString("}")
	case '[':
		b.WriteString("[")
		for i := 0; dec.More(); i++ {
			var value json.RawMessage
			if err := dec.Decode(&value); err != nil {
				break
			}
			if i > 0 {
				b.WriteString(", ")
			}
			writeSpaced(b, value)
		}
		b.WriteString("]")
	default:
		b.Write(bytes.TrimSpace(raw))
	}
}

// xmlCallArgs renders a call's JSON arguments as the <parameter> blocks
// Qwen3.5 reads, in the order they arrived. A string goes in bare --
// the quotes are JSON's, not the format's -- and anything structured
// stays JSON, which is what its template writes.
func xmlCallArgs(args string) string {
	dec := json.NewDecoder(strings.NewReader(args))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return ""
	}
	var b strings.Builder
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			break
		}
		name, _ := key.(string)
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			break
		}
		value := string(raw)
		var s string
		if json.Unmarshal(raw, &s) == nil {
			value = s
		}
		b.WriteString("<parameter=" + name + ">\n" + value + "\n</parameter>\n")
	}
	return b.String()
}

// render walks the messages through the model's template, ending with an
// open assistant turn. Models without a system role (Gemma) get the
// system text folded into the first user message. When tools are offered
// and the family speaks a calling convention, the signatures join the
// system turn, past calls replay as <tool_call> blocks, and each result
// goes back as a <tool_response> in a user turn -- consecutive results
// share one turn, as the trained template does.
func render(tm tmpl, msgs []chatMessage, defaultSystem string, tools []toolDef) string {
	var b strings.Builder
	b.WriteString(tm.bos)
	system := defaultSystem
	rest := msgs
	if len(msgs) > 0 && msgs[0].Role == "system" {
		system = msgs[0].Content
		rest = msgs[1:]
	}
	hermes := tm.toolCalls == "hermes" && len(tools) > 0
	qwenXML := tm.toolCalls == "qwen3xml" && len(tools) > 0
	// Gemma 4 writes its signatures into the system turn as blocks of
	// its own, and keeps a call and the result answering it inside one
	// model turn rather than passing the result back as a user turn.
	gemma4 := tm.toolCalls == "gemma4" && len(tools) > 0
	// Where the signatures go is part of the convention: Hermes appends
	// them to whatever system text the caller sent, Qwen3.5 leads with
	// them and appends the caller's text after.
	if hermes {
		system += toolPreamble(tools)
	}
	if qwenXML {
		if system = strings.TrimSpace(system); system != "" {
			system = xmlToolPreamble(tools) + "\n\n" + system
		} else {
			system = xmlToolPreamble(tools)
		}
	}
	if gemma4 {
		system += gemma4ToolPreamble(tools)
	}
	offered := hermes || qwenXML
	// An empty system prompt means no system turn, not an empty one.
	if !tm.foldSystem && system != "" {
		b.WriteString(tm.sysOpen + system + tm.sysClose)
	}
	// A thinking family keeps the think block on the assistant turns
	// that belong to the query still being answered -- the ones an agent
	// loop replays around its tool results -- and drops it from
	// everything older, which is what its template does.
	lastQuery := -1
	for i, m := range rest {
		if m.Role == "user" {
			lastQuery = i
		}
	}
	first := true
	inTools := false
	// openTurn is a gemma4 model turn left open for the results of the
	// calls it just made; generation carries on inside it.
	openTurn := false
	for i, m := range rest {
		// A run of tool results is one user turn, so close the open one
		// only when the next message leaves the run.
		if inTools && m.Role != "tool" {
			b.WriteString(tm.userClose)
			inTools = false
		}
		switch {
		case m.Role == "tool" && gemma4:
			// The result belongs to the model turn that called for it,
			// which is still open.
			b.WriteString(gemma4Response(gemma4ToolName(rest[:i], m), m.Content))
			if i+1 < len(rest) && rest[i+1].Role != "tool" {
				b.WriteString(tm.asstClose)
				openTurn = false
			}
			first = false
		case m.Role == "tool" && offered:
			if !inTools {
				b.WriteString(tm.userOpen)
				inTools = true
			} else {
				// Each result opens its own line inside the shared turn,
				// as the trained template writes them.
				b.WriteString("\n")
			}
			b.WriteString("<tool_response>\n" + m.Content + "\n</tool_response>")
			if i == len(rest)-1 {
				b.WriteString(tm.userClose)
				inTools = false
			}
			first = false
		case m.Role == "assistant":
			// The model's own template drops reasoning from history, so a
			// client that echoes a whole turn back does not get to teach
			// it that thinking belongs in the answer.
			text := m.Content
			if tm.reasonOpen != "" {
				_, text = splitReasoning(text, tm.reasonOpen, tm.reasonClose)
			}
			b.WriteString(tm.asstOpen)
			if i > lastQuery {
				b.WriteString(tm.asstPrefill)
			}
			b.WriteString(text)
			if hermes {
				for _, c := range m.ToolCalls {
					args := strings.TrimSpace(c.Function.Arguments)
					if args == "" {
						args = "{}"
					}
					b.WriteString("\n<tool_call>\n" +
						`{"name": ` + strconv.Quote(c.Function.Name) + `, "arguments": ` + args + "}" +
						"\n</tool_call>")
				}
			}
			if gemma4 {
				for _, c := range m.ToolCalls {
					b.WriteString(gemma4Call(c.Function.Name, c.Function.Arguments))
				}
				// A turn whose calls are answered next stays open.
				if len(m.ToolCalls) > 0 && i+1 < len(rest) && rest[i+1].Role == "tool" {
					openTurn = true
					continue
				}
			}
			if qwenXML {
				for i, c := range m.ToolCalls {
					// A call follows text with a blank line between them
					// and another call with a single newline, but opens
					// a turn of its own with nothing in front.
					if i == 0 && strings.TrimSpace(text) != "" {
						b.WriteString("\n\n")
					} else if i > 0 {
						b.WriteString("\n")
					}
					b.WriteString("<tool_call>\n<function=" + c.Function.Name + ">\n" +
						xmlCallArgs(c.Function.Arguments) + "</function>\n</tool_call>")
				}
			}
			b.WriteString(tm.asstClose)
		default:
			text := m.Content
			if tm.foldSystem && first && system != "" {
				text = system + "\n\n" + text
			}
			first = false
			b.WriteString(tm.userOpen + text + tm.userClose)
		}
	}
	if inTools {
		b.WriteString(tm.userClose)
	}
	// An open gemma4 turn is where the answer goes: opening another one
	// would tell the model its own call and result belong to somebody
	// else's turn.
	if !openTurn {
		b.WriteString(tm.asstOpen)
	}
	b.WriteString(tm.asstPrefill)
	return b.String()
}

// callMarkers is what the streamer holds text back on: the opening of a
// call in the family's own convention, plus the fence a weak model
// writes a call in when it forgets the convention (parseLooseToolCalls
// rescues those at the end of the turn).
func callMarkers(conv string) []string {
	if conv == "gemma4" {
		return []string{"<|tool_call>", "```json"}
	}
	return []string{"<tool_call>", "```json"}
}

// splitReasoning takes the block a thinking model writes before its
// answer off the front of a finished turn. A turn cut short mid-block
// is all reasoning and no answer, which is what it is.
func splitReasoning(s, open, close string) (reason, rest string) {
	i := strings.Index(s, open)
	if i < 0 {
		return "", s
	}
	head, body := s[:i], s[i+len(open):]
	if j := strings.Index(body, close); j >= 0 {
		return strings.TrimSpace(body[:j]), strings.TrimSpace(head + body[j+len(close):])
	}
	return strings.TrimSpace(body), strings.TrimSpace(head)
}

// partialMarker returns how many trailing bytes of s could still grow
// into marker, so a stream can hold exactly that much back instead of
// printing half of an opening tag.
func partialMarker(s, marker string) int {
	n := len(marker) - 1
	if n > len(s) {
		n = len(s)
	}
	for ; n > 0; n-- {
		if strings.HasPrefix(marker, s[len(s)-n:]) {
			return n
		}
	}
	return 0
}

// parseToolCalls splits a finished assistant turn into the text it meant
// to say and the calls it emitted. A model that runs out of tokens mid
// block leaves the closing tag off, so an unterminated one takes the
// rest of the turn rather than being dropped.
func parseToolCalls(s string) (string, []toolCall) {
	const openTag, closeTag = "<tool_call>", "</tool_call>"
	if !strings.Contains(s, openTag) {
		return s, nil
	}
	var text strings.Builder
	var calls []toolCall
	for {
		i := strings.Index(s, openTag)
		if i < 0 {
			text.WriteString(s)
			break
		}
		text.WriteString(s[:i])
		s = s[i+len(openTag):]
		body := s
		if j := strings.Index(s, closeTag); j >= 0 {
			body, s = s[:j], s[j+len(closeTag):]
		} else {
			s = ""
		}
		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(body)), &call); err != nil || call.Name == "" {
			// Not a call after all: keep the text so a reply that merely
			// talks about the tag is not silently swallowed.
			text.WriteString(openTag + body)
			continue
		}
		args := strings.TrimSpace(string(call.Arguments))
		if args == "" || args == "null" {
			args = "{}"
		}
		idx := len(calls)
		calls = append(calls, toolCall{
			Index:    &idx,
			ID:       fmt.Sprintf("call_%d", idx),
			Type:     "function",
			Function: callFunc{Name: call.Name, Arguments: args},
		})
	}
	return strings.TrimSpace(text.String()), calls
}

// parseLooseToolCalls rescues a call the model wrote into its text
// instead of its trained format: a ```json fenced block — or the whole
// turn — holding {"name": ..., "arguments": {...}} whose name matches a
// declared tool. Four-bit sub-1B models keep the intent to call but
// lose the tag discipline, and without this they answer as prose and
// the agent loop never runs the tool. Requiring a known name keeps a
// reply that merely shows JSON from being swallowed.
func parseLooseToolCalls(s string, tools []toolDef) (string, []toolCall) {
	known := make(map[string]bool, len(tools))
	for _, t := range tools {
		known[t.Function.Name] = true
	}
	try := func(body string) (toolCall, bool) {
		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(body)), &call); err != nil || !known[call.Name] {
			return toolCall{}, false
		}
		args := strings.TrimSpace(string(call.Arguments))
		if args == "" || args == "null" {
			args = "{}"
		}
		return toolCall{Type: "function", Function: callFunc{Name: call.Name, Arguments: args}}, true
	}
	add := func(calls []toolCall, c toolCall) []toolCall {
		idx := len(calls)
		c.Index = &idx
		c.ID = fmt.Sprintf("call_%d", idx)
		return append(calls, c)
	}
	var calls []toolCall
	var text strings.Builder
	rest := s
	for {
		i := strings.Index(rest, "```")
		if i < 0 {
			text.WriteString(rest)
			break
		}
		after := rest[i+3:]
		body, tail := after, ""
		closed := false
		if j := strings.Index(after, "```"); j >= 0 {
			body, tail, closed = after[:j], after[j+3:], true
		}
		// Drop a language word on the fence line ("json", "tool_code", ...).
		if k := strings.IndexByte(body, '\n'); k >= 0 && len(strings.Fields(body[:k])) <= 1 {
			body = body[k+1:]
		}
		if c, ok := try(body); ok {
			text.WriteString(rest[:i])
			calls = add(calls, c)
		} else if closed {
			text.WriteString(rest[:i+3] + after[:len(after)-len(tail)])
		} else {
			text.WriteString(rest)
			break
		}
		rest = tail
		if !closed {
			break
		}
	}
	if len(calls) == 0 {
		if c, ok := try(s); ok {
			return "", add(nil, c)
		}
		return s, nil
	}
	return strings.TrimSpace(text.String()), calls
}

// parseXMLToolCalls does for Qwen3.5 what parseToolCalls does for the
// Hermes families: it splits a finished turn into what the model meant
// to say and the calls it emitted. The block opens with the same
// <tool_call> tag, so only its body differs.
func parseXMLToolCalls(s string, tools []toolDef) (string, []toolCall) {
	const openTag, closeTag = "<tool_call>", "</tool_call>"
	if !strings.Contains(s, openTag) {
		return s, nil
	}
	var text strings.Builder
	var calls []toolCall
	for {
		i := strings.Index(s, openTag)
		if i < 0 {
			text.WriteString(s)
			break
		}
		text.WriteString(s[:i])
		s = s[i+len(openTag):]
		body := s
		if j := strings.Index(s, closeTag); j >= 0 {
			body, s = s[:j], s[j+len(closeTag):]
		} else {
			s = ""
		}
		name, args, ok := parseXMLCall(body, tools)
		if !ok {
			// Not a call after all: keep the text rather than swallow a
			// reply that merely talks about the tag.
			text.WriteString(openTag + body)
			continue
		}
		idx := len(calls)
		calls = append(calls, toolCall{
			Index:    &idx,
			ID:       fmt.Sprintf("call_%d", idx),
			Type:     "function",
			Function: callFunc{Name: name, Arguments: args},
		})
	}
	return strings.TrimSpace(text.String()), calls
}

// parseXMLCall reads one <function=name> element and its <parameter>
// children back into the JSON object the API hands a caller. A block
// cut short mid-way still yields the parameters it got through.
func parseXMLCall(body string, tools []toolDef) (name, args string, ok bool) {
	const fnOpen, fnClose = "<function=", "</function>"
	i := strings.Index(body, fnOpen)
	if i < 0 {
		return "", "", false
	}
	rest := body[i+len(fnOpen):]
	j := strings.Index(rest, ">")
	if j < 0 {
		return "", "", false
	}
	name = strings.TrimSpace(rest[:j])
	if name == "" {
		return "", "", false
	}
	rest = rest[j+1:]
	if k := strings.Index(rest, fnClose); k >= 0 {
		rest = rest[:k]
	}
	types := paramTypes(tools, name)
	const pOpen, pClose = "<parameter=", "</parameter>"
	var b strings.Builder
	b.WriteString("{")
	for {
		i := strings.Index(rest, pOpen)
		if i < 0 {
			break
		}
		rest = rest[i+len(pOpen):]
		j := strings.Index(rest, ">")
		if j < 0 {
			break
		}
		key := strings.TrimSpace(rest[:j])
		rest = rest[j+1:]
		value := rest
		if k := strings.Index(rest, pClose); k >= 0 {
			value, rest = rest[:k], rest[k+len(pClose):]
		} else {
			rest = ""
		}
		// The template brackets a value with one newline on each side;
		// everything between them, blank lines included, is the value.
		value = strings.TrimSuffix(strings.TrimPrefix(value, "\n"), "\n")
		if b.Len() > 1 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Quote(key) + ": " + xmlArgJSON(value, types[key]))
	}
	b.WriteString("}")
	return name, b.String(), true
}

// xmlArgJSON turns one parameter body back into a JSON value. The wire
// format carries no types, so the tool's own schema decides: a declared
// string is taken verbatim however numeric it looks, and anything else
// is JSON when it parses as JSON. An argument the schema never mentions
// is a string unless it is bracketed like an object or an array.
func xmlArgJSON(value, typ string) string {
	trimmed := strings.TrimSpace(value)
	structured := strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
	if typ != "string" && (typ != "" || structured) {
		if trimmed != "" && json.Valid([]byte(trimmed)) {
			return trimmed
		}
	}
	return strconv.Quote(value)
}

// paramTypes reads the declared type of each of a tool's parameters out
// of its JSON Schema. A type given as a union is left unknown, which
// falls back to reading the value as written.
func paramTypes(tools []toolDef, name string) map[string]string {
	for _, t := range tools {
		if t.Function.Name != name {
			continue
		}
		var schema struct {
			Properties map[string]struct {
				Type json.RawMessage `json:"type"`
			} `json:"properties"`
		}
		if json.Unmarshal(t.Function.Parameters, &schema) != nil {
			return nil
		}
		types := make(map[string]string, len(schema.Properties))
		for k, p := range schema.Properties {
			var s string
			if json.Unmarshal(p.Type, &s) == nil {
				types[k] = s
			}
		}
		return types
	}
	return nil
}

type server struct {
	mu      sync.Mutex // one request drives the model at a time
	apiKey  string     // "" leaves /v1 open
	model   *qwen
	draft   *qwen
	specK   int
	tok     tokenizerIface
	system  string
	nCtx    int
	temp    float64
	topP    float64
	imEnd   int
	eot     int
	tm      tmpl
	cache   promptCache
	prefill func([]int, int) []float32
	step    func(int, int) []float32
	reset   func()
	vlog    io.Writer
}

// tokenizerIface is the slice of the tokenizer the server needs.
type tokenizerIface interface {
	Encode(string) []int
	Decode([]int) string
}

// auth wraps a /v1 handler with the bearer check. The demo page stays
// open — static HTML holds no secrets and a browser cannot attach a
// bearer header to plain navigation anyway — while every API route
// answers 401 until the right key arrives.
func (s *server) auth(h http.HandlerFunc) http.HandlerFunc {
	if s.apiKey == "" {
		return h
	}
	want := []byte("Bearer " + s.apiKey)
	return func(w http.ResponseWriter, r *http.Request) {
		got := []byte(r.Header.Get("Authorization"))
		if subtle.ConstantTimeCompare(got, want) == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{"message": "invalid api key", "type": "invalid_request_error"},
			})
			return
		}
		h(w, r)
	}
}

//go:embed webui.html
var webUI []byte

func (s *server) listen(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.auth(s.chatCompletions))
	mux.HandleFunc("/v1/models", s.auth(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{
			"object": "list",
			"data": []map[string]any{{
				"id": "tensai", "object": "model", "owned_by": "tensai",
			}},
		})
	}))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(webUI)
	})
	fmt.Printf("listening on %s (POST /v1/chat/completions)\n", addr)
	return http.ListenAndServe(addr, mux)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

func (s *server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpError(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	began := time.Now()
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		httpError(w, http.StatusBadRequest, "messages is required")
		return
	}
	temp := s.temp
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	topP := s.topP
	if req.TopP != nil {
		topP = *req.TopP
	}
	limit := req.MaxTokens
	if limit <= 0 {
		limit = 512
	}
	seed := time.Now().UnixNano()
	if req.Seed != nil {
		seed = *req.Seed
	}
	rng := rand.New(rand.NewSource(seed))

	tools := req.Tools
	// "none" is the one choice that changes the prompt: with no
	// constrained sampler the others cannot be enforced, so offering the
	// signatures and letting the model decide is all "auto" and
	// "required" can mean here.
	if choice := strings.Trim(string(req.ToolChoice), `"`); choice == "none" {
		tools = nil
	}
	if len(tools) > 0 && s.tm.toolCalls == "" {
		httpError(w, http.StatusBadRequest, "this model's chat template has no tool-calling convention, so tools cannot be offered to it")
		return
	}
	prompt := render(s.tm, req.Messages, s.system, tools)
	if s.vlog != nil {
		fmt.Fprintf(s.vlog, "request: %d messages, %d tools, stream=%v, limit %d, temp %.2f\n",
			len(req.Messages), len(tools), req.Stream, limit, temp)
		fmt.Fprintf(s.vlog, "rendered prompt: %s\n", clip(prompt, 600))
	}
	ids := s.tok.Encode(prompt)
	if len(ids) >= s.nCtx-1 {
		httpError(w, http.StatusBadRequest, fmt.Sprintf("prompt of %d tokens exceeds the %d-token context", len(ids), s.nCtx))
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// A draft model verifies against a rebuilt context, so it starts over
	// whatever the target does.
	if s.draft != nil {
		s.draft.reset()
	}
	start, restore := s.cache.plan(ids)
	switch {
	case restore:
		s.model.truncate(len(s.cache.ckpt))
		restoreDelta(s.model, s.cache.ckptDelta)
	case start > 0:
		s.model.truncate(start)
	default:
		s.reset()
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	// With tools in play the turn may open a <tool_call> block, which is
	// not content for the client to print. Text streams as usual until
	// the marker appears; any tail that could still grow into it is held
	// back, and once a call has started nothing more is streamed.
	sawCall := false
	// send names the field a piece belongs in: what a thinking model
	// writes before it answers is reasoning, not content, and clients
	// that show the two differently need them apart.
	var send func(field, piece string)
	// streamed counts the content bytes already sent, so text held back
	// on a false call marker can go out after the final parse clears it.
	streamed := 0
	flush := func(piece string) {
		if send != nil && piece != "" {
			streamed += len(piece)
			send("content", piece)
		}
	}
	flushReason := func(piece string) {
		if send != nil && piece != "" {
			send("reasoning_content", piece)
		}
	}
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, _ := w.(http.Flusher)
		send = func(field, piece string) {
			chunk := map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": created,
				"model": "tensai",
				"choices": []map[string]any{{
					"index": 0, "delta": map[string]any{field: piece},
				}},
			}
			raw, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
	// held is the text accepted but not yet streamed, kept only long
	// enough to rule out the start of a call marker.
	markers := callMarkers(s.tm.toolCalls)
	if s.tm.toolCalls == "gemma4" {
		// A gemma4 call sometimes arrives without its marker; the name
		// and its brace are the next best thing to wait on, and no
		// ordinary sentence writes one.
		for _, t := range tools {
			markers = append(markers, t.Function.Name+"{")
		}
	}
	var held strings.Builder
	answer := func(piece string, final bool) {
		if len(tools) == 0 {
			flush(piece)
			return
		}
		if sawCall {
			return
		}
		held.WriteString(piece)
		text := held.String()
		// A ```json fence can open a call a weak model writes as plain
		// JSON (see parseLooseToolCalls); held text that turns out to be
		// ordinary content goes out after the final parse, via streamed.
		// A bare fence streams as usual — a fenced code answer must not
		// fall silent — and the final parse still rescues a call in one.
		for _, marker := range markers {
			if i := strings.Index(text, marker); i >= 0 {
				sawCall = true
				held.Reset()
				if i > 0 {
					flush(text[:i])
				}
				return
			}
		}
		keep := 0
		if !final {
			for _, marker := range markers {
				keep = max(keep, partialMarker(text, marker))
			}
		}
		held.Reset()
		held.WriteString(text[len(text)-keep:])
		if len(text) > keep {
			flush(text[:len(text)-keep])
		}
	}

	// A thinking model opens its turn with a reasoning block; what is
	// inside goes out as reasoning and what follows is the answer. The
	// two markers are looked for one at a time, holding back only a tail
	// that could still grow into the one being awaited.
	var reasonHeld strings.Builder
	reasonState := 0 // 0 before the block, 1 inside it, 2 past it
	emit := func(piece string, final bool) {
		if s.tm.reasonOpen == "" || reasonState == 2 {
			answer(piece, final)
			return
		}
		reasonHeld.WriteString(piece)
		for {
			text := reasonHeld.String()
			marker := s.tm.reasonOpen
			if reasonState == 1 {
				marker = s.tm.reasonClose
			}
			if i := strings.Index(text, marker); i >= 0 {
				reasonHeld.Reset()
				reasonHeld.WriteString(text[i+len(marker):])
				if reasonState == 0 {
					answer(text[:i], false)
					reasonState = 1
					continue
				}
				flushReason(text[:i])
				reasonState = 2
				answer(reasonHeld.String(), final)
				reasonHeld.Reset()
				return
			}
			keep := 0
			if !final {
				keep = partialMarker(text, marker)
			}
			out := text[:len(text)-keep]
			reasonHeld.Reset()
			reasonHeld.WriteString(text[len(text)-keep:])
			if reasonState == 0 {
				answer(out, final)
			} else {
				flushReason(out)
			}
			return
		}
	}

	// Byte-fallback BPE tokens can split a multi-byte character, so a
	// chunk holds back an incomplete trailing rune until the next token
	// completes it; deliberately invalid bytes still flush once older
	// than one rune, so garbage cannot stall the stream.
	var pend []byte
	push := func(piece string, final bool) {
		pend = append(pend, piece...)
		cut := len(pend)
		if !final {
			for k := 1; k <= 3 && k <= len(pend); k++ {
				b := pend[len(pend)-k]
				if b&0xC0 == 0x80 {
					continue
				}
				var l int
				switch {
				case b < 0x80:
					l = 1
				case b >= 0xF0:
					l = 4
				case b >= 0xE0:
					l = 3
				case b >= 0xC0:
					l = 2
				default:
					l = 1
				}
				if l > k {
					cut = len(pend) - k
				}
				break
			}
		}
		if cut > 0 {
			emit(string(pend[:cut]), final)
			pend = append(pend[:0], pend[cut:]...)
		}
	}

	// Where two prompts part company is worth a checkpoint: the next one
	// to share this opening starts from there instead of from nothing.
	// The KV rows need no copy; only a recurrent state does.
	if mark := commonPrefix(s.cache.live, ids); s.cache.enabled && start == 0 && mark > 0 && mark > len(s.cache.ckpt) {
		s.prefill(ids[:mark], 0)
		s.cache.ckpt = append(s.cache.ckpt[:0], ids[:mark]...)
		if s.cache.hasDelta {
			s.cache.ckptDelta = snapshotDelta(s.model)
		}
		start = mark
	}
	if s.vlog != nil {
		fmt.Fprintf(s.vlog, "prefilling %d tokens (%d already cached)\n", len(ids)-start, start)
	}
	prefillStart := time.Now()
	logits := s.prefill(ids[start:], start)
	if s.vlog != nil && len(ids) > start {
		took := time.Since(prefillStart)
		fmt.Fprintf(s.vlog, "prefilled in %v (%.1f tokens/s)\n",
			took.Round(time.Millisecond), float64(len(ids)-start)/took.Seconds())
	}
	if s.draft != nil {
		s.draft.prefill(ids, 0)
	}
	s.cache.live = append(s.cache.live[:0], ids...)
	steps := len(ids)
	var out []int
	// What the model generates is part of the context a follow-up turn
	// replays, so it belongs in the cache alongside the prompt.
	defer func() { s.cache.live = append(s.cache.live, out...) }()
	finish := "length"
	// One line per request on stderr: when a turn takes half a minute,
	// the reader should be able to see it went to a long prompt, a cold
	// cache, or a long answer — not wonder whether the server hung.
	defer func() {
		fmt.Fprintf(os.Stderr, "%d prompt tokens (%d cached), %d generated, %s in %s\n",
			len(ids), start, len(out), finish, time.Since(began).Round(100*time.Millisecond))
	}()
	ctx := r.Context()
	if s.draft != nil {
		emit := func(next int) bool {
			select {
			case <-ctx.Done():
				return false
			default:
			}
			out = append(out, next)
			if flush != nil {
				push(s.tok.Decode([]int{next}), false)
			}
			return true
		}
		_, _, finish, _ = generateSpeculative(s.model, s.draft, logits, steps, limit,
			s.nCtx, s.specK, temp, topP, func(id int) bool {
				return id == s.imEnd || id == s.eot
			}, rng, emit)
	} else {
		for len(out) < limit && steps < s.nCtx-1 {
			// A disconnected client stops the generation instead of holding
			// the model for tokens nobody will read.
			select {
			case <-ctx.Done():
				finish = "abort"
				steps = s.nCtx // fall out below
				continue
			default:
			}
			next := sample(logits, temp, topP, rng)
			if next == s.imEnd || next == s.eot {
				finish = "stop"
				break
			}
			out = append(out, next)
			if flush != nil {
				push(s.tok.Decode([]int{next}), false)
			}
			logits = s.step(next, steps)
			steps++
		}
	}
	if finish == "abort" {
		return
	}

	content := s.tok.Decode(out)
	reasoning := ""
	if s.tm.reasonOpen != "" {
		reasoning, content = splitReasoning(content, s.tm.reasonOpen, s.tm.reasonClose)
	}
	var calls []toolCall
	if len(tools) > 0 {
		if s.tm.toolCalls == "gemma4" {
			content, calls = parseGemma4ToolCalls(content)
			if len(calls) == 0 {
				content, calls = parseLooseGemma4Calls(content, tools)
			}
		} else if s.tm.toolCalls == "qwen3xml" {
			content, calls = parseXMLToolCalls(content, tools)
		} else {
			content, calls = parseToolCalls(content)
		}
		if len(calls) == 0 {
			content, calls = parseLooseToolCalls(content, tools)
		}
		if len(calls) > 0 {
			finish = "tool_calls"
		}
	}
	if req.Stream {
		if len(pend) > 0 {
			push("", true)
		}
		// Text held back on a marker that never became a call still
		// belongs to the client.
		if len(calls) == 0 && len(content) > streamed {
			flush(content[streamed:])
		}
		// The calls go out as deltas of their own, one chunk each and
		// complete, so a client accumulating fragments by index ends up
		// with the same thing either way.
		for _, c := range calls {
			chunk := map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": created,
				"model": "tensai",
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{"tool_calls": []toolCall{c}},
				}},
			}
			raw, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", raw)
		}
		final := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created,
			"model": "tensai",
			"choices": []map[string]any{{
				"index": 0, "delta": map[string]any{}, "finish_reason": finish,
			}},
		}
		raw, _ := json.Marshal(final)
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", raw)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		return
	}
	msg := map[string]any{"role": "assistant", "content": content}
	if reasoning != "" {
		msg["reasoning_content"] = reasoning
	}
	if len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	writeJSON(w, map[string]any{
		"id": id, "object": "chat.completion", "created": created,
		"model": "tensai",
		"choices": []map[string]any{{
			"index":         0,
			"message":       msg,
			"finish_reason": finish,
		}},
		"usage": map[string]any{
			"prompt_tokens":     len(ids),
			"completion_tokens": len(out),
			"total_tokens":      len(ids) + len(out),
		},
	})
}

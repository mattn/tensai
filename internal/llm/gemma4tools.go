package llm

// Gemma 4's tool convention, which is neither the ChatML families' JSON
// nor Qwen3.5's XML: signatures, calls, and results are written in a
// small brace DSL where a string is wrapped in <|"|> instead of quoted
// and an object key is bare. A call and the result answering it also
// stay inside one model turn, which the renderer has to keep open.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// The DSL's string delimiter, on both sides of the value.
const g4quote = `<|"|>`

// gemma4ToolPreamble writes the signatures into the system turn, one
// <|tool> block apiece, the way the model's own template does.
func gemma4ToolPreamble(tools []toolDef) string {
	var b strings.Builder
	for _, t := range tools {
		b.WriteString("<|tool>declaration:" + t.Function.Name +
			"{description:" + g4string(t.Function.Description))
		var params g4schema
		if len(t.Function.Parameters) > 0 && json.Unmarshal(t.Function.Parameters, &params) == nil {
			b.WriteString(",parameters:{")
			if len(params.Properties) > 0 {
				b.WriteString("properties:{" + gemma4Props(params.Properties) + "},")
			}
			if len(params.Required) > 0 {
				b.WriteString("required:[" + g4strings(params.Required) + "],")
			}
			// The template hangs the closing brace off the type, which
			// every function schema has; without one, close it anyway
			// rather than write an unbalanced signature.
			if params.Type != "" {
				b.WriteString("type:" + g4string(strings.ToUpper(params.Type)))
			}
			b.WriteString("}")
		}
		b.WriteString("}<tool|>")
	}
	return b.String()
}

// g4schema is the part of a JSON Schema the convention writes out.
type g4schema struct {
	Description string                     `json:"description"`
	Type        string                     `json:"type"`
	Enum        json.RawMessage            `json:"enum"`
	Items       map[string]json.RawMessage `json:"items"`
	Nullable    bool                       `json:"nullable"`
	Properties  map[string]json.RawMessage `json:"properties"`
	Required    []string                   `json:"required"`
}

// gemma4Props writes a property map, each property a brace block that
// ends with its uppercased type, the properties themselves in sorted
// order (the template sorts, and the order is tokens the model reads).
func gemma4Props(props map[string]json.RawMessage) string {
	var b strings.Builder
	for i, name := range sortedKeys(props) {
		var p g4schema
		if json.Unmarshal(props[name], &p) != nil {
			continue
		}
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(name + ":{")
		parts := []string{}
		if p.Description != "" {
			parts = append(parts, "description:"+g4string(p.Description))
		}
		switch strings.ToUpper(p.Type) {
		case "STRING":
			if len(p.Enum) > 0 {
				parts = append(parts, "enum:"+g4value(p.Enum, true))
			}
		case "ARRAY":
			if len(p.Items) > 0 {
				parts = append(parts, "items:{"+gemma4Items(p.Items)+"}")
			}
		}
		if p.Nullable {
			parts = append(parts, "nullable:true")
		}
		if strings.ToUpper(p.Type) == "OBJECT" {
			if len(p.Properties) > 0 {
				parts = append(parts, "properties:{"+gemma4Props(p.Properties)+"}")
			}
			if len(p.Required) > 0 {
				parts = append(parts, "required:["+g4strings(p.Required)+"]")
			}
		}
		parts = append(parts, "type:"+g4string(strings.ToUpper(p.Type)))
		b.WriteString(strings.Join(parts, ",") + "}")
	}
	return b.String()
}

// gemma4Items writes an array property's item schema, which the template
// walks key by key rather than through the property shape: properties,
// required, and type are spelled out, and anything else goes through as
// an ordinary value.
func gemma4Items(items map[string]json.RawMessage) string {
	var parts []string
	for _, key := range sortedKeys(items) {
		raw := items[key]
		if len(raw) == 0 || string(raw) == "null" {
			continue
		}
		switch key {
		case "properties":
			var props map[string]json.RawMessage
			if json.Unmarshal(raw, &props) != nil {
				continue
			}
			var required []string
			if r, ok := items["required"]; ok {
				_ = json.Unmarshal(r, &required)
			}
			parts = append(parts, "properties:{"+gemma4Props(props)+"}")
		case "required":
			var required []string
			if json.Unmarshal(raw, &required) != nil {
				continue
			}
			parts = append(parts, "required:["+g4strings(required)+"]")
		case "type":
			var one string
			if json.Unmarshal(raw, &one) == nil {
				parts = append(parts, "type:"+g4string(strings.ToUpper(one)))
				continue
			}
			var many []string
			if json.Unmarshal(raw, &many) != nil {
				continue
			}
			for i := range many {
				many[i] = strings.ToUpper(many[i])
			}
			parts = append(parts, "type:["+g4strings(many)+"]")
		default:
			parts = append(parts, key+":"+g4value(raw, true))
		}
	}
	return strings.Join(parts, ",")
}

// gemma4Call writes one call the way the model emits it, which is also
// how a call replayed from history has to read.
func gemma4Call(name, args string) string {
	return "<|tool_call>call:" + name + "{" + gemma4Args(args) + "}<tool_call|>"
}

// gemma4Args turns the API's JSON argument object into the DSL's brace
// body: bare keys in sorted order, values in the DSL's own spelling.
// Anything that is not an object is passed through as it stands, which
// is what the template does with a pre-serialized argument string.
func gemma4Args(args string) string {
	args = strings.TrimSpace(args)
	if args == "" || args == "{}" {
		return ""
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal([]byte(args), &obj) != nil {
		return strings.TrimSuffix(strings.TrimPrefix(args, "{"), "}")
	}
	var parts []string
	for _, k := range sortedKeys(obj) {
		parts = append(parts, k+":"+g4value(obj[k], false))
	}
	return strings.Join(parts, ",")
}

// gemma4Response writes one tool result. The template treats a result
// that is not an object as a single value, which is what an OpenAI tool
// message carries: a string, JSON or not.
func gemma4Response(name, content string) string {
	return "<|tool_response>response:" + name + "{value:" + g4string(content) + "}<tool_response|>"
}

// g4string wraps a string in the DSL's delimiters. There is no escape:
// the closing delimiter ends the value, so a string containing one is
// the one thing the convention cannot say, and it is dropped rather
// than left to truncate everything after it.
func g4string(s string) string {
	return g4quote + strings.ReplaceAll(s, g4quote, "") + g4quote
}

// g4strings writes a comma-separated list of DSL strings.
func g4strings(ss []string) string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = g4string(s)
	}
	return strings.Join(out, ",")
}

// g4value writes one JSON value in the DSL. Keys of a nested object are
// wrapped like strings in a signature and left bare in an argument,
// which is the one place the two spellings differ.
func g4value(raw json.RawMessage, escapeKeys bool) string {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	var v any
	if dec.Decode(&v) != nil {
		return strings.TrimSpace(string(raw))
	}
	return g4any(v, escapeKeys)
}

func g4any(v any, escapeKeys bool) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case string:
		return g4string(x)
	case bool:
		if x {
			return "true"
		}
		return "false"
	case json.Number:
		return x.String()
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, len(keys))
		for i, k := range keys {
			key := k
			if escapeKeys {
				key = g4string(k)
			}
			parts[i] = key + ":" + g4any(x[k], escapeKeys)
		}
		return "{" + strings.Join(parts, ",") + "}"
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = g4any(e, escapeKeys)
		}
		return "[" + strings.Join(parts, ",") + "]"
	}
	return fmt.Sprint(v)
}

func sortedKeys(m map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// parseGemma4ToolCalls pulls the calls out of a finished turn, turning
// each brace body back into the JSON object the API hands a caller. A
// call cut short by the token limit still yields what it got through.
func parseGemma4ToolCalls(s string) (string, []toolCall) {
	const open, close = "<|tool_call>", "<tool_call|>"
	if !strings.Contains(s, open) {
		return s, nil
	}
	// The answer is what comes before the first call. What follows one
	// is the turn's own scaffolding -- further calls, and the model
	// opening a <|tool_response> it expects to be handed -- rather than
	// anything addressed to the caller.
	content, rest := s, s
	var calls []toolCall
	for {
		i := strings.Index(rest, open)
		if i < 0 {
			break
		}
		rest = rest[i+len(open):]
		body := rest
		if j := strings.Index(rest, close); j >= 0 {
			body, rest = rest[:j], rest[j+len(close):]
		} else {
			rest = ""
		}
		name, args, ok := parseGemma4Call(body)
		if !ok {
			continue
		}
		if len(calls) == 0 {
			// Trim the answer at this call, wherever it turned out to be.
			if j := strings.Index(s, open+body); j >= 0 {
				content = s[:j]
			}
		}
		idx := len(calls)
		calls = append(calls, toolCall{
			Index:    &idx,
			ID:       fmt.Sprintf("call_%d", idx),
			Type:     "function",
			Function: callFunc{Name: name, Arguments: args},
		})
	}
	if len(calls) == 0 {
		// Nothing parsed as a call, so the marker was the model talking
		// about one: the reply is the reply.
		return s, nil
	}
	return strings.TrimSpace(content), calls
}

// parseGemma4Call reads one call:name{...} body back into a name and a
// JSON argument object.
func parseGemma4Call(body string) (name, args string, ok bool) {
	body = strings.TrimSpace(body)
	body = strings.TrimPrefix(body, "call:")
	i := strings.Index(body, "{")
	if i < 0 {
		return "", "", false
	}
	name = strings.TrimSpace(body[:i])
	if name == "" {
		return "", "", false
	}
	v, _ := g4parse(body[i:], true)
	if v == "" {
		v = "{}"
	}
	return name, v, true
}

// g4parse reads one DSL value off the front of s and returns its JSON
// form together with what is left. bare says whether object keys come
// without the string delimiters, which they do everywhere a model
// writes them.
func g4parse(s string, bare bool) (string, string) {
	s = strings.TrimLeft(s, " \t\n")
	switch {
	case strings.HasPrefix(s, g4quote):
		rest := s[len(g4quote):]
		body := rest
		if j := strings.Index(rest, g4quote); j >= 0 {
			body, rest = rest[:j], rest[j+len(g4quote):]
		} else {
			rest = ""
		}
		out, _ := json.Marshal(body)
		return string(out), rest
	case strings.HasPrefix(s, "{"):
		rest := s[1:]
		var parts []string
		for {
			rest = strings.TrimLeft(rest, " \t\n,")
			if rest == "" {
				break
			}
			if rest[0] == '}' {
				rest = rest[1:]
				break
			}
			var key string
			if !bare && strings.HasPrefix(rest, g4quote) {
				key, rest = g4parse(rest, bare)
				key = strings.Trim(key, `"`)
			} else {
				// A bare key runs to its colon, which is what the
				// convention's own grammar says.
				j := strings.IndexAny(rest, ":}")
				if j < 0 || rest[j] == '}' {
					rest = ""
					break
				}
				key, rest = strings.TrimSpace(rest[:j]), rest[j:]
			}
			rest = strings.TrimPrefix(strings.TrimLeft(rest, " \t\n"), ":")
			var val string
			val, rest = g4parse(rest, bare)
			k, _ := json.Marshal(key)
			parts = append(parts, string(k)+":"+val)
		}
		return "{" + strings.Join(parts, ",") + "}", rest
	case strings.HasPrefix(s, "["):
		rest := s[1:]
		var parts []string
		for {
			rest = strings.TrimLeft(rest, " \t\n,")
			if rest == "" {
				break
			}
			if rest[0] == ']' {
				rest = rest[1:]
				break
			}
			var val string
			val, rest = g4parse(rest, bare)
			parts = append(parts, val)
		}
		return "[" + strings.Join(parts, ",") + "]", rest
	}
	// A bare literal runs to the end of the value it sits in.
	j := strings.IndexAny(s, ",}]")
	if j < 0 {
		j = len(s)
	}
	lit := strings.TrimSpace(s[:j])
	rest := s[j:]
	switch lit {
	case "true", "false", "null":
		return lit, rest
	}
	if isJSONNumber(lit) {
		return lit, rest
	}
	out, _ := json.Marshal(lit)
	return string(out), rest
}

// isJSONNumber reports whether a bare literal can go through as a JSON
// number rather than as a string.
func isJSONNumber(s string) bool {
	if s == "" {
		return false
	}
	var n json.Number
	return json.Unmarshal([]byte(s), &n) == nil
}

// gemma4ToolName is the function a result answers, which the convention
// writes into the result itself: the id it carries names a call in an
// earlier turn, and only that call knows the name.
func gemma4ToolName(before []chatMessage, m chatMessage) string {
	for i := len(before) - 1; i >= 0; i-- {
		for _, c := range before[i].ToolCalls {
			if c.ID != "" && c.ID == m.ToolCallID {
				return c.Function.Name
			}
		}
	}
	if m.Name != "" {
		return m.Name
	}
	return "unknown"
}

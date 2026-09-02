package llm

// The tools a command-line run can offer the model. There is no plugin
// mechanism here on purpose: a tool the model may call is something the
// user should be able to name in full, and the list is short.

import (
	"fmt"
	"sort"
	"strings"
)

// builtinTools maps the name -tool takes to the signature the model is
// handed.
var builtinTools = map[string]func() toolDef{
	"wikipedia": wikipediaTool,
}

// ToolNames lists what -tool accepts, for the flag's help text.
func ToolNames() []string {
	names := make([]string, 0, len(builtinTools))
	for name := range builtinTools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// resolveTools turns the comma-separated list into signatures. An
// unknown name is an error rather than a silent omission: a run that
// quietly offered nothing would look like a model that refused to call.
func resolveTools(list string) ([]toolDef, error) {
	var tools []toolDef
	for _, name := range strings.Split(list, ",") {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		make, ok := builtinTools[name]
		if !ok {
			return nil, fmt.Errorf("no tool named %q; -tool takes %s",
				name, strings.Join(ToolNames(), ", "))
		}
		tools = append(tools, make())
	}
	return tools, nil
}

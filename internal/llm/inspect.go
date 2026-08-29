package llm

// Capabilities of a model already on disk, answered the same way the
// server answers them at load time, so a listing predicts what serve
// will actually do rather than guessing at what the model is good at.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/mattn/tensai/encoding/gguf"
)

// Caps says what serve would do with a model, not how well it would do
// it: Tools is whether a request offering tools is accepted rather than
// refused, and Think whether -think leaves the model a block to reason
// in. Both follow the loader exactly, family fallback included, so a
// checkpoint whose template is not on disk gets the same answer here as
// it would get there.
type Caps struct {
	Tools bool
	Think bool
}

// Inspect reads the capabilities of a cached model: a directory holding
// a config.json, or a .gguf file. Anything else, or a checkpoint whose
// architecture the loader does not speak, comes back as neither.
func Inspect(path string) Caps {
	info, err := os.Stat(path)
	if err != nil {
		return Caps{}
	}
	var style, tpl string
	if info.IsDir() {
		raw, err := os.ReadFile(filepath.Join(path, "config.json"))
		if err != nil {
			return Caps{}
		}
		var cfg struct {
			ModelType string `json:"model_type"`
		}
		if json.Unmarshal(raw, &cfg) != nil {
			return Caps{}
		}
		style = cfg.ModelType
		tpl = chatTemplate("", path, true)
	} else {
		if !strings.HasSuffix(path, ".gguf") {
			return Caps{}
		}
		g, err := gguf.Open(path)
		if err != nil {
			return Caps{}
		}
		defer g.Close()
		style, _ = g.String("general.architecture")
		tpl, _ = g.String("tokenizer.chat_template")
		// The same turn markers the loader reads the style from.
		if strings.Contains(tpl, "<｜User｜>") {
			style = "deepseek"
		} else if strings.Contains(tpl, "[INST]") {
			style = "mistral"
		}
	}
	if style == "" {
		return Caps{}
	}
	tm := templateFor(style, true)
	c := Caps{Tools: tm.toolCalls != "", Think: tm.reasonOpen != ""}
	// A template on disk has the last word on tools, exactly as it does
	// at load time; without one the family answer stands, there too.
	if c.Tools && tpl != "" && !templateTakesTools(tpl) {
		c.Tools = false
	}
	return c
}

// String renders the capabilities for a listing, naming what is there
// rather than tabulating a column of yes and no.
func (c Caps) String() string {
	var have []string
	if c.Tools {
		have = append(have, "tools")
	}
	if c.Think {
		have = append(have, "think")
	}
	if len(have) == 0 {
		return "-"
	}
	return strings.Join(have, " ")
}

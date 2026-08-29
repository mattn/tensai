package llm

import (
	"os"
	"path/filepath"
	"testing"
)

func modelDir(t *testing.T, modelType, template string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"),
		[]byte(`{"model_type":"`+modelType+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if template != "" {
		body := `{"chat_template":` + quoteJSON(template) + `}`
		if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func quoteJSON(s string) string {
	out := []byte{'"'}
	for _, r := range []byte(s) {
		switch r {
		case '"', '\\':
			out = append(out, '\\', r)
		case '\n':
			out = append(out, '\\', 'n')
		default:
			out = append(out, r)
		}
	}
	return string(append(out, '"'))
}

func TestInspect(t *testing.T) {
	for _, tt := range []struct {
		name      string
		modelType string
		template  string
		want      string
	}{
		{"qwen2 with a tools template", "qwen2", qwenTemplate, "tools"},
		{"qwen3 thinks as well", "qwen3", qwenTemplate, "tools think"},
		{"smollm3 thinks as well", "smollm3", qwenTemplate, "tools think"},
		// A template that never branches on tools takes the capability
		// away, whatever the family speaks.
		{"chatml family, no tools in template", "llama", smolTemplate, "-"},
		// No template on disk: the family answer stands, which is what
		// the loader falls back to as well.
		{"no template, chatml family", "qwen2", "", "tools"},
		{"no template, thinking family", "qwen3", "", "tools think"},
		// Families whose calling convention tensai does not render stay
		// off even when their own template has one.
		{"mistral has tools it cannot use here", "mistral", qwenTemplate, "-"},
		{"gemma3", "gemma3", qwenTemplate, "-"},
		{"deepseek thinks but takes no tools", "deepseek", qwenTemplate, "think"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := Inspect(modelDir(t, tt.modelType, tt.template)).String(); got != tt.want {
				t.Errorf("Inspect = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestInspectRejectsNonModels(t *testing.T) {
	// A dataset directory, a missing path, and a file that is not a
	// checkpoint all answer "nothing", never a guess.
	empty := t.TempDir()
	for _, path := range []string{empty, filepath.Join(empty, "nope"), filepath.Join(empty, "x.txt")} {
		if got := Inspect(path).String(); got != "-" {
			t.Errorf("Inspect(%s) = %q, want %q", path, got, "-")
		}
	}
}

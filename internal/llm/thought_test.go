package llm

import (
	"strings"
	"testing"
)

// The reasoning channel does not belong in the terminal unless it was
// asked for. Gemma 4 opens one in front of every answer, often empty,
// and it arrives a token at a time with the markers split across writes.
func TestThoughtFilter(t *testing.T) {
	for _, tt := range []struct {
		name  string
		parts []string
		want  string
	}{
		{"an empty block in front of the answer",
			[]string{"<|channel>thought\n", "<channel|>", "Tokyo", " it is."}, "Tokyo it is."},
		{"a block with something in it",
			[]string{"<|channel>thought\nlet me think", "<channel|>", "the answer"}, "the answer"},
		{"no block at all", []string{"just ", "an answer"}, "just an answer"},
		{"markers split across writes",
			[]string{"<|chan", "nel>thou", "ght\nhmm<chan", "nel|>done"}, "done"},
		{"a block the model never closed",
			[]string{"<|channel>thought\nstill thinking"}, ""},
		{"text before the block survives",
			[]string{"hello <|channel>thought\nx<channel|> world"}, "hello  world"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var out strings.Builder
			f := &thoughtFilter{w: &out, open: "<|channel>thought\n", close: "<channel|>"}
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

package tokenizer

import (
	"reflect"
	"testing"
)

// TestO200KSplit pins the o200k scanner's pre-tokenization, verified
// token-exact against llama-tokenize (--no-escape) with gpt-oss-20b's
// vocabulary over a torture corpus and a fuzz corpus.
func TestO200KSplit(t *testing.T) {
	tok := &Tokenizer{cfg: o200kConfig}
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"camelCase", []string{"camel", "Case"}},
		{"HTTPResponse", []string{"HTTPResponse"}},
		{"I'm can't THEY'LL", []string{"I'm", " can't", " THEY'LL"}},
		{" 'standalone", []string{" '", "standalone"}},
		{"日本語abc", []string{"日本語abc"}},
		{"abc日本語", []string{"abc日本語"}},
		{"12345", []string{"123", "45"}},
		{"a=1234", []string{"a", "=", "123", "4"}},
		{"!!\n/next", []string{"!!\n/", "next"}},
		{"paths/like/this", []string{"paths", "/like", "/this"}},
		{"  x", []string{" ", " x"}},
		{"x  ", []string{"x", "  "}},
		{"line1\nline2", []string{"line", "1", "\n", "line", "2"}},
	} {
		if got := tok.split(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("split(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

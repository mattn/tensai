package tokenizer

import (
	"reflect"
	"testing"
)

// TestK2Split pins the K2-Horizon scanner's pre-tokenization. The
// expectations come from the Hugging Face tokenizer for IFM/K2-Horizon-7B
// (tokenizers 0.23, pre_tokenize_str on NFC input); the same scanner
// then matched it token for token on a 25-line corpus and on 412 fuzzed
// lines of marks, joiners, emoji and scripts, with the only differences
// the NFC normalizer this package leaves to the caller.
func TestK2Split(t *testing.T) {
	tok := &Tokenizer{cfg: k2Config}
	for _, c := range []struct {
		in   string
		want []string
	}{
		// Combining marks stay inside the word; cl100k would split
		// "नमस्ते" at the virama.
		{"नमस्ते दुनिया", []string{"नमस्ते", " दुनिया"}},
		{"मैं ठीक हूँ", []string{"मैं", " ठीक", " हूँ"}},
		{"שָׁלוֹם", []string{"שָׁלוֹם"}},
		{"مرحبا بالعالم", []string{"مرحبا", " بالعالم"}},
		// The joiners belong to the word class, so an emoji followed by
		// one is taken as the optional prefix plus a one-joiner word:
		// the family sequence splits per person, as the regex says.
		{"👨\u200d👩\u200d👧 family", []string{"👨\u200d", "👩\u200d", "👧", " family"}},
		{"abc\u200ddef", []string{"abc\u200ddef"}},
		{"x\u200cy", []string{"x\u200cy"}},
		{"\u200d", []string{"\u200d"}},
		{"्", []string{"्"}},
		// Everything cl100k does is unchanged.
		{"I'm can't", []string{"I", "'m", " can", "'t"}},
		{"café naïve", []string{"café", " naïve"}},
		{"日本語abc", []string{"日本語abc"}},
		{"12345", []string{"123", "45"}},
		{"a=1234", []string{"a", "=", "123", "4"}},
		{"  x", []string{" ", " x"}},
		{"line1\nline2", []string{"line", "1", "\n", "line", "2"}},
	} {
		if got := tok.split(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("split(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// The K2 regex is recognized from its word class.
func TestClassifyK2Regex(t *testing.T) {
	re := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	cfg, err := classifyRegex(re)
	if err != nil {
		t.Fatal(err)
	}
	if cfg != k2Config {
		t.Fatalf("classifyRegex = %+v, want %+v", cfg, k2Config)
	}
}

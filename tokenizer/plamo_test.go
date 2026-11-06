package tokenizer

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// plamoFixture is a tiny PLaMo vocabulary: a control token, a few
// scored pieces, and the 256 byte pieces every PLaMo vocabulary has.
func plamoFixture(t *testing.T) *Tokenizer {
	t.Helper()
	tokens := []string{"<|plamo:unk|>", "<|plamo:bos|>", "<|plamo:tag|>", "a", "b", "ab", "abc", "c", " ", "  "}
	scores := []float64{0, 0, 0, -1, -1, -2, -2.5, -1, -1, -1}
	types := []string{"UNKNOWN", "CONTROL", "CONTROL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL"}
	for b := range 256 {
		tokens = append(tokens, fmt.Sprintf("<0x%02X>", b))
		scores = append(scores, 0)
		types = append(types, "BYTE")
	}
	tok, err := NewPlamo(tokens, scores, types, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestPlamoEncode(t *testing.T) {
	tok := plamoFixture(t)
	byteID := func(b byte) int { return 10 + int(b) }
	for _, tt := range []struct {
		in   string
		want []int
	}{
		// "ab" (-2) ties "a"+"b" (-2): the longer piece wins.
		{"ab", []int{5}},
		// "abc" (-2.5) beats "ab"+"c" (-3).
		{"abc", []int{6}},
		// Control tokens match verbatim, and "<|plamo:...|>" shapes split
		// the text around them.
		{"a<|plamo:tag|>b", []int{3, 2, 4}},
		{"<|plamo:bos|>abc", []int{1, 6}},
		// A character no piece covers is spelled in UTF-8 bytes.
		{"aé", []int{3, byteID(0xC3), byteID(0xA9)}},
		// Two spaces are fenced off, so "a" and "b" cannot absorb them;
		// the run itself is one piece.
		{"a  b", []int{3, 9, 4}},
		// Four repeats of one character are fenced: "aaaab" can not end in
		// "ab".
		{"aaaab", []int{3, 3, 3, 3, 4}},
		{"aaab", []int{3, 3, 5}},
	} {
		got := tok.Encode(tt.in)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("Encode(%q) = %v, want %v", tt.in, got, tt.want)
		}
		if dec := tok.Decode(got); dec != tt.in {
			t.Errorf("Decode(Encode(%q)) = %q", tt.in, dec)
		}
	}
	if id, ok := tok.ID("<|plamo:tag|>"); !ok || id != 2 {
		t.Errorf("ID(<|plamo:tag|>) = %d, %v", id, ok)
	}
	if n := tok.Len(); n != 266 {
		t.Errorf("Len = %d, want 266", n)
	}
}

func TestPlamoSegments(t *testing.T) {
	tok := plamoFixture(t)
	for _, tt := range []struct {
		in   string
		want []string
	}{
		{"x<|plamo:foo|>y", []string{"x", "<|plamo:foo|>", "y"}},
		// Whitespace or a '|' inside is no special-token shape.
		{"x<|plamo:a b|>y", []string{"x<|plamo:a b|>y"}},
		{"<|plamo:a|b|>", []string{"<|plamo:a|b|>"}},
		{"<|plamo:<|plamo:x|>", []string{"<|plamo:", "<|plamo:x|>"}},
		{"x   y", []string{"x", "   ", "y"}},
		{"ーーーー!", []string{"ーーーー", "!"}},
		// Newlines are exempt from the repeated-character fence.
		{"a\n\n\n\nb", []string{"a\n\n\n\nb"}},
	} {
		if got := tok.plamoSegments(tt.in); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("plamoSegments(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestPlamoRealVocab round-trips text through PLaMo 3's own vocabulary
// when a download of it is around.
func TestPlamoRealVocab(t *testing.T) {
	dir, err := os.UserCacheDir()
	if err != nil {
		t.Skip(err)
	}
	path := filepath.Join(dir, "tensai", "pfnet", "plamo-3-translate-0.8b", "tokenizer.jsonl")
	if _, err := os.Stat(path); err != nil {
		t.Skip("no PLaMo 3 tokenizer.jsonl in the cache")
	}
	tok, err := LoadPlamo(path)
	if err != nil {
		t.Fatal(err)
	}
	if id, ok := tok.ID("<|plamo:msg|>"); !ok || id != 19 {
		t.Errorf("ID(<|plamo:msg|>) = %d, %v", id, ok)
	}
	for _, s := range []string{
		"会議は来週の月曜日に変更されました。",
		"<|plamo:tag|>input<|plamo:msg|>Hello, world!<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>",
		"emoji 🎉🧑‍💻 and 𠮷野家",
		"def f(x):\n    return x  # comment\n",
		"ーーーーー！！！！？？",
	} {
		ids := tok.Encode(s)
		if got := tok.Decode(ids); got != s {
			t.Errorf("round trip of %q gave %q (%v)", s, got, ids)
		}
	}
}

package tokenizer

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
	"unicode"
)

// o200k's split, as tiktoken and the Hugging Face tokenizer.json spell it:
//
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	|\p{N}{1,3}
//	| ?[^\s\p{L}\p{N}]+[\r\n/]*
//	|\s*[\r\n]+
//	|\s+(?!\S)
//	|\s+
//
// Every alternative but the last two is a plain RE2 expression, so the
// reference below runs them with regexp and hand-rolls only the two
// whitespace ones, whose (?!\S) RE2 cannot express. That gives the
// hand-written scanner something independent to be checked against --
// the Python reference in verify_hf.py needs a network and a pip
// install, and this does not.
var canonicalAlts = []*regexp.Regexp{
	regexp.MustCompile(`^[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`),
	regexp.MustCompile(`^[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`),
	regexp.MustCompile(`^\p{N}{1,3}`),
	regexp.MustCompile(`^ ?[^\s\p{L}\p{N}]+[\r\n/]*`),
	regexp.MustCompile(`^\s*[\r\n]+`),
}

func canonicalSplit(s string) []string {
	var out []string
	for len(s) > 0 {
		matched := false
		for _, re := range canonicalAlts {
			if m := re.FindString(s); m != "" {
				out = append(out, m)
				s = s[len(m):]
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		// `\s+(?!\S)` then `\s+`. A greedy \s run is followed by a
		// non-space or by end of text, so the lookahead alternative
		// takes the run minus its last rune when something follows, and
		// the whole run at the end; the bare \s+ then takes whatever is
		// left.
		rs := []rune(s)
		n := 0
		for n < len(rs) && unicode.IsSpace(rs[n]) {
			n++
		}
		if n == 0 {
			panic("canonicalSplit: no alternative matched " + s)
		}
		take := n
		if n < len(rs) && n > 1 {
			take = n - 1
		}
		out = append(out, string(rs[:take]))
		s = string(rs[take:])
	}
	return out
}

// corpus stresses the classes where llama.cpp's ASCII-range spelling of
// o200k and tiktoken's category spelling disagree, alongside the cases
// they share.
var canonicalCorpus = []string{
	// Scripts whose NFC form keeps combining marks -- the whole point.
	"नमस्ते दुनिया",
	"ที่นี่มีอะไร",
	"עִבְרִית עם ניקוד",
	"العربية مع تشكيل",
	"한국어 텍스트",
	"မြန်မာဘာသာ",
	"ਪੰਜਾਬੀ",
	"தமிழ் மொழி",
	// Latin, cased and accented.
	"Hello, I'm a language model,",
	"camelCase HTTPResponse XMLHttpRequest",
	"café naïve résumé Ærø",
	"ÉCOLE Ünicode ĲSSELMEER",
	"I'LL and I'll and it's and IT'S and won't and WON'T",
	// Digits, punctuation, whitespace, the o200k slash tail.
	"numbers 123 4567 89 0 12345678901234567890",
	"punct!!! ...---??? ***(())[]{}",
	"paths/like/this and //double// and /",
	"  leading and   multiple   gaps  ",
	"tabs\tand\nnewlines\r\nmixed \n\n whitespace \t\n x",
	"trailing newline\n",
	"\n\n\nleading newlines",
	"   ",
	"",
	// Marks and symbols on their own, and mixed scripts.
	"́standalone combining",
	"emoji 👍🏽 and 🇯🇵 flags",
	"日本語abc mixed 한글 and ไทย",
	"math ∑∫√ and ©®™",
	"áb́ć",
}

func TestO200KCanonicalSplit(t *testing.T) {
	tok := &Tokenizer{cfg: o200kConfig}
	for _, in := range canonicalCorpus {
		want := canonicalSplit(in)
		got := tok.split(in)
		if len(want) == 0 && len(got) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("split(%q)\n got %q\nwant %q", in, got, want)
		}
		if strings.Join(got, "") != in {
			t.Errorf("split(%q) does not reassemble: %q", in, strings.Join(got, ""))
		}
	}
}

// TestO200KCanonicalPatternAccepted pins that the tokenizer.json spelling
// gpt-oss and the Hugging Face o200k models ship resolves to the o200k
// scanner rather than being rejected.
func TestO200KCanonicalPatternAccepted(t *testing.T) {
	const harmony = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	cfg, err := classifyRegex(harmony)
	if err != nil {
		t.Fatalf("classifyRegex(o200k_harmony) = %v", err)
	}
	if cfg != o200kConfig {
		t.Errorf("classifyRegex(o200k_harmony) = %+v, want o200kConfig", cfg)
	}
}

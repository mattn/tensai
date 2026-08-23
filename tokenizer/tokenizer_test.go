package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fixture builds a tiny byte-level BPE tokenizer.json. Ġ (U+0120) is the
// byte-level stand-in for a space.
func fixture(pre string, merges string) []byte {
	return []byte(`{
		"added_tokens": [{"id": 12, "content": "<|end|>", "special": true}],
		"normalizer": null,
		"pre_tokenizer": ` + pre + `,
		"model": {
			"type": "BPE",
			"vocab": {"h":0,"e":1,"l":2,"o":3,"Ġ":4,"1":5,"2":6,"3":7,"he":8,"ll":9,"hell":10,"hello":11,"12":13,"123":14},
			"merges": ` + merges + `
		}
	}`)
}

const gpt2Pre = `{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true}`
const cl100kPre = `{"type": "Sequence", "pretokenizers": [
	{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
	{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]}`
const stringMerges = `["h e", "l l", "he ll", "hell o", "1 2", "12 3"]`
const pairMerges = `[["h","e"], ["l","l"], ["he","ll"], ["hell","o"], ["1","2"], ["12","3"]]`

func TestEncodeDecode(t *testing.T) {
	for _, merges := range []string{stringMerges, pairMerges} {
		tok, err := Parse(fixture(gpt2Pre, merges))
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := tok.Encode("hello hello")
		// "hello" merges to one token; " hello" pre-tokenizes with its
		// leading space, whose Ġ does not merge further.
		want := []int{11, 4, 11}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("encode: got %v want %v", got, want)
		}
		if dec := tok.Decode(got); dec != "hello hello" {
			t.Fatalf("decode: %q", dec)
		}
	}
}

func TestSpecialTokens(t *testing.T) {
	tok, err := Parse(fixture(gpt2Pre, stringMerges))
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("hello<|end|>hello")
	want := []int{11, 12, 11}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("encode with special: got %v want %v", got, want)
	}
	if dec := tok.Decode(got); dec != "hello<|end|>hello" {
		t.Fatalf("decode with special: %q", dec)
	}
	if id, ok := tok.ID("<|end|>"); !ok || id != 12 {
		t.Fatalf("ID lookup: %d %v", id, ok)
	}
	if tok.Len() != 15 {
		t.Fatalf("len: %d", tok.Len())
	}
}

func TestDigitSplitting(t *testing.T) {
	// The gpt2 pattern keeps a digit run as one pre-token, so the merges
	// build "123"; the cl100k pattern with \p{N} isolates each digit, so
	// no cross-token merge can happen.
	g, err := Parse(fixture(gpt2Pre, stringMerges))
	if err != nil {
		t.Fatal(err)
	}
	if got := g.Encode("123"); !reflect.DeepEqual(got, []int{14}) {
		t.Fatalf("gpt2 digits: %v", got)
	}
	c, err := Parse(fixture(cl100kPre, stringMerges))
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Encode("123"); !reflect.DeepEqual(got, []int{5, 6, 7}) {
		t.Fatalf("cl100k digits: %v", got)
	}
}

func TestParseErrors(t *testing.T) {
	if _, err := Parse([]byte(`{`)); err == nil {
		t.Fatal("expected error for bad json")
	}
	if _, err := Parse(fixture(`{"type": "Whitespace"}`, stringMerges)); err == nil {
		t.Fatal("expected error for unsupported pre_tokenizer")
	}
	if _, err := Parse(fixture(`{"type": "Split", "pattern": {"Regex": "[a-z]+"}}`, stringMerges)); err == nil {
		t.Fatal("expected error for unknown split pattern")
	}
	bad := []byte(`{"normalizer": {"type": "Lowercase"}, "pre_tokenizer": ` + gpt2Pre + `,
		"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}`)
	if _, err := Parse(bad); err == nil {
		t.Fatal("expected error for unsupported normalizer")
	}
	nfc := []byte(`{"normalizer": {"type": "NFC"}, "pre_tokenizer": ` + gpt2Pre + `,
		"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}`)
	if _, err := Parse(nfc); err != nil {
		t.Fatalf("NFC must pass through: %v", err)
	}
}

// TestAgainstHF compares against reference encodings from the Python
// tokenizers library when they are present; generate them with
// verify_hf.py (see its docstring) into testdata/.
func TestAgainstHF(t *testing.T) {
	for _, name := range []string{"gpt2", "qwen", "smollm2"} {
		tokPath := filepath.Join("testdata", name+".json")
		refPath := filepath.Join("testdata", "ref_"+name+".json")
		if _, err := os.Stat(tokPath); err != nil {
			t.Skipf("no %s; run verify_hf.py to generate testdata", tokPath)
		}
		tok, err := Load(tokPath)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		blob, err := os.ReadFile(refPath)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var ref struct {
			Corpus []string `json:"corpus"`
			IDs    [][]int  `json:"ids"`
		}
		if err := json.Unmarshal(blob, &ref); err != nil {
			t.Fatal(err)
		}
		for i, s := range ref.Corpus {
			got := tok.Encode(s)
			if !reflect.DeepEqual(got, ref.IDs[i]) {
				t.Errorf("%s: %q\n got  %v\n want %v", name, s, got, ref.IDs[i])
			}
			if dec := tok.Decode(got); dec != s {
				t.Errorf("%s decode: %q -> %q", name, s, dec)
			}
		}
	}
}

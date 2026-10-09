package tokenizer

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// unigramFixture is a tiny Unigram tokenizer.json with the normalizer
// LLM-jp ships: U+2581 in front of every segment and for every space.
func unigramFixture() []byte {
	pieces := []string{
		`["<unk>", 0.0]`, `["<|start|>", 0.0]`, `["<|message|>", 0.0]`,
		`["▁", -2.0]`, `["a", -3.0]`, `["b", -3.0]`, `["▁a", -2.5]`,
		`["▁ab", -6.0]`, `["ab", -2.0]`, `["▁b", -4.0]`, `["c", -3.0]`,
	}
	for b := range 256 {
		pieces = append(pieces, fmt.Sprintf(`["<0x%02X>", 0.0]`, b))
	}
	return []byte(`{
		"added_tokens": [
			{"id": 0, "content": "<unk>", "special": true},
			{"id": 1, "content": "<|start|>", "special": true},
			{"id": 2, "content": "<|message|>", "special": true}],
		"normalizer": {"type": "Sequence", "normalizers": [
			{"type": "Replace", "pattern": {"Regex": "(?<!\\n)^"}, "content": "▁"},
			{"type": "Replace", "pattern": {"Regex": " "}, "content": "▁"}]},
		"pre_tokenizer": null,
		"model": {"type": "Unigram", "unk_id": 0, "byte_fallback": true,
			"vocab": [` + strings.Join(pieces, ",") + `]}
	}`)
}

func TestUnigramEncode(t *testing.T) {
	tok, err := Parse(unigramFixture())
	if err != nil {
		t.Fatal(err)
	}
	const (
		sp, a, b, spA, spAB, ab, spB, c = 3, 4, 5, 6, 7, 8, 9, 10
		byteBase                        = 11
	)
	for _, tc := range []struct {
		in   string
		want []int
	}{
		// "▁ab" scores -6 alone, "▁a"+"b" -5.5, "▁"+"ab" -4: the best
		// sum wins, not the longest piece.
		{"ab", []int{sp, ab}},
		{"a b", []int{spA, spB}},
		// é is in no piece: it costs the lowest score less ten, and
		// comes out as its UTF-8 bytes.
		{"cé", []int{sp, c, byteBase + 0xC3, byteBase + 0xA9}},
		// Every segment between added tokens gets its own U+2581.
		{"<|start|>a<|message|>b", []int{1, spA, 2, spB}},
		{"", nil},
	} {
		if got := tok.Encode(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Encode(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// The space a segment's U+2581 decodes to comes off at the start and
// after an added token, and a stream of single tokens decoded with
// DecodeNext adds up to the whole run's Decode.
func TestUnigramDecode(t *testing.T) {
	tok, err := Parse(unigramFixture())
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range []string{"a b", "<|start|>a<|message|>b ab", "cé a"} {
		ids := tok.Encode(in)
		whole := tok.Decode(ids)
		if whole != in {
			t.Errorf("Decode(Encode(%q)) = %q", in, whole)
		}
		var stream strings.Builder
		prev := -1
		for _, id := range ids {
			stream.WriteString(tok.DecodeNext(prev, id))
			prev = id
		}
		if stream.String() != whole {
			t.Errorf("streamed %q, whole %q", stream.String(), whole)
		}
	}
}

func TestUnigramRefusesOtherNormalizers(t *testing.T) {
	raw := strings.Replace(string(unigramFixture()), `"content": "▁"},
			{"type": "Replace"`, `"content": "_"},
			{"type": "Replace"`, 1)
	if _, err := Parse([]byte(raw)); err == nil {
		t.Error("a normalizer the tokenizer does not implement was accepted")
	}
}

// A sequence fed in pieces, each encoded after the token the last one
// ended in, comes out as the whole encoded at once.
func TestUnigramEncodeAfter(t *testing.T) {
	tok, err := Parse(unigramFixture())
	if err != nil {
		t.Fatal(err)
	}
	// The pieces break where the whole would break anyway: no piece
	// spans them, so only the U+2581 a segment start adds can differ.
	for _, parts := range [][]string{
		{"a", " b"},
		{"<|start|>a", "<|message|>b"},
		{"c", "<|message|>a", " b"},
	} {
		var whole string
		var fed []int
		for _, p := range parts {
			whole += p
			prev := -1
			if len(fed) > 0 {
				prev = fed[len(fed)-1]
			}
			fed = append(fed, tok.EncodeAfter(prev, p)...)
		}
		if want := tok.Encode(whole); !reflect.DeepEqual(fed, want) {
			t.Errorf("%q fed in pieces = %v, whole = %v", parts, fed, want)
		}
	}
}

// metaspaceFixture is unigramFixture with Ruri v3's spelling: no
// normalizer, and a Metaspace pre-tokenizer that turns spaces into
// U+2581 without putting one in front.
func metaspaceFixture() []byte {
	raw := string(unigramFixture())
	i := strings.Index(raw, `"normalizer"`)
	j := strings.Index(raw, `"model"`)
	return []byte(raw[:i] + `"normalizer": null,
		"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": false},
		` + raw[j:])
}

// Without the front U+2581 the same vocabulary splits differently, and
// the GGUF arrays build the same tokenizer as the tokenizer.json does.
func TestUnigramNoPrefix(t *testing.T) {
	fromJSON, err := Parse(metaspaceFixture())
	if err != nil {
		t.Fatal(err)
	}
	pieces := []string{"<unk>", "<|start|>", "<|message|>", "▁", "a", "b", "▁a", "▁ab", "ab", "▁b", "c"}
	scores := []float32{0, 0, 0, -2, -3, -3, -2.5, -6, -2, -4, -3}
	types := []int32{spmControl, spmControl, spmControl, 1, 1, 1, 1, 1, 1, 1, 1}
	for b := range 256 {
		pieces = append(pieces, fmt.Sprintf("<0x%02X>", b))
		scores = append(scores, 0)
		types = append(types, spmByte)
	}
	fromGGUF, err := NewUnigram(pieces, scores, types, false)
	if err != nil {
		t.Fatal(err)
	}
	const (
		a, b, spB, ab, c = 4, 5, 9, 8, 10
		byteBase         = 11
	)
	for _, tc := range []struct {
		in   string
		want []int
	}{
		{"ab", []int{ab}},
		{"a b", []int{a, spB}},
		{"cé", []int{c, byteBase + 0xC3, byteBase + 0xA9}},
		{"<|start|>a<|message|>b", []int{1, a, 2, b}},
	} {
		for name, tok := range map[string]*Tokenizer{"json": fromJSON, "gguf": fromGGUF} {
			got := tok.Encode(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("%s: Encode(%q) = %v, want %v", name, tc.in, got, tc.want)
			}
			if back := tok.Decode(got); back != tc.in {
				t.Errorf("%s: Decode(Encode(%q)) = %q", name, tc.in, back)
			}
		}
	}
}

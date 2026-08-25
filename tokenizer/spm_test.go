package tokenizer

import "testing"

// A tiny hand-built SentencePiece vocabulary: scores steer the merges.
func testSPM(t *testing.T, pre bool) *Tokenizer {
	t.Helper()
	tokens := []string{"<unk>", "<s>", "</s>", "▁", "a", "b", "c", "ab", "abc", "▁ab", "▁abc", "<0x41>", "<0x42>", "<0xE3>", "<0x81>", "<0x82>"}
	scores := []float32{0, 0, 0, -3, -1, -1, -1, -2, -1.5, -2.5, -1.2, 0, 0, 0, 0, 0}
	types := []int32{2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6}
	tok, err := NewSPM(tokens, scores, types, pre)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestSPMEncode(t *testing.T) {
	tok := testSPM(t, false)

	// "abc" merges all the way: ab (score -2) then abc (-1.5), because the
	// best-scoring bigram wins each round.
	if got := tok.Encode("abc"); len(got) != 1 || got[0] != 8 {
		t.Fatalf("abc: %v", got)
	}
	// " abc" becomes ▁abc via the ▁ab / ▁abc chain.
	if got := tok.Encode(" abc"); len(got) != 1 || got[0] != 10 {
		t.Fatalf(" abc: %v", got)
	}
	// Specials match verbatim and split the text around them.
	if got := tok.Encode("<s>abc</s>"); len(got) != 3 || got[0] != 1 || got[1] != 8 || got[2] != 2 {
		t.Fatalf("specials: %v", got)
	}
	// Out-of-vocabulary characters fall back to bytes: "A" -> <0x41>,
	// and あ (E3 81 82) -> three byte tokens.
	if got := tok.Encode("A"); len(got) != 1 || got[0] != 11 {
		t.Fatalf("byte fallback: %v", got)
	}
	if got := tok.Encode("あ"); len(got) != 3 || got[0] != 13 || got[1] != 14 || got[2] != 15 {
		t.Fatalf("multibyte fallback: %v", got)
	}

	// Decode round-trips text, spaces and byte tokens included.
	for _, s := range []string{"abc", " abc", "A", "あ", "<s>ab c</s>"} {
		if got := tok.Decode(tok.Encode(s)); got != s {
			t.Fatalf("roundtrip %q: got %q", s, got)
		}
	}
}

func TestSPMSpacePrefix(t *testing.T) {
	tok := testSPM(t, true)
	// With add_space_prefix the fragment gains a leading space, so "abc"
	// encodes like " abc".
	if got := tok.Encode("abc"); len(got) != 1 || got[0] != 10 {
		t.Fatalf("prefixed abc: %v", got)
	}
}

func TestSPMArrayMismatch(t *testing.T) {
	if _, err := NewSPM([]string{"a"}, nil, nil, false); err == nil {
		t.Fatal("expected error for mismatched arrays")
	}
}

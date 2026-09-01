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

// A tiny SentencePiece-over-BPE vocabulary, the shape Gemma 4 ships:
// merge ranks instead of unigram scores, and the ▁ standing in for a
// space is part of the merges rather than a separate token.
func testSPMBPE(t *testing.T, pre bool) *Tokenizer {
	t.Helper()
	tokens := []string{"<unk>", "<turn|>", "▁", "a", "b", "c", "\n", "ab", "abc", "▁a", "▁ab", "<0x41>", "<0xE3>", "<0x81>", "<0x82>", "<div>"}
	types := []int32{2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 1}
	merges := []string{"a b", "ab c", "▁ a", "▁a b", "▁ ab"}
	tok, err := NewSPMBPE(tokens, merges, types, pre)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestSPMBPEEncode(t *testing.T) {
	tok := testSPMBPE(t, false)

	// The merges run in rank order: a+b first, then ab+c.
	if got := tok.Encode("abc"); len(got) != 1 || got[0] != 8 {
		t.Fatalf("abc: %v", got)
	}
	// A space normalizes to ▁ and merges with what follows.
	if got := tok.Encode(" ab"); len(got) != 1 || got[0] != 10 {
		t.Fatalf(" ab: %v", got)
	}
	// Newlines break the text up; nothing merges across one.
	if got := tok.Encode("ab\nab"); len(got) != 3 || got[0] != 7 || got[1] != 6 || got[2] != 7 {
		t.Fatalf("newline: %v", got)
	}
	// A control token matches verbatim, but an ordinary token that merely
	// looks like one is merged like any other text -- Gemma 4's vocabulary
	// spells <div> and <=> exactly this way.
	if got := tok.Encode("ab<turn|>"); len(got) != 2 || got[0] != 7 || got[1] != 1 {
		t.Fatalf("control token: %v", got)
	}
	if got := tok.Encode("<div>"); len(got) == 1 && got[0] == 15 {
		t.Fatalf("<div> was matched as a special: %v", got)
	}
	// Anything the merges leave out of the vocabulary falls back to bytes.
	if got := tok.Encode("A"); len(got) != 1 || got[0] != 11 {
		t.Fatalf("byte fallback: %v", got)
	}
	if got := tok.Encode("あ"); len(got) != 3 || got[0] != 12 || got[1] != 13 || got[2] != 14 {
		t.Fatalf("multibyte fallback: %v", got)
	}

	for _, s := range []string{"abc", " ab", "ab\nab", "A", "あ", "ab<turn|>"} {
		if got := tok.Decode(tok.Encode(s)); got != s {
			t.Fatalf("roundtrip %q: got %q", s, got)
		}
	}
}

func TestSPMBPESpacePrefix(t *testing.T) {
	tok := testSPMBPE(t, true)
	if got := tok.Encode("ab"); len(got) != 1 || got[0] != 10 {
		t.Fatalf("prefixed ab: %v", got)
	}
}

func TestSPMBPEArrayMismatch(t *testing.T) {
	if _, err := NewSPMBPE([]string{"a"}, nil, nil, false); err == nil {
		t.Fatal("expected error for mismatched arrays")
	}
	if _, err := NewSPMBPE([]string{"a"}, []string{"ab"}, []int32{1}, false); err == nil {
		t.Fatal("expected error for a malformed merge")
	}
}

package main

// The GPT-2 byte-level BPE tokenizer, from vocab.json and merges.txt: bytes
// are mapped to printable unicode stand-ins, pre-tokenized with the GPT-2
// splitting rules, and merged greedily by rank.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

type tokenizer struct {
	vocab    map[string]int
	inverse  map[int]string
	ranks    map[[2]string]int
	byteEnc  [256]rune
	byteDec  map[rune]byte
	wordSeen map[string][]int // per-word BPE cache
}

func newTokenizer(vocabPath, mergesPath string) (*tokenizer, error) {
	t := &tokenizer{
		vocab:    map[string]int{},
		inverse:  map[int]string{},
		ranks:    map[[2]string]int{},
		byteDec:  map[rune]byte{},
		wordSeen: map[string][]int{},
	}

	raw, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &t.vocab); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", vocabPath, err)
	}
	for s, id := range t.vocab {
		t.inverse[id] = s
	}

	f, err := os.Open(mergesPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	rank := 0
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		t.ranks[[2]string{parts[0], parts[1]}] = rank
		rank++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	// bytes_to_unicode: printable bytes map to themselves, the rest to
	// 256 and up, so every byte has a visible stand-in character.
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff) {
			t.byteEnc[b] = rune(b)
		} else {
			t.byteEnc[b] = rune(256 + n)
			n++
		}
		t.byteDec[t.byteEnc[b]] = byte(b)
	}
	return t, nil
}

// preTokenize reproduces the GPT-2 splitting regex
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// which Go's regexp cannot express because of the lookahead.
func preTokenize(s string) []string {
	rs := []rune(s)
	var out []string
	i := 0
	for i < len(rs) {
		// Contractions.
		if rs[i] == '\'' {
			rest := string(rs[i:])
			matched := false
			for _, c := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
				if strings.HasPrefix(rest, c) {
					out = append(out, c)
					i += len([]rune(c))
					matched = true
					break
				}
			}
			if matched {
				continue
			}
		}
		start := i
		j := i
		if rs[j] == ' ' && j+1 < len(rs) && !unicode.IsSpace(rs[j+1]) {
			j++ // the optional leading space of the next class
		}
		switch {
		case j < len(rs) && unicode.IsLetter(rs[j]):
			for j < len(rs) && unicode.IsLetter(rs[j]) {
				j++
			}
			out = append(out, string(rs[start:j]))
		case j < len(rs) && unicode.IsNumber(rs[j]):
			for j < len(rs) && unicode.IsNumber(rs[j]) {
				j++
			}
			out = append(out, string(rs[start:j]))
		case j < len(rs) && !unicode.IsSpace(rs[j]):
			for j < len(rs) && !unicode.IsSpace(rs[j]) && !unicode.IsLetter(rs[j]) && !unicode.IsNumber(rs[j]) && rs[j] != '\'' {
				j++
			}
			if j == start || (j == start+1 && rs[start] == ' ') {
				// An apostrophe that did not start a contraction.
				j++
			}
			out = append(out, string(rs[start:j]))
		default:
			// Whitespace run. \s+(?!\S) keeps the last whitespace char
			// for the following token when something follows.
			for j < len(rs) && unicode.IsSpace(rs[j]) {
				j++
			}
			if j < len(rs) && j-start > 1 {
				j--
			}
			if j > start {
				out = append(out, string(rs[start:j]))
			} else {
				j++ // lone space directly before another space-led class
				out = append(out, string(rs[start:j]))
			}
		}
		i = j
	}
	return out
}

// bpe merges one pre-token's stand-in characters by rank.
func (t *tokenizer) bpe(word string) []int {
	if ids, ok := t.wordSeen[word]; ok {
		return ids
	}
	parts := make([]string, 0, len(word))
	for _, r := range word {
		parts = append(parts, string(r))
	}
	for len(parts) > 1 {
		best, bestRank := -1, int(^uint(0)>>1)
		for i := 0; i < len(parts)-1; i++ {
			if r, ok := t.ranks[[2]string{parts[i], parts[i+1]}]; ok && r < bestRank {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		merged := parts[best] + parts[best+1]
		parts = append(parts[:best], append([]string{merged}, parts[best+2:]...)...)
	}
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		if id, ok := t.vocab[p]; ok {
			ids = append(ids, id)
		}
	}
	t.wordSeen[word] = ids
	return ids
}

// Encode turns text into GPT-2 token ids.
func (t *tokenizer) Encode(s string) []int {
	var ids []int
	for _, word := range preTokenize(s) {
		var sb strings.Builder
		for _, b := range []byte(word) {
			sb.WriteRune(t.byteEnc[b])
		}
		ids = append(ids, t.bpe(sb.String())...)
	}
	return ids
}

// Decode turns token ids back into text.
func (t *tokenizer) Decode(ids []int) string {
	var bs []byte
	for _, id := range ids {
		for _, r := range t.inverse[id] {
			bs = append(bs, t.byteDec[r])
		}
	}
	return string(bs)
}

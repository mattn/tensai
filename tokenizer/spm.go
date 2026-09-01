package tokenizer

import (
	"container/heap"
	"fmt"
	"strings"
)

// SentencePiece support: the tokenizer family Gemma and Llama-2-era
// models ship (GGUF calls it "llama"). Instead of byte-level BPE ranks,
// the vocabulary carries a score per piece; encoding greedily merges the
// adjacent pair whose concatenation has the highest score, spaces are
// rewritten to the visible U+2581, and anything that ends up outside the
// vocabulary falls back to per-byte <0xXX> tokens.

// spmToken classifies GGUF token types.
const (
	spmNormal  = 1
	spmUnknown = 2
	spmControl = 3
	spmUserDef = 4
	spmByte    = 6
)

// NewSPM builds a SentencePiece tokenizer from parallel vocabulary
// arrays — tokens, their scores, and their GGUF token types. Control and
// user-defined tokens match verbatim in the input like the BPE
// tokenizers' added tokens; addSpacePrefix prepends a space before
// encoding, the way most Llama-2-family models expect (Gemma does not).
func NewSPM(tokens []string, scores []float32, types []int32, addSpacePrefix bool) (*Tokenizer, error) {
	if len(tokens) != len(scores) || len(tokens) != len(types) {
		return nil, fmt.Errorf("tokenizer: spm arrays disagree: %d tokens, %d scores, %d types",
			len(tokens), len(scores), len(types))
	}
	t := &Tokenizer{
		vocab:       make(map[string]int, len(tokens)),
		inverse:     make(map[int]string, len(tokens)),
		byID:        map[int]string{},
		ranks:       map[[2]string]int{},
		byteDec:     map[rune]byte{},
		cache:       map[string][]int{},
		spm:         true,
		scores:      scores,
		spacePrefix: addSpacePrefix,
		unkID:       -1,
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for id, tok := range tokens {
		switch types[id] {
		case spmControl, spmUserDef:
			t.specials = append(t.specials, special{content: tok, id: id})
			t.byID[id] = tok
		case spmUnknown:
			t.unkID = id
			t.inverse[id] = tok
		case spmByte:
			var b byte
			if _, err := fmt.Sscanf(tok, "<0x%02X>", &b); err == nil {
				t.byteID[b] = id
			}
			t.inverse[id] = tok
		default:
			t.vocab[tok] = id
			t.inverse[id] = tok
		}
	}
	sortSpecials(t)
	return t, nil
}

// spmSymbol is one live segment during merging; merged-away segments get
// size 0 and stay linked out.
type spmSymbol struct {
	text       string
	prev, next int
}

// spmBigram is a candidate merge in the priority queue.
type spmBigram struct {
	left, right int
	score       float32
	size        int // combined byte length when queued, to detect staleness
}

type spmQueue []spmBigram

func (q spmQueue) Len() int { return len(q) }
func (q spmQueue) Less(i, j int) bool {
	if q[i].score != q[j].score {
		return q[i].score > q[j].score
	}
	return q[i].left < q[j].left // ties resolve left-most, like the reference
}
func (q spmQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *spmQueue) Push(x any)   { *q = append(*q, x.(spmBigram)) }
func (q *spmQueue) Pop() any     { old := *q; n := len(old); v := old[n-1]; *q = old[:n-1]; return v }

// spmEncode tokenizes one special-free fragment.
func (t *Tokenizer) spmEncode(s string) []int {
	if s == "" {
		return nil
	}
	if t.spacePrefix {
		s = " " + s
	}
	s = strings.ReplaceAll(s, " ", "▁")

	// One symbol per UTF-8 character to start.
	var syms []spmSymbol
	for i, r := range s {
		_ = r
		if len(syms) > 0 {
			syms[len(syms)-1].next = len(syms)
		}
		end := i + len(string(r))
		syms = append(syms, spmSymbol{text: s[i:end], prev: len(syms) - 1, next: -1})
	}

	q := &spmQueue{}
	tryAdd := func(l, r int) {
		if l < 0 || r < 0 {
			return
		}
		cat := syms[l].text + syms[r].text
		if id, ok := t.vocab[cat]; ok {
			heap.Push(q, spmBigram{left: l, right: r, score: t.scores[id], size: len(cat)})
		}
	}
	for i := 0; i+1 < len(syms); i++ {
		tryAdd(i, i+1)
	}
	for q.Len() > 0 {
		bg := heap.Pop(q).(spmBigram)
		l, r := bg.left, bg.right
		if syms[l].text == "" || syms[r].text == "" || len(syms[l].text)+len(syms[r].text) != bg.size {
			continue // one side was merged away since this was queued
		}
		syms[l].text += syms[r].text
		syms[r].text = ""
		syms[l].next = syms[r].next
		if syms[r].next >= 0 {
			syms[syms[r].next].prev = l
		}
		tryAdd(syms[l].prev, l)
		tryAdd(l, syms[l].next)
	}

	var ids []int
	for i := 0; i >= 0 && i < len(syms); i = syms[i].next {
		if syms[i].text == "" {
			continue
		}
		if id, ok := t.vocab[syms[i].text]; ok {
			ids = append(ids, id)
			continue
		}
		// Byte fallback for pieces outside the vocabulary.
		ok := true
		for _, b := range []byte(syms[i].text) {
			if t.byteID[b] < 0 {
				ok = false
				break
			}
		}
		if ok {
			for _, b := range []byte(syms[i].text) {
				ids = append(ids, t.byteID[b])
			}
		} else if t.unkID >= 0 {
			ids = append(ids, t.unkID)
		}
	}
	return ids
}

// spmDecode renders ids back to text: U+2581 becomes a space, byte
// tokens their byte, specials their literal content.
func (t *Tokenizer) spmDecode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if sp, ok := t.byID[id]; ok {
			sb.WriteString(sp)
			continue
		}
		piece := t.inverse[id]
		var b byte
		if _, err := fmt.Sscanf(piece, "<0x%02X>", &b); err == nil && t.byteID[b] == id {
			sb.WriteByte(b)
			continue
		}
		sb.WriteString(strings.ReplaceAll(piece, "▁", " "))
	}
	return sb.String()
}

// NewSPMBPE builds the tokenizer Gemma 4 ships, which is a SentencePiece
// normalizer over BPE merges rather than over unigram scores: spaces
// become the visible U+2581, there is no word-level pre-splitting, and
// the merges run on raw UTF-8 rather than on byte-encoded stand-ins.
// Only newlines break the text up, because a merge never spans one.
//
// tokens is the vocabulary in id order, merges the "a b" pairs in rank
// order, and types classifies each token the way NewSPM's do.
func NewSPMBPE(tokens, merges []string, types []int32, addSpacePrefix bool) (*Tokenizer, error) {
	if len(tokens) != len(types) {
		return nil, fmt.Errorf("tokenizer: %d tokens but %d types", len(tokens), len(types))
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocabulary")
	}
	t := &Tokenizer{
		vocab:       make(map[string]int, len(tokens)),
		inverse:     make(map[int]string, len(tokens)),
		byID:        map[int]string{},
		ranks:       make(map[[2]string]int, len(merges)),
		byteDec:     map[rune]byte{},
		cache:       map[string][]int{},
		spm:         true,
		spmBPE:      true,
		spacePrefix: addSpacePrefix,
		unkID:       -1,
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for id, tok := range tokens {
		switch types[id] {
		case spmControl, spmUserDef:
			t.specials = append(t.specials, special{content: tok, id: id})
			t.byID[id] = tok
		case spmUnknown:
			t.unkID = id
			t.inverse[id] = tok
		case spmByte:
			var b byte
			if _, err := fmt.Sscanf(tok, "<0x%02X>", &b); err == nil {
				t.byteID[b] = id
			}
			t.inverse[id] = tok
		default:
			t.vocab[tok] = id
			t.inverse[id] = tok
		}
	}
	sortSpecials(t)
	for rank, m := range merges {
		first, second, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("tokenizer: merge %q is not a pair", m)
		}
		t.ranks[[2]string{first, second}] = rank
	}
	return t, nil
}

// spmBPEEncode normalizes the way SentencePiece does and then merges by
// rank. Newlines are the only split: a merge never spans one, so cutting
// there changes nothing and keeps the merge loop off long inputs.
func (t *Tokenizer) spmBPEEncode(s string) []int {
	if s == "" {
		return nil
	}
	if t.spacePrefix && !strings.HasPrefix(s, " ") {
		s = " " + s
	}
	s = strings.ReplaceAll(s, " ", "▁")
	var ids []int
	for len(s) > 0 {
		i := strings.IndexByte(s, '\n')
		switch {
		case i < 0:
			ids = append(ids, t.bpeOrBytes(s)...)
			s = ""
		case i > 0:
			ids = append(ids, t.bpeOrBytes(s[:i])...)
			s = s[i:]
		default:
			j := 0
			for j < len(s) && s[j] == '\n' {
				j++
			}
			ids = append(ids, t.bpeOrBytes(s[:j])...)
			s = s[j:]
		}
	}
	return ids
}

// bpeOrBytes merges one run and falls back to the per-byte tokens for
// anything the merges leave outside the vocabulary.
func (t *Tokenizer) bpeOrBytes(run string) []int {
	parts := t.bpeParts(run)
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		if id, ok := t.vocab[p]; ok {
			ids = append(ids, id)
			continue
		}
		for _, b := range []byte(p) {
			if id := t.byteID[b]; id >= 0 {
				ids = append(ids, id)
			} else if t.unkID >= 0 {
				ids = append(ids, t.unkID)
			}
		}
	}
	return ids
}

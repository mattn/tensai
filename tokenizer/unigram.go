package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// Unigram support: the SentencePiece model tokenizer.json calls
// "Unigram" (LLM-jp-4 ships one). Every piece carries a log probability
// and a text segments into the pieces whose scores sum highest -- a
// Viterbi search over every split, not the greedy pair merging of
// spmEncode, which gives different tokens on the same vocabulary. A
// character no piece covers costs the lowest score less ten, as in
// sentencepiece and HF's tokenizers, and with byte fallback comes out as
// its UTF-8 bytes.

// unigramPenalty is what an uncovered character costs below the lowest
// piece score.
const unigramPenalty = 10

// parseUnigram builds a Unigram tokenizer from tokenizer.json's parts.
// Only the normalizer this family uses is understood: prefix U+2581 and
// turn every space into one. Anything else is refused rather than
// tokenized differently from the model's training.
func parseUnigram(f *jsonFile, rawVocab json.RawMessage) (*Tokenizer, error) {
	var vocab [][2]json.RawMessage
	if err := json.Unmarshal(rawVocab, &vocab); err != nil {
		return nil, fmt.Errorf("tokenizer: parsing unigram vocab: %w", err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}
	if err := checkUnigramNormalizer(f.Normalizer, f.PreTokenizer); err != nil {
		return nil, err
	}
	t := &Tokenizer{
		vocab:    make(map[string]int, len(vocab)),
		inverse:  make([]string, len(vocab)),
		byID:     map[int]string{},
		ranks:    map[[2]string]int{},
		byteDec:  map[rune]byte{},
		cache:    map[string][]int{},
		scores:   make([]float32, len(vocab)),
		unigram:  true,
		unkID:    -1,
		fallback: f.Model.ByteFallback,
	}
	if f.Model.UnkID != nil {
		t.unkID = *f.Model.UnkID
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	added := map[int]bool{}
	for _, at := range f.AddedTokens {
		added[at.ID] = true
	}
	minScore := math.Inf(1)
	for id, entry := range vocab {
		var piece string
		var score float64
		if err := json.Unmarshal(entry[0], &piece); err != nil {
			return nil, fmt.Errorf("tokenizer: unigram piece %d: %w", id, err)
		}
		if err := json.Unmarshal(entry[1], &score); err != nil {
			return nil, fmt.Errorf("tokenizer: unigram score %d: %w", id, err)
		}
		t.inverse[id] = piece
		t.scores[id] = float32(score)
		var b byte
		if len(piece) == 6 && strings.HasPrefix(piece, "<0x") && strings.HasSuffix(piece, ">") {
			if _, err := fmt.Sscanf(piece, "<0x%02X>", &b); err == nil {
				t.byteID[b] = id
			}
		}
		// Added tokens are split out before the search ever sees the
		// text, so they take no part in it.
		if added[id] {
			continue
		}
		if _, dup := t.vocab[piece]; !dup {
			t.vocab[piece] = id
		}
		minScore = min(minScore, score)
		t.maxPiece = max(t.maxPiece, len(piece))
	}
	t.unkScore = minScore - unigramPenalty
	added2 := make([]AddedToken, len(f.AddedTokens))
	for i, at := range f.AddedTokens {
		added2[i] = AddedToken{ID: at.ID, Content: at.Content}
	}
	for _, at := range added2 {
		t.specials = append(t.specials, special{content: at.Content, id: at.ID})
		t.byID[at.ID] = at.Content
	}
	sortSpecials(t)
	return t, nil
}

// checkUnigramNormalizer accepts the sentencepiece-style normalizer: a
// U+2581 put in front of the text and in place of every space, spelled
// as two Replace steps.
func checkUnigramNormalizer(norm, pre json.RawMessage) error {
	if len(pre) > 0 && string(pre) != "null" {
		return fmt.Errorf("tokenizer: unsupported unigram pre_tokenizer %s", pre)
	}
	var n struct {
		Type        string `json:"type"`
		Normalizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex  *string `json:"Regex"`
				String *string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		} `json:"normalizers"`
	}
	if err := json.Unmarshal(norm, &n); err != nil || n.Type != "Sequence" || len(n.Normalizers) != 2 {
		return fmt.Errorf("tokenizer: unsupported unigram normalizer %s", norm)
	}
	pat := func(i int) string {
		p := n.Normalizers[i].Pattern
		switch {
		case p.Regex != nil:
			return *p.Regex
		case p.String != nil:
			return *p.String
		}
		return ""
	}
	if n.Normalizers[0].Type != "Replace" || pat(0) != `(?<!\n)^` || n.Normalizers[0].Content != "▁" ||
		n.Normalizers[1].Type != "Replace" || pat(1) != " " || n.Normalizers[1].Content != "▁" {
		return fmt.Errorf("tokenizer: unsupported unigram normalizer %s", norm)
	}
	return nil
}

// unigramEncode normalizes one text segment (the stretch between two
// added tokens) and finds its highest-scoring split.
func (t *Tokenizer) unigramEncode(s string, segStart bool) []int {
	if s == "" {
		return nil
	}
	s = strings.ReplaceAll(s, " ", "▁")
	if segStart {
		s = "▁" + s
	}
	n := len(s)
	// best[i] is the highest score of any split of s[:i]; from[i] the
	// start of its last piece and id[i] that piece, -1 for a character
	// no piece covers.
	best := make([]float64, n+1)
	from := make([]int, n+1)
	id := make([]int, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	for i := 0; i < n; i++ {
		if math.IsInf(best[i], -1) || !runeStart(s, i) {
			continue
		}
		covered := false
		charEnd := i + runeLen(s, i)
		for j := i + 1; j <= n && j-i <= t.maxPiece; j++ {
			if j < n && !runeStart(s, j) {
				continue
			}
			pid, ok := t.vocab[s[i:j]]
			if !ok {
				continue
			}
			if j == charEnd {
				covered = true
			}
			if sc := best[i] + float64(t.scores[pid]); sc > best[j] {
				best[j], from[j], id[j] = sc, i, pid
			}
		}
		if !covered {
			if sc := best[i] + t.unkScore; sc > best[charEnd] {
				best[charEnd], from[charEnd], id[charEnd] = sc, i, -1
			}
		}
	}
	var rev []int
	for j := n; j > 0; j = from[j] {
		if id[j] >= 0 {
			rev = append(rev, id[j])
			continue
		}
		// The uncovered character, as bytes when the model has them.
		piece := s[from[j]:j]
		ok := t.fallback
		for k := 0; ok && k < len(piece); k++ {
			ok = t.byteID[piece[k]] >= 0
		}
		switch {
		case ok:
			for k := len(piece) - 1; k >= 0; k-- {
				rev = append(rev, t.byteID[piece[k]])
			}
		case t.unkID >= 0:
			rev = append(rev, t.unkID)
		}
	}
	ids := make([]int, len(rev))
	for i, v := range rev {
		ids[len(rev)-1-i] = v
	}
	return ids
}

// unigramDecode renders ids as text the way the decoder sequence does:
// byte tokens fuse into their bytes, U+2581 turns back into a space, and
// the space the normalizer put in front of a segment -- at the start, or
// right after an added token -- comes off again.
func (t *Tokenizer) unigramDecode(ids []int, segStart bool) string {
	var sb strings.Builder
	for _, id := range ids {
		if sp, ok := t.byID[id]; ok {
			sb.WriteString(sp)
			segStart = true
			continue
		}
		piece := t.piece(id)
		if b, ok := t.byteOf(id, piece); ok {
			sb.WriteByte(b)
			segStart = false
			continue
		}
		text := strings.ReplaceAll(piece, "▁", " ")
		if segStart {
			text = strings.TrimPrefix(text, " ")
		}
		sb.WriteString(text)
		segStart = false
	}
	return sb.String()
}

func (t *Tokenizer) byteOf(id int, piece string) (byte, bool) {
	if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") {
		return 0, false
	}
	var b byte
	if _, err := fmt.Sscanf(piece, "<0x%02X>", &b); err != nil || t.byteID[b] != id {
		return 0, false
	}
	return b, true
}

func runeStart(s string, i int) bool { return s[i]&0xC0 != 0x80 }

func runeLen(s string, i int) int {
	j := i + 1
	for j < len(s) && !runeStart(s, j) {
		j++
	}
	return j - i
}

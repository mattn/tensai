// Package tokenizer loads Hugging Face tokenizer.json files and implements
// the byte-level BPE family they describe — the tokenizers of GPT-2,
// Llama 3, Qwen, and most other published byte-level models — with no
// dependencies.
//
// The pre-tokenization patterns these models use need lookahead and inline
// case-insensitive groups that regexp cannot express, so the two patterns
// that exist in the wild — the GPT-2 split and the cl100k-style split —
// are implemented as hand-written scanners; a tokenizer.json with any
// other pattern is rejected rather than silently mis-tokenized.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

type special struct {
	content string
	id      int
}

// Tokenizer encodes text to token ids and back.
type Tokenizer struct {
	vocab       map[string]int
	inverse     map[int]string
	specials    []special // sorted longest-first
	byID        map[int]string
	ranks       map[[2]string]int
	cfg         splitConfig
	spm         bool
	scores      []float32
	byteID      [256]int
	unkID       int
	spacePrefix bool
	byteEnc     [256]rune
	byteDec     map[rune]byte
	cache       map[string][]int
}

// splitConfig selects between the two pre-tokenization scanners.
type splitConfig struct {
	ciContractions bool // (?i:'s|'t|...) instead of case-sensitive
	letterPrefix   bool // [^\r\n\p{L}\p{N}]? before letters instead of " ?"
	maxDigits      int  // 0: gpt2's " ?\p{N}+"; else \p{N}{1,maxDigits}
	newlineRuns    bool // punct tail [\r\n]* and the \s*[\r\n]+ alternative
	// individualDigits models HF's Digits(individual_digits) pre-tokenizer
	// running before the ByteLevel split (SmolLM2 and friends): every digit
	// is its own pre-token, so a digit never absorbs a preceding space, and
	// whitespace running into a digit behaves as if at end of text.
	individualDigits bool
}

var (
	gpt2Config   = splitConfig{}
	cl100kConfig = splitConfig{ciContractions: true, letterPrefix: true, maxDigits: 3, newlineRuns: true}
)

// jsonFile mirrors the subset of tokenizer.json this package understands.
type jsonFile struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`
}

type preTok struct {
	Type             string          `json:"type"`
	UseRegex         *bool           `json:"use_regex"`
	IndividualDigits *bool           `json:"individual_digits"`
	Pretokenizers    json.RawMessage `json:"pretokenizers"`
	Pattern          struct {
		Regex string `json:"Regex"`
	} `json:"pattern"`
}

// Load reads a tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse builds a Tokenizer from tokenizer.json bytes.
func Parse(raw []byte) (*Tokenizer, error) {
	var f jsonFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("tokenizer: parsing json: %w", err)
	}
	// An NFC normalizer passes through: virtually all real-world text is
	// already NFC, and canonical composition tables are not worth a
	// dependency. Callers with decomposed input can pre-normalize it.
	if len(f.Normalizer) > 0 && string(f.Normalizer) != "null" {
		var norm struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(f.Normalizer, &norm); err != nil || norm.Type != "NFC" {
			return nil, fmt.Errorf("tokenizer: unsupported normalizer %s", f.Normalizer)
		}
	}
	if f.Model.Type != "" && f.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q", f.Model.Type)
	}
	if len(f.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}

	t := &Tokenizer{
		vocab:   f.Model.Vocab,
		inverse: make(map[int]string, len(f.Model.Vocab)),
		byID:    map[int]string{},
		ranks:   map[[2]string]int{},
		byteDec: map[rune]byte{},
		cache:   map[string][]int{},
	}
	for s, id := range t.vocab {
		t.inverse[id] = s
	}

	cfg, err := detectSplit(f.PreTokenizer)
	if err != nil {
		return nil, err
	}
	t.cfg = cfg

	if err := t.parseMerges(f.Model.Merges); err != nil {
		return nil, err
	}

	for _, at := range f.AddedTokens {
		t.specials = append(t.specials, special{content: at.Content, id: at.ID})
		t.byID[at.ID] = at.Content
	}
	sortSpecials(t)

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

// detectSplit maps the pre_tokenizer JSON onto one of the two known
// scanners.
func detectSplit(raw json.RawMessage) (splitConfig, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return gpt2Config, fmt.Errorf("tokenizer: missing pre_tokenizer")
	}
	var p preTok
	if err := json.Unmarshal(raw, &p); err != nil {
		return gpt2Config, fmt.Errorf("tokenizer: parsing pre_tokenizer: %w", err)
	}
	switch p.Type {
	case "ByteLevel":
		if p.UseRegex == nil || *p.UseRegex {
			return gpt2Config, nil // ByteLevel's built-in regex is the GPT-2 split
		}
		return gpt2Config, fmt.Errorf("tokenizer: ByteLevel without a split pattern")
	case "Sequence":
		var subs []json.RawMessage
		if err := json.Unmarshal(p.Pretokenizers, &subs); err != nil {
			return gpt2Config, fmt.Errorf("tokenizer: parsing pre_tokenizer sequence: %w", err)
		}
		digits := false
		for _, sub := range subs {
			var sp preTok
			if err := json.Unmarshal(sub, &sp); err != nil {
				continue
			}
			switch {
			case sp.Type == "Split" && sp.Pattern.Regex != "":
				return classifyRegex(sp.Pattern.Regex)
			case sp.Type == "Digits":
				if sp.IndividualDigits == nil || !*sp.IndividualDigits {
					return gpt2Config, fmt.Errorf("tokenizer: Digits without individual_digits is unsupported")
				}
				digits = true
			case sp.Type == "ByteLevel" && (sp.UseRegex == nil || *sp.UseRegex):
				cfg := gpt2Config
				cfg.individualDigits = digits
				return cfg, nil
			}
		}
		return gpt2Config, fmt.Errorf("tokenizer: no Split pattern in pre_tokenizer sequence")
	case "Split":
		return classifyRegex(p.Pattern.Regex)
	}
	return gpt2Config, fmt.Errorf("tokenizer: unsupported pre_tokenizer type %q", p.Type)
}

func classifyRegex(re string) (splitConfig, error) {
	switch {
	case strings.HasPrefix(re, `'s|'t|'re|'ve|'m|'ll|'d`):
		return gpt2Config, nil
	case strings.HasPrefix(re, `(?i:'s|'t|'re|'ve|'m|'ll|'d)`):
		cfg := cl100kConfig
		if strings.Contains(re, `\p{N}{1,3}`) {
			cfg.maxDigits = 3
		} else {
			cfg.maxDigits = 1
		}
		return cfg, nil
	}
	return gpt2Config, fmt.Errorf("tokenizer: unsupported split pattern %q", re)
}

func (t *Tokenizer) parseMerges(raw json.RawMessage) error {
	// merges are serialized either as ["a b", ...] or [["a","b"], ...].
	var asStrings []string
	if err := json.Unmarshal(raw, &asStrings); err == nil {
		for rank, line := range asStrings {
			parts := strings.SplitN(line, " ", 2)
			if len(parts) != 2 {
				return fmt.Errorf("tokenizer: bad merge %q", line)
			}
			t.ranks[[2]string{parts[0], parts[1]}] = rank
		}
		return nil
	}
	var asPairs [][2]string
	if err := json.Unmarshal(raw, &asPairs); err != nil {
		return fmt.Errorf("tokenizer: parsing merges: %w", err)
	}
	for rank, p := range asPairs {
		t.ranks[p] = rank
	}
	return nil
}

// Len reports the number of ids the tokenizer can produce, including added
// tokens.
func (t *Tokenizer) Len() int { return len(t.vocab) + len(t.byID) }

// ID looks up a token by its content — the way to find special tokens like
// "<|im_start|>".
func (t *Tokenizer) ID(content string) (int, bool) {
	for _, sp := range t.specials {
		if sp.content == content {
			return sp.id, true
		}
	}
	id, ok := t.vocab[content]
	return id, ok
}

// Encode turns text into token ids. Special tokens appearing verbatim in
// the text are matched as themselves.
func (t *Tokenizer) Encode(s string) []int {
	var ids []int
	for len(s) > 0 {
		// Find the earliest special token occurrence; specials are sorted
		// longest-first, so overlapping specials resolve to the longest.
		cut, cutEnd, cutID := len(s), len(s), -1
		for _, sp := range t.specials {
			if i := strings.Index(s, sp.content); i >= 0 && i < cut {
				cut, cutEnd, cutID = i, i+len(sp.content), sp.id
			}
		}
		ids = append(ids, t.encodeText(s[:cut])...)
		if cutID >= 0 {
			ids = append(ids, cutID)
		}
		s = s[cutEnd:]
	}
	return ids
}

func (t *Tokenizer) encodeText(s string) []int {
	if t.spm {
		return t.spmEncode(s)
	}
	var ids []int
	for _, word := range t.split(s) {
		var sb strings.Builder
		for _, b := range []byte(word) {
			sb.WriteRune(t.byteEnc[b])
		}
		ids = append(ids, t.bpe(sb.String())...)
	}
	return ids
}

// Decode turns token ids back into text.
func (t *Tokenizer) Decode(ids []int) string {
	if t.spm {
		return t.spmDecode(ids)
	}
	var bs []byte
	for _, id := range ids {
		if sp, ok := t.byID[id]; ok {
			bs = append(bs, sp...)
			continue
		}
		for _, r := range t.inverse[id] {
			bs = append(bs, t.byteDec[r])
		}
	}
	return string(bs)
}

// bpe merges one pre-token's stand-in characters by rank.
func (t *Tokenizer) bpe(word string) []int {
	if ids, ok := t.cache[word]; ok {
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
	t.cache[word] = ids
	return ids
}

// split reproduces the model's pre-tokenization regex with a hand-written
// scanner; alternatives are tried in the regex's order at each position.
func (t *Tokenizer) split(s string) []string {
	rs := []rune(s)
	cfg := t.cfg
	var out []string
	i := 0
	for i < len(rs) {
		// Contractions.
		if rs[i] == '\'' && i+1 < len(rs) {
			if n := matchContraction(rs[i:], cfg.ciContractions); n > 0 {
				out = append(out, string(rs[i:i+n]))
				i += n
				continue
			}
		}
		start := i
		j := i

		// Letters, with either " ?" (gpt2) or "[^\r\n\p{L}\p{N}]?"
		// (cl100k) as the optional prefix.
		pfx := j
		if cfg.letterPrefix {
			if !unicode.IsLetter(rs[j]) && !unicode.IsNumber(rs[j]) && rs[j] != '\r' && rs[j] != '\n' && j+1 < len(rs) && unicode.IsLetter(rs[j+1]) {
				pfx = j + 1
			}
		} else if rs[j] == ' ' && j+1 < len(rs) && unicode.IsLetter(rs[j+1]) {
			pfx = j + 1
		}
		if pfx < len(rs) && unicode.IsLetter(rs[pfx]) {
			j = pfx
			for j < len(rs) && unicode.IsLetter(rs[j]) {
				j++
			}
			out = append(out, string(rs[start:j]))
			i = j
			continue
		}

		// Numbers: " ?\p{N}+" (gpt2) or "\p{N}{1,max}" (cl100k), unless
		// a Digits pre-tokenizer already isolated every digit.
		if cfg.individualDigits && unicode.IsNumber(rs[j]) {
			out = append(out, string(rs[j]))
			i = j + 1
			continue
		}
		if cfg.maxDigits > 0 {
			if unicode.IsNumber(rs[j]) {
				for j < len(rs) && j-start < cfg.maxDigits && unicode.IsNumber(rs[j]) {
					j++
				}
				out = append(out, string(rs[start:j]))
				i = j
				continue
			}
		} else {
			d := j
			// A digit in its own segment never absorbs a space.
			if rs[d] == ' ' && d+1 < len(rs) && unicode.IsNumber(rs[d+1]) && !cfg.individualDigits {
				d++
			}
			if d < len(rs) && unicode.IsNumber(rs[d]) {
				j = d
				for j < len(rs) && unicode.IsNumber(rs[j]) {
					j++
				}
				out = append(out, string(rs[start:j]))
				i = j
				continue
			}
		}

		// Punctuation: " ?[^\s\p{L}\p{N}]+", with "[\r\n]*" appended in
		// the cl100k pattern.
		p := j
		if rs[p] == ' ' && p+1 < len(rs) && isPunct(rs[p+1]) {
			p++
		}
		if p < len(rs) && isPunct(rs[p]) {
			j = p
			for j < len(rs) && isPunct(rs[j]) {
				j++
			}
			if cfg.newlineRuns {
				for j < len(rs) && (rs[j] == '\r' || rs[j] == '\n') {
					j++
				}
			}
			out = append(out, string(rs[start:j]))
			i = j
			continue
		}

		// Whitespace run [start, j).
		for j < len(rs) && unicode.IsSpace(rs[j]) {
			j++
		}
		// "\s*[\r\n]+": up to and including the last newline of the run.
		if cfg.newlineRuns {
			last := -1
			for k := start; k < j; k++ {
				if rs[k] == '\r' || rs[k] == '\n' {
					last = k
				}
			}
			if last >= 0 {
				out = append(out, string(rs[start:last+1]))
				i = last + 1
				continue
			}
		}
		// "\s+(?!\S)" then "\s+": keep the last whitespace char for the
		// following token when something follows — but not for a digit,
		// which under individualDigits sits in its own segment, leaving
		// this run at a segment end.
		if j < len(rs) && j-start > 1 && !(cfg.individualDigits && unicode.IsNumber(rs[j])) {
			j--
		}
		if j == start {
			j++
		}
		out = append(out, string(rs[start:j]))
		i = j
	}
	return out
}

var contractions = []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

func matchContraction(rs []rune, ci bool) int {
	rest := string(rs)
	lower := rest
	if ci {
		lower = strings.ToLower(rest)
	}
	for _, c := range contractions {
		if strings.HasPrefix(lower, c) {
			return len([]rune(c))
		}
	}
	return 0
}

func isPunct(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// sortSpecials orders special tokens longest-first so overlapping matches
// resolve to the longest.
func sortSpecials(t *Tokenizer) {
	sort.Slice(t.specials, func(i, j int) bool {
		return len(t.specials[i].content) > len(t.specials[j].content)
	})
}

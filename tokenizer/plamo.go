package tokenizer

// PLaMo's tokenizer (PLaMo 2 and 3): a tokenizer.jsonl of
// [piece, score, type] rows searched for the split with the highest
// total score. The reference implementation walks an Aho-Corasick table
// from the end of the text; what it computes is a Viterbi search over
// code points with scores rounded to integers (score * 1e4), ties going
// to the longest piece, and every character also reachable as an
// "unknown" piece costing 1e7 that is spelled out as UTF-8 byte tokens.
// Before the search, special tokens of the form <|plamo:...|> and runs
// of repeated characters or spaces are fenced off with a boundary no
// piece may cross, which here simply splits the text into segments
// searched one at a time.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	plamoUnknownCost = 10000000
	plamoMaxSpecial  = 64
)

// plamoConfig is the part of tokenizer_config.json PLaMo's tokenizer reads.
type plamoConfig struct {
	SpacesThreshold   *int `json:"break_around_consecutive_spaces_threshold"`
	RepeatedThreshold *int `json:"break_around_repeated_chars_threshold"`
}

// LoadPlamo reads a PLaMo tokenizer.jsonl. The boundary thresholds come
// from the tokenizer_config.json beside it when there is one.
func LoadPlamo(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows [][3]any
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for n := 1; sc.Scan(); n++ {
		var row []any
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, n, err)
		}
		if len(row) < 3 {
			return nil, fmt.Errorf("%s:%d: want [piece, score, type]", path, n)
		}
		rows = append(rows, [3]any{row[0], row[1], row[2]})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	tokens := make([]string, len(rows))
	scores := make([]float64, len(rows))
	types := make([]string, len(rows))
	for i, r := range rows {
		var ok1, ok2, ok3 bool
		tokens[i], ok1 = r[0].(string)
		scores[i], ok2 = r[1].(float64)
		types[i], ok3 = r[2].(string)
		if !ok1 || !ok2 || !ok3 {
			return nil, fmt.Errorf("%s:%d: malformed row", path, i+1)
		}
	}
	var cfg plamoConfig
	if raw, err := os.ReadFile(filepath.Join(filepath.Dir(path), "tokenizer_config.json")); err == nil {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("tokenizer_config.json: %w", err)
		}
	}
	spaces, repeated := 0, 0
	if cfg.SpacesThreshold != nil {
		spaces = *cfg.SpacesThreshold
	}
	if cfg.RepeatedThreshold != nil {
		repeated = *cfg.RepeatedThreshold
	}
	return NewPlamo(tokens, scores, types, spaces, repeated)
}

// NewPlamo builds PLaMo's tokenizer from its vocabulary rows. types are
// the jsonl's NORMAL, CONTROL, UNKNOWN, and BYTE. spaces and repeated are
// the boundary thresholds (0 disables either).
func NewPlamo(tokens []string, scores []float64, types []string, spaces, repeated int) (*Tokenizer, error) {
	if len(tokens) != len(scores) || len(tokens) != len(types) {
		return nil, fmt.Errorf("tokenizer: plamo arrays disagree")
	}
	t := &Tokenizer{
		vocab:       make(map[string]int, len(tokens)),
		inverse:     make([]string, len(tokens)),
		byID:        map[int]string{},
		ranks:       map[[2]string]int{},
		byteDec:     map[rune]byte{},
		cache:       map[string][]int{},
		plamo:       true,
		plamoCost:   make([]int64, len(tokens)),
		plamoSpaces: spaces,
		plamoRepeat: repeated,
		unkID:       -1,
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for id, piece := range tokens {
		t.inverse[id] = piece
		switch types[id] {
		case "BYTE":
			var b byte
			if _, err := fmt.Sscanf(piece, "<0x%02X>", &b); err != nil {
				return nil, fmt.Errorf("tokenizer: bad byte piece %q", piece)
			}
			t.byteID[b] = id
			continue
		case "CONTROL", "UNKNOWN":
			// Matched verbatim ahead of the search, the way the
			// reference's <|plamo:...|> boundary makes them come out.
			t.specials = append(t.specials, special{content: piece, id: id})
			t.byID[id] = piece
			if types[id] == "UNKNOWN" {
				t.unkID = id
			}
			continue
		}
		if _, dup := t.vocab[piece]; !dup {
			t.vocab[piece] = id
		}
		// Python's round() is half-to-even.
		t.plamoCost[id] = -int64(math.RoundToEven(scores[id] * 1e4))
		t.maxPiece = max(t.maxPiece, len([]rune(piece)))
	}
	for b, id := range t.byteID {
		if id < 0 {
			return nil, fmt.Errorf("tokenizer: no byte piece for <0x%02X>", b)
		}
	}
	sortSpecials(t)
	return t, nil
}

// plamoEncode tokenizes text that holds no added token.
func (t *Tokenizer) plamoEncode(s string) []int {
	var ids []int
	for _, seg := range t.plamoSegments(s) {
		ids = append(ids, t.plamoViterbi(seg)...)
	}
	return ids
}

// plamoSegments cuts s where the reference puts its boundary character:
// around anything shaped like a special token, around runs of one
// character repeated at least plamoRepeat times (newlines excepted, as
// the regex's "." does), and around runs of at least plamoSpaces spaces.
func (t *Tokenizer) plamoSegments(s string) []string {
	var segs []string
	text := func(x string) {
		segs = append(segs, t.plamoRuns(x)...)
	}
	start := 0
	for i := 0; i < len(s); {
		j := strings.Index(s[i:], "<|plamo:")
		if j < 0 {
			break
		}
		i += j
		if end := plamoSpecialEnd(s[i:]); end > 0 {
			text(s[start:i])
			segs = append(segs, s[i:i+end])
			i += end
			start = i
			continue
		}
		i++
	}
	text(s[start:])
	return segs
}

// plamoSpecialEnd is the length of the special-token shape s starts
// with, 0 if it starts with none: <|plamo: then up to 64 characters
// that are neither '|' nor whitespace, then |>.
func plamoSpecialEnd(s string) int {
	const open = "<|plamo:"
	n := 0
	for j, r := range s[len(open):] {
		if r == '|' {
			if strings.HasPrefix(s[len(open)+j:], "|>") {
				return len(open) + j + 2
			}
			return 0
		}
		if isPlamoSpace(r) || n == plamoMaxSpecial {
			return 0
		}
		n++
	}
	return 0
}

// plamoRuns fences off runs of repeated characters and of spaces.
func (t *Tokenizer) plamoRuns(s string) []string {
	var segs []string
	add := func(x []rune) {
		if len(x) > 0 {
			segs = append(segs, string(x))
		}
	}
	rs := []rune(s)
	start := 0
	for i := 0; i < len(rs); {
		j := i + 1
		for j < len(rs) && rs[j] == rs[i] {
			j++
		}
		run := j - i
		if (t.plamoRepeat > 0 && rs[i] != '\n' && run >= t.plamoRepeat) ||
			(t.plamoSpaces > 0 && rs[i] == ' ' && run >= t.plamoSpaces) {
			add(rs[start:i])
			add(rs[i:j])
			start = j
		}
		i = j
	}
	add(rs[start:])
	return segs
}

func isPlamoSpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r', '\f', '\v', 0x1c, 0x1d, 0x1e, 0x1f, 0x85:
		return true
	}
	return r > 0x7f && strings.ContainsRune("                 　", r)
}

// plamoViterbi finds the cheapest split of one segment, searching from
// the end as the reference does so that ties resolve the same way: at
// each position the longest piece wins among equal costs.
func (t *Tokenizer) plamoViterbi(s string) []int {
	rs := []rune(s)
	n := len(rs)
	if n == 0 {
		return nil
	}
	cost := make([]int64, n+1)
	plen := make([]int, n+1)
	pid := make([]int, n+1)
	for i := range cost[:n] {
		cost[i] = math.MaxInt64
	}
	// Byte offsets of each rune, for slicing pieces without allocating.
	off := make([]int, n+1)
	for i, b := 0, 0; i < n; i++ {
		off[i] = b
		b += len(string(rs[i]))
		off[i+1] = b
	}
	for i := n - 1; i >= 0; i-- {
		for l := min(t.maxPiece, n-i); l >= 1; l-- {
			id, ok := t.vocab[s[off[i]:off[i+l]]]
			if !ok {
				continue
			}
			if c := cost[i+l] + t.plamoCost[id]; c < cost[i] {
				cost[i], plen[i], pid[i] = c, l, id
			}
		}
		if c := cost[i+1] + plamoUnknownCost; c < cost[i] {
			cost[i], plen[i], pid[i] = c, 1, -1
		}
	}
	var ids []int
	for i := 0; i < n; i += plen[i] {
		if pid[i] >= 0 {
			ids = append(ids, pid[i])
			continue
		}
		for _, b := range []byte(string(rs[i])) {
			ids = append(ids, t.byteID[b])
		}
	}
	return ids
}

// plamoDecode concatenates pieces, byte tokens fusing into their bytes.
func (t *Tokenizer) plamoDecode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		piece := t.piece(id)
		if b, ok := t.byteOf(id, piece); ok {
			sb.WriteByte(b)
			continue
		}
		sb.WriteString(piece)
	}
	return sb.String()
}

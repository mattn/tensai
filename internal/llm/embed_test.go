package llm

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/tokenizer"
)

// tableRows serves an in-memory embedding table the way a gguf does.
type tableRows struct{ m *tensai.Matrix }

func (r tableRows) TensorRows(_ string, from, to int) (*tensai.Tensor, error) {
	c := r.m.Cols
	return tensai.NewTensorFromSlice(r.m.Data[from*c:to*c], to-from, c)
}

// tinyEmbedder is a two-layer encoder over random weights, the first
// layer global and the second local, with a five-piece vocabulary.
func tinyEmbedder(t *testing.T) *Embedder {
	t.Helper()
	rng := rand.New(rand.NewPCG(1, 2))
	mat := func(r, c int) *tensai.Matrix {
		m := tensai.NewMatrix(r, c)
		for i := range m.Data {
			m.Data[i] = float32(rng.NormFloat64() * 0.3)
		}
		return m
	}
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1 + float32(rng.NormFloat64()*0.1)
		}
		return v
	}
	const hs, heads, ff = 16, 2, 24
	pieces := []string{"<s>", "</s>", "a", "b", "▁"}
	tok, err := tokenizer.NewUnigram(pieces, []float32{0, 0, -1, -1, -2}, []int32{3, 3, 1, 1, 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	e := &Embedder{
		tok: tok, emb: newEmbedTable(tableRows{mat(len(pieces), hs)}, ""),
		hs: hs, heads: heads, headSz: hs / heads, ff: ff, eps: 1e-5,
		window: 3, maxPos: 64, vocab: len(pieces), bos: 0, eos: 1,
		embNrm: ones(hs), outNrm: ones(hs),
	}
	for i := 0; i < 2; i++ {
		b := embedBlock{
			ffnNorm: ones(hs),
			wQKV:    mat(hs, 3*hs), wOut: mat(hs, hs),
			wUp: mat(hs, 2*ff), wDown: mat(ff, hs),
			local: i == 1,
			freq:  make([]float64, e.headSz/2),
		}
		if i > 0 {
			b.attnNorm = ones(hs)
		}
		theta := 160000.0
		if b.local {
			theta = 10000
		}
		for j := range b.freq {
			b.freq[j] = math.Pow(theta, -2*float64(j)/float64(e.headSz))
		}
		e.blocks = append(e.blocks, b)
	}
	return e
}

// attend agrees with attention written out the plain way: NEOX
// rotation by position within the token's own sequence, then a softmax
// over that sequence's keys within the window, both ends included, or
// over all of them on a global layer. Two sequences share the batch.
func TestEmbedAttendMatchesNaive(t *testing.T) {
	e := tinyEmbedder(t)
	const n = 11
	rng := rand.New(rand.NewPCG(3, 4))
	for _, b := range e.blocks {
		qkv := tensai.NewMatrix(n, 3*e.hs)
		for i := range qkv.Data {
			qkv.Data[i] = float32(rng.NormFloat64())
		}
		orig := append([]float32(nil), qkv.Data...)
		const split = 4
		span := make([][2]int, n)
		for t := range span {
			span[t] = [2]int{0, split}
			if t >= split {
				span[t] = [2]int{split, n}
			}
		}
		out := tensai.NewMatrix(n, e.hs)
		e.attend(&b, qkv, out, span)

		d, half := e.headSz, e.headSz/2
		rot := func(v []float32, pos int) []float64 {
			r := make([]float64, d)
			for j := 0; j < half; j++ {
				s, c := math.Sincos(float64(pos) * b.freq[j])
				x0, x1 := float64(v[j]), float64(v[j+half])
				r[j], r[j+half] = x0*c-x1*s, x1*c+x0*s
			}
			return r
		}
		row := func(t int) []float32 { return orig[t*3*e.hs : (t+1)*3*e.hs] }
		for tq := 0; tq < n; tq++ {
			for h := 0; h < e.heads; h++ {
				q := rot(row(tq)[h*d:(h+1)*d], tq-span[tq][0])
				var ws []float64
				var keys []int
				for tk := span[tq][0]; tk < span[tq][1]; tk++ {
					if b.local && (tk < tq-e.window || tk > tq+e.window) {
						continue
					}
					k := rot(row(tk)[e.hs+h*d:e.hs+(h+1)*d], tk-span[tk][0])
					var s float64
					for j := range q {
						s += q[j] * k[j]
					}
					ws = append(ws, math.Exp(s/math.Sqrt(float64(d))))
					keys = append(keys, tk)
				}
				var sum float64
				for _, w := range ws {
					sum += w
				}
				for j := 0; j < d; j++ {
					var want float64
					for i, tk := range keys {
						want += ws[i] / sum * float64(row(tk)[2*e.hs+h*d+j])
					}
					got := out.Data[tq*e.hs+h*d+j]
					if math.Abs(float64(got)-want) > 1e-4 {
						t.Fatalf("local=%v token %d head %d dim %d: %v, want %v", b.local, tq, h, j, got, want)
					}
				}
			}
		}
	}
}

// Vectors come out unit length and framed by the model's own tokens,
// and ids the model cannot read are refused rather than embedded.
func TestEmbedIDs(t *testing.T) {
	e := tinyEmbedder(t)
	if got, want := e.Tokenize("ab a"), []int{0, 2, 3, 4, 2, 1}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("Tokenize = %v, want %v", got, want)
	}
	v, n, err := e.Embed("ab a")
	if err != nil {
		t.Fatal(err)
	}
	if n != 6 || len(v) != e.Dim() {
		t.Fatalf("%d tokens, %d dims", n, len(v))
	}
	var sq float64
	for _, x := range v {
		sq += float64(x) * float64(x)
	}
	if math.Abs(sq-1) > 1e-5 {
		t.Errorf("squared length %v, want 1", sq)
	}
	if _, err := e.EmbedIDs([]int{0, 9, 1}); err == nil {
		t.Error("a token outside the vocabulary was embedded")
	}
	if _, err := e.EmbedIDs(make([]int, e.maxPos+1)); err == nil {
		t.Error("a text longer than the context was embedded")
	}
}

// Texts packed into one pass come out as they do alone, including when
// the cap splits them across passes.
func TestEmbedBatchMatchesSingle(t *testing.T) {
	e := tinyEmbedder(t)
	idss := [][]int{{0, 2, 3, 1}, {0, 4, 2, 4, 3, 2, 3, 3, 2, 4, 1}, {0, 1}}
	var alone [][]float32
	for _, ids := range idss {
		v, err := e.EmbedIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		alone = append(alone, v)
	}
	defer func(n int) { batchTokens = n }(batchTokens)
	// One pass for all three, then one each.
	for _, limit := range []int{batchTokens, 12} {
		batchTokens = limit
		got, err := e.EmbedBatch(idss)
		if err != nil {
			t.Fatal(err)
		}
		for i := range idss {
			for j := range alone[i] {
				if math.Abs(float64(got[i][j]-alone[i][j])) > 1e-6 {
					t.Fatalf("cap %d, sequence %d dim %d: batched %v, alone %v", limit, i, j, got[i][j], alone[i][j])
				}
			}
		}
	}
}

func TestEmbeddingsHandler(t *testing.T) {
	e := tinyEmbedder(t)
	s := &server{embed: &EmbedServer{Embedder: e, Name: "tiny"}}
	post := func(body string) (int, map[string]any) {
		rec := httptest.NewRecorder()
		s.embeddings(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body)))
		var out map[string]any
		json.Unmarshal(rec.Body.Bytes(), &out)
		return rec.Code, out
	}
	want, _, _ := e.Embed("ab")
	vector := func(item any, format string) []float32 {
		raw := item.(map[string]any)["embedding"]
		if format == "base64" {
			b, err := base64.StdEncoding.DecodeString(raw.(string))
			if err != nil {
				t.Fatal(err)
			}
			v := make([]float32, len(b)/4)
			for i := range v {
				v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
			}
			return v
		}
		var v []float32
		for _, x := range raw.([]any) {
			v = append(v, float32(x.(float64)))
		}
		return v
	}
	for _, tc := range []struct {
		body   string
		format string
		count  int
	}{
		{`{"input": "ab"}`, "", 1},
		{`{"input": ["ab", "ab a"]}`, "", 2},
		{`{"input": [0, 2, 3, 1]}`, "", 1},
		{`{"input": [[0, 2, 3, 1], [0, 1]]}`, "", 2},
		{`{"input": "ab", "encoding_format": "base64"}`, "base64", 1},
	} {
		code, out := post(tc.body)
		if code != http.StatusOK {
			t.Fatalf("%s: status %d: %v", tc.body, code, out)
		}
		data := out["data"].([]any)
		if len(data) != tc.count || out["model"] != "tiny" {
			t.Fatalf("%s: %d vectors from %v", tc.body, len(data), out["model"])
		}
		got := vector(data[0], tc.format)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: dim %d = %v, want %v", tc.body, i, got[i], want[i])
			}
		}
	}
	for _, body := range []string{`{"input": {}}`, `{"input": []}`, `{"input": [[0, 99]]}`, `{"input": "a", "encoding_format": "int8"}`} {
		if code, _ := post(body); code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400", body, code)
		}
	}
}

// Every embedding model in the cache gives unit vectors that its int8
// load keeps close to the float32 one. Skips when there is none.
func TestEmbeddingModels(t *testing.T) {
	ggufs, _ := findModels(t, CacheRoot())
	found := false
	for _, path := range ggufs {
		if !IsEmbeddingModel(path) {
			continue
		}
		found = true
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vs [2][]float32
			for i, bits := range []int{0, 8} {
				e, err := LoadEmbedder(path, bits, io.Discard)
				if err != nil {
					t.Fatal(err)
				}
				v, _, err := e.Embed("検索クエリ: 瑠璃色はどんな色？")
				e.Close()
				if err != nil {
					t.Fatal(err)
				}
				vs[i] = v
			}
			var dot, sq float64
			for i := range vs[0] {
				dot += float64(vs[0][i]) * float64(vs[1][i])
				sq += float64(vs[0][i]) * float64(vs[0][i])
			}
			if math.Abs(sq-1) > 1e-4 {
				t.Errorf("squared length %v, want 1", sq)
			}
			if dot < 0.98 {
				t.Errorf("int8 against float32: cosine %v", dot)
			}
		})
	}
	if !found {
		t.Skip("no embedding model in the cache")
	}
}

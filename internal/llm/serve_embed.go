package llm

// POST /v1/embeddings in OpenAI's shape: input is a string, an array of
// strings, or token ids (one array, or an array of them), and each comes
// back as a unit-length vector, as JSON floats or, when
// encoding_format is "base64", as base64 of little-endian float32s --
// the form OpenAI's own clients ask for by default. Embedding keeps no
// state, so requests run side by side and never wait on a chat.

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"strings"
)

// EmbedServer serves embeddings from an Embedder, alone or next to a
// chat model.
type EmbedServer struct {
	Embedder *Embedder
	Name     string // the model id requests and responses carry
}

// NewEmbedServer names e after the file it was read from.
func NewEmbedServer(e *Embedder, path string) *EmbedServer {
	return &EmbedServer{Embedder: e, Name: strings.TrimSuffix(filepath.Base(path), ".gguf")}
}

// ServeEmbeddings blocks on a server that answers /v1/embeddings and
// /v1/models only, for an embedding model with no chat model beside it.
func ServeEmbeddings(addr, apiKey string, es *EmbedServer) error {
	s := &server{apiKey: apiKey, embed: es}
	return s.listen(addr)
}

// embedInputs reads the request's input field into texts or token
// lists, exactly one of which comes back non-nil.
func embedInputs(raw json.RawMessage) ([]string, [][]int, error) {
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil && len(many) > 0 {
		return many, nil, nil
	}
	var ids []int
	if json.Unmarshal(raw, &ids) == nil && len(ids) > 0 {
		return nil, [][]int{ids}, nil
	}
	var idss [][]int
	if json.Unmarshal(raw, &idss) == nil && len(idss) > 0 {
		return nil, idss, nil
	}
	return nil, nil, fmt.Errorf("input must be a string, an array of strings, or token ids")
}

func (s *server) embeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpError(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	var req struct {
		Input  json.RawMessage `json:"input"`
		Model  string          `json:"model"`
		Format string          `json:"encoding_format"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Format != "" && req.Format != "float" && req.Format != "base64" {
		httpError(w, http.StatusBadRequest, fmt.Sprintf("encoding_format %q is not float or base64", req.Format))
		return
	}
	texts, idss, err := embedInputs(req.Input)
	if err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}
	e := s.embed.Embedder
	if texts != nil {
		idss = make([][]int, len(texts))
		for i, t := range texts {
			idss[i] = e.Tokenize(t)
		}
	}
	vs, err := e.EmbedBatch(idss)
	if err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}
	data := make([]map[string]any, len(idss))
	total := 0
	for i, v := range vs {
		total += len(idss[i])
		var out any = v
		if req.Format == "base64" {
			b := make([]byte, 4*len(v))
			for j, f := range v {
				binary.LittleEndian.PutUint32(b[4*j:], math.Float32bits(f))
			}
			out = base64.StdEncoding.EncodeToString(b)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": out}
	}
	writeJSON(w, map[string]any{
		"object": "list",
		"data":   data,
		"model":  s.embed.Name,
		"usage":  map[string]any{"prompt_tokens": total, "total_tokens": total},
	})
}

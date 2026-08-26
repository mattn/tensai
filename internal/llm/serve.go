package llm

// -serve turns the example into a minimal OpenAI-compatible endpoint:
// POST /v1/chat/completions accepts the familiar messages array (with
// optional streaming), renders it through the ChatML template, prefills
// the prompt through the batched path, and decodes with the same
// sampling as the CLI. One request holds the model at a time — the KV
// cache is rebuilt per request — so any OpenAI client pointed at the
// address works against a pure-Go model.

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature *float64      `json:"temperature"`
	TopP        *float64      `json:"top_p"`
	MaxTokens   int           `json:"max_tokens"`
	Seed        *int64        `json:"seed"`
}

// reset drops the KV cache so the next request starts a fresh context.
func (m *qwen) reset() {
	for i := range m.blocks {
		m.blocks[i].kc = nil
		m.blocks[i].vc = nil
	}
}

// render walks the messages through the model's template, ending with an
// open assistant turn. Models without a system role (Gemma) get the
// system text folded into the first user message.
func render(tm tmpl, msgs []chatMessage, defaultSystem string) string {
	var b strings.Builder
	b.WriteString(tm.bos)
	system := defaultSystem
	rest := msgs
	if len(msgs) > 0 && msgs[0].Role == "system" {
		system = msgs[0].Content
		rest = msgs[1:]
	}
	if !tm.foldSystem {
		b.WriteString(tm.sysOpen + system + tm.sysClose)
	}
	first := true
	for _, m := range rest {
		switch m.Role {
		case "assistant":
			b.WriteString(tm.asstOpen + m.Content + tm.asstClose)
		default:
			text := m.Content
			if tm.foldSystem && first {
				text = system + "\n\n" + text
			}
			first = false
			b.WriteString(tm.userOpen + text + tm.userClose)
		}
	}
	b.WriteString(tm.asstOpen)
	return b.String()
}

type server struct {
	mu      sync.Mutex // one request drives the model at a time
	model   *qwen
	draft   *qwen
	specK   int
	tok     tokenizerIface
	system  string
	nCtx    int
	temp    float64
	topP    float64
	imEnd   int
	eot     int
	tm      tmpl
	prefill func([]int, int) []float32
	step    func(int, int) []float32
	reset   func()
}

// tokenizerIface is the slice of the tokenizer the server needs.
type tokenizerIface interface {
	Encode(string) []int
	Decode([]int) string
}

func (s *server) listen(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.chatCompletions)
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{
			"object": "list",
			"data": []map[string]any{{
				"id": "tensai", "object": "model", "owned_by": "tensai",
			}},
		})
	})
	fmt.Printf("listening on %s (POST /v1/chat/completions)\n", addr)
	return http.ListenAndServe(addr, mux)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

func (s *server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpError(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		httpError(w, http.StatusBadRequest, "messages is required")
		return
	}
	temp := s.temp
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	topP := s.topP
	if req.TopP != nil {
		topP = *req.TopP
	}
	limit := req.MaxTokens
	if limit <= 0 {
		limit = 512
	}
	seed := time.Now().UnixNano()
	if req.Seed != nil {
		seed = *req.Seed
	}
	rng := rand.New(rand.NewSource(seed))

	ids := s.tok.Encode(render(s.tm, req.Messages, s.system))
	if len(ids) >= s.nCtx-1 {
		httpError(w, http.StatusBadRequest, fmt.Sprintf("prompt of %d tokens exceeds the %d-token context", len(ids), s.nCtx))
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.reset()
	if s.draft != nil {
		s.draft.reset()
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	var flush func(string)
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, _ := w.(http.Flusher)
		flush = func(piece string) {
			chunk := map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": created,
				"model": "tensai",
				"choices": []map[string]any{{
					"index": 0, "delta": map[string]any{"content": piece},
				}},
			}
			raw, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
	// Byte-fallback BPE tokens can split a multi-byte character, so a
	// chunk holds back an incomplete trailing rune until the next token
	// completes it; deliberately invalid bytes still flush once older
	// than one rune, so garbage cannot stall the stream.
	var pend []byte
	push := func(piece string, final bool) {
		pend = append(pend, piece...)
		cut := len(pend)
		if !final {
			for k := 1; k <= 3 && k <= len(pend); k++ {
				b := pend[len(pend)-k]
				if b&0xC0 == 0x80 {
					continue
				}
				var l int
				switch {
				case b < 0x80:
					l = 1
				case b >= 0xF0:
					l = 4
				case b >= 0xE0:
					l = 3
				case b >= 0xC0:
					l = 2
				default:
					l = 1
				}
				if l > k {
					cut = len(pend) - k
				}
				break
			}
		}
		if cut > 0 {
			flush(string(pend[:cut]))
			pend = append(pend[:0], pend[cut:]...)
		}
	}

	logits := s.prefill(ids, 0)
	if s.draft != nil {
		s.draft.prefill(ids, 0)
	}
	steps := len(ids)
	var out []int
	finish := "length"
	ctx := r.Context()
	if s.draft != nil {
		emit := func(next int) bool {
			select {
			case <-ctx.Done():
				return false
			default:
			}
			out = append(out, next)
			if flush != nil {
				push(s.tok.Decode([]int{next}), false)
			}
			return true
		}
		_, _, finish, _ = generateSpeculative(s.model, s.draft, logits, steps, limit,
			s.nCtx, s.specK, temp, topP, func(id int) bool {
				return id == s.imEnd || id == s.eot
			}, rng, emit)
	} else {
		for len(out) < limit && steps < s.nCtx-1 {
			// A disconnected client stops the generation instead of holding
			// the model for tokens nobody will read.
			select {
			case <-ctx.Done():
				finish = "abort"
				steps = s.nCtx // fall out below
				continue
			default:
			}
			next := sample(logits, temp, topP, rng)
			if next == s.imEnd || next == s.eot {
				finish = "stop"
				break
			}
			out = append(out, next)
			if flush != nil {
				push(s.tok.Decode([]int{next}), false)
			}
			logits = s.step(next, steps)
			steps++
		}
	}
	if finish == "abort" {
		return
	}

	if req.Stream {
		if len(pend) > 0 {
			push("", true)
		}
		final := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created,
			"model": "tensai",
			"choices": []map[string]any{{
				"index": 0, "delta": map[string]any{}, "finish_reason": finish,
			}},
		}
		raw, _ := json.Marshal(final)
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", raw)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		return
	}
	writeJSON(w, map[string]any{
		"id": id, "object": "chat.completion", "created": created,
		"model": "tensai",
		"choices": []map[string]any{{
			"index": 0,
			"message": map[string]any{
				"role": "assistant", "content": s.tok.Decode(out),
			},
			"finish_reason": finish,
		}},
		"usage": map[string]any{
			"prompt_tokens":     len(ids),
			"completion_tokens": len(out),
			"total_tokens":      len(ids) + len(out),
		},
	})
}

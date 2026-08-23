package main

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

// chatML renders the messages through the same template the CLI uses,
// ending with an open assistant turn.
func chatML(msgs []chatMessage, defaultSystem string) string {
	var b strings.Builder
	if len(msgs) == 0 || msgs[0].Role != "system" {
		b.WriteString("<|im_start|>system\n" + defaultSystem + "<|im_end|>\n")
	}
	for _, m := range msgs {
		role := m.Role
		if role != "system" && role != "user" && role != "assistant" {
			role = "user"
		}
		b.WriteString("<|im_start|>" + role + "\n" + m.Content + "<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

type server struct {
	mu     sync.Mutex // one request drives the model at a time
	model  *qwen
	tok    tokenizerIface
	system string
	nCtx   int
	temp   float64
	topP   float64
	imEnd  int
	eot    int
}

// tokenizerIface is the slice of the tokenizer the server needs.
type tokenizerIface interface {
	Encode(string) []int
	Decode([]int) string
}

func serve(addr string, model *qwen, tok tokenizerIface, system string, nCtx int, temp, topP float64, imEnd, eot int) error {
	s := &server{model: model, tok: tok, system: system, nCtx: nCtx,
		temp: temp, topP: topP, imEnd: imEnd, eot: eot}
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

	ids := s.tok.Encode(chatML(req.Messages, s.system))
	if len(ids) >= s.nCtx-1 {
		httpError(w, http.StatusBadRequest, fmt.Sprintf("prompt of %d tokens exceeds the %d-token context", len(ids), s.nCtx))
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.model.reset()

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

	logits := s.model.prefill(ids, 0)
	steps := len(ids)
	var out []int
	finish := "length"
	for len(out) < limit && steps < s.nCtx-1 {
		next := sample(logits, temp, topP, rng)
		if next == s.imEnd || next == s.eot {
			finish = "stop"
			break
		}
		out = append(out, next)
		if flush != nil {
			flush(s.tok.Decode([]int{next}))
		}
		logits = s.model.step(next, steps)
		steps++
	}

	if req.Stream {
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

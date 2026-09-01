package llm

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// Verbose narrates; it never changes what the caller gets. The server is
// the case that matters, since a request's long silence is what the flag
// was asked for.
func TestVerboseNarratesWithoutChangingTheAnswer(t *testing.T) {
	body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}`

	// The id and the timestamp differ between any two requests, so the
	// comparison is of the answer itself.
	answer := func(s *server) string {
		var out struct {
			Choices []struct {
				Message      chatMessage `json:"message"`
				FinishReason string      `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(post(t, s, body).Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		if len(out.Choices) != 1 {
			t.Fatalf("got %d choices", len(out.Choices))
		}
		return out.Choices[0].Message.Content + "|" + out.Choices[0].FinishReason
	}
	want := answer(scriptedServer(t, chatml(), []string{"one", " two"}))

	var log bytes.Buffer
	loud := scriptedServer(t, chatml(), []string{"one", " two"})
	loud.vlog = &log
	if got := answer(loud); got != want {
		t.Errorf("verbose changed the response: %q, want %q", got, want)
	}
	for _, line := range []string{
		"request: 1 messages, 0 tools",
		"rendered prompt:",
		"prefilling ",
		"prefilled in ",
	} {
		if !strings.Contains(log.String(), line) {
			t.Errorf("verbose log has no %q:\n%s", line, log.String())
		}
	}
	// The rendered prompt is what the model was actually handed, markers
	// and all -- the thing a caller cannot otherwise see.
	if !strings.Contains(log.String(), `<|im_start|>user\nhi<|im_end|>`) {
		t.Errorf("the prompt was not logged as rendered:\n%s", log.String())
	}
}

func TestClip(t *testing.T) {
	if got := clip("short", 100); got != `"short"` {
		t.Errorf("clip(short) = %s", got)
	}
	got := clip(strings.Repeat("x", 50), 10)
	if !strings.HasPrefix(got, `"xxxxxxxxxx"...`) || !strings.Contains(got, "50 bytes total") {
		t.Errorf("clip(long) = %s", got)
	}
	if or("", "none") != "none" || or("hermes", "none") != "hermes" {
		t.Error("or picked the wrong string")
	}
}

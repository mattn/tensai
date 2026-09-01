package llm

import "testing"

// An empty system prompt asks for no system turn, which is not the same
// as a turn with nothing in it: llama.cpp renders Gemma 4's template
// exactly this way when the caller sends no system message.
func TestEmptySystemWritesNoTurn(t *testing.T) {
	e := &Engine{tm: templateFor("gemma4", false), system: "You are a helpful assistant."}
	if got, want := e.systemTurn(), "<|turn>system\nYou are a helpful assistant.<turn|>\n"; got != want {
		t.Errorf("systemTurn() = %q, want %q", got, want)
	}
	e.system = ""
	if got := e.systemTurn(); got != "" {
		t.Errorf("empty system wrote %q", got)
	}
	// A family that folds the system prompt into the first user turn has
	// no system turn to write either way.
	folding := &Engine{tm: templateFor("gemma3", false), system: "sys"}
	if got := folding.systemTurn(); got != "" {
		t.Errorf("folding family wrote %q", got)
	}

	// The server renders the same way.
	msgs := []chatMessage{{Role: "user", Content: "hi"}}
	if got, want := render(templateFor("gemma4", false), msgs, "", nil),
		"<bos><|turn>user\nhi<turn|>\n<|turn>model\n"; got != want {
		t.Errorf("render() = %q, want %q", got, want)
	}
	if got := render(templateFor("gemma4", false), msgs, "sys", nil); got !=
		"<bos><|turn>system\nsys<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n" {
		t.Errorf("render() with a system prompt = %q", got)
	}
	// Folding families put the system text in the first user turn, and
	// nothing extra when there is none.
	if got, want := render(templateFor("gemma3", false), msgs, "", nil),
		"<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"; got != want {
		t.Errorf("folding render() = %q, want %q", got, want)
	}
}

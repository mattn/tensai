package llm

import (
	"strings"
	"testing"
)

func TestFinishToolTurnDoesNotExposeCall(t *testing.T) {
	e := &Engine{tm: gemma4Tmpl(), tools: []toolDef{wikipediaTool()}}
	call := `<|tool_call>call:wikipedia{query:<|"|>Mattn<|"|>}<tool_call|>`
	content, visible, calls := e.finishToolTurn(call+"\n", false)
	if content != "" || visible != "" {
		t.Fatalf("content, visible = %q, %q; want both empty", content, visible)
	}
	if len(calls) != 1 || calls[0].Function.Name != "wikipedia" {
		t.Fatalf("calls = %#v; want one wikipedia call", calls)
	}
}

func TestFinishToolTurnShowsAnswerButNotCall(t *testing.T) {
	e := &Engine{tm: gemma4Tmpl(), tools: []toolDef{wikipediaTool()}}
	turn := "The answer.\n<|tool_call>call:wikipedia{query:<|\"|>Mattn<|\"|>}<tool_call|>\n"
	content, visible, calls := e.finishToolTurn(turn, false)
	if content != "The answer." || visible != content || len(calls) != 1 {
		t.Fatalf("content, visible, calls = %q, %q, %d", content, visible, len(calls))
	}
	if strings.Contains(visible, "tool_call") {
		t.Fatalf("tool call leaked into visible output: %q", visible)
	}
}

func TestFinishToolTurnKeepsRequestedReasoning(t *testing.T) {
	tm := gemma4Tmpl()
	e := &Engine{tm: tm, opts: Options{Think: true}}
	content, visible, calls := e.finishToolTurn("thinking"+tm.reasonClose+"answer\n", true)
	if content != "answer" || len(calls) != 0 {
		t.Fatalf("content, calls = %q, %d", content, len(calls))
	}
	want := tm.reasonOpen + "thinking" + tm.reasonClose + "answer"
	if visible != want {
		t.Fatalf("visible = %q, want %q", visible, want)
	}
}

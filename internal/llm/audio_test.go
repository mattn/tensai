package llm

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	tensai "github.com/mattn/tensai"
)

func TestChatMessageContentParts(t *testing.T) {
	clip := []byte("RIFF fake")
	body := `{"role": "user", "content": [
		{"type": "input_audio", "input_audio": {"data": "` + base64.StdEncoding.EncodeToString(clip) + `", "format": "wav"}},
		{"type": "text", "text": "What is this?"}]}`
	var m chatMessage
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatal(err)
	}
	if m.Role != "user" || m.Content != audioToken+"What is this?" || len(m.Audio) != 1 || string(m.Audio[0]) != string(clip) {
		t.Fatalf("got %+v", m)
	}

	// A plain string, and an assistant turn with no content, read as before.
	for body, want := range map[string]string{
		`{"role": "user", "content": "hi"}`:                                                           "hi",
		`{"role": "assistant", "content": null, "tool_calls": []}`:                                    "",
		`{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}`: "ab",
	} {
		var m chatMessage
		if err := json.Unmarshal([]byte(body), &m); err != nil || m.Content != want || m.Audio != nil {
			t.Errorf("%s: %+v, %v", body, m, err)
		}
	}
	for _, bad := range []string{
		`{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "mp3"}}]}`,
		`{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "not base64!"}}]}`,
		`{"role": "user", "content": [{"type": "image_url"}]}`,
	} {
		var m chatMessage
		if err := json.Unmarshal([]byte(bad), &m); err == nil {
			t.Errorf("%s was accepted", bad)
		}
	}
}

func TestNumberAudio(t *testing.T) {
	got := numberAudio("<|im_start|>user\n" + audioToken + "hi<|im_end|>\n<|im_start|>user\n" + audioToken + audioToken + "and?")
	want := "<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nhi<|im_end|>\n<|im_start|>user\n" +
		"Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\nAudio 3: <|audio_bos|><|AUDIO|><|audio_eos|>\nand?"
	if got != want {
		t.Errorf("got\n%q\nwant\n%q", got, want)
	}
	if strings.Contains(numberAudio("plain"), "Audio") {
		t.Error("a prompt with no audio changed")
	}
}

// Clips the store has encoded map to the same ids every time they come
// back, and different clips to different ids.
func TestAudioStoreReusesSpans(t *testing.T) {
	const vocab, placeholder = 100, 7
	m := &qwen{cfg: config{Vocab: vocab, HiddenSize: 2}, soft: tensai.NewMatrix(5, 2)}
	a := &audioStore{at: map[[sha256.Size]byte][2]int{
		sha256.Sum256([]byte("one")): {0, 2},
		sha256.Sum256([]byte("two")): {2, 3},
	}}
	got, reset, err := a.expand(m, []int{1, placeholder, 2, placeholder}, [][]byte{[]byte("two"), []byte("one")}, placeholder)
	if err != nil || reset {
		t.Fatal(err, reset)
	}
	if want := []int{1, 102, 103, 104, 2, 100, 101}; !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	if _, _, err := a.expand(m, []int{placeholder}, nil, placeholder); err == nil {
		t.Error("a marker with no clip was accepted")
	}
}

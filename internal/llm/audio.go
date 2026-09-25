package llm

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"runtime/debug"
	"strings"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/qwenaudio"
)

// A chat message can carry audio the way OpenAI's API sends it: content
// as a list of parts, some of them {"type": "input_audio", "input_audio":
// {"data": <base64>, "format": "wav"}}. Each clip leaves audioToken where
// it stood in the text and its bytes in the message; a model that takes
// audio numbers the markers and encodes the clips, and any other answers
// that it cannot hear.

// audioToken is Qwen2-Audio's placeholder, which the encoded clip's rows
// replace one position each.
const audioToken = "<|AUDIO|>"

// UnmarshalJSON reads content as a plain string or as a list of parts.
func (m *chatMessage) UnmarshalJSON(b []byte) error {
	type plain chatMessage
	var raw struct {
		*plain
		Content json.RawMessage `json:"content"`
	}
	raw.plain = (*plain)(m)
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Content, m.Audio = "", nil
	c := bytes.TrimSpace(raw.Content)
	if len(c) == 0 || string(c) == "null" {
		return nil
	}
	if c[0] == '"' {
		return json.Unmarshal(c, &m.Content)
	}
	var parts []struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		InputAudio *struct {
			Data   string `json:"data"`
			Format string `json:"format"`
		} `json:"input_audio"`
	}
	if err := json.Unmarshal(c, &parts); err != nil {
		return fmt.Errorf("content is neither a string nor a list of parts: %w", err)
	}
	var sb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text":
			sb.WriteString(p.Text)
		case "input_audio":
			if p.InputAudio == nil {
				return fmt.Errorf("an input_audio part has no input_audio")
			}
			if f := p.InputAudio.Format; f != "" && f != "wav" {
				return fmt.Errorf("audio in %s format; send wav", f)
			}
			data, err := base64.StdEncoding.DecodeString(p.InputAudio.Data)
			if err != nil {
				return fmt.Errorf("input_audio data is not base64: %w", err)
			}
			m.Audio = append(m.Audio, data)
			sb.WriteString(audioToken)
		default:
			return fmt.Errorf("unsupported content part %q", p.Type)
		}
	}
	m.Content = sb.String()
	return nil
}

// numberAudio lays each clip out the way Qwen2-Audio's chat template
// does, numbered through the whole conversation.
func numberAudio(prompt string) string {
	var b strings.Builder
	for n := 1; ; n++ {
		i := strings.Index(prompt, audioToken)
		if i < 0 {
			b.WriteString(prompt)
			return b.String()
		}
		fmt.Fprintf(&b, "%sAudio %d: <|audio_bos|>%s<|audio_eos|>\n", prompt[:i], n, audioToken)
		prompt = prompt[i+len(audioToken):]
	}
}

// audioStore is the server's side of a model that hears. Encoded clips
// accumulate in the model's soft rows, each at a range of ids past the
// vocabulary chosen by the clip's hash: a conversation that resends the
// same clip with every turn gets the same ids, so the prompt cache keeps
// working, while a different clip can never pass for it.
type audioStore struct {
	weights string
	at      map[[sha256.Size]byte][2]int // first row and row count
}

// maxSoftRows bounds what the store keeps, about sixteen thirty-second
// clips; past it the store and everything cached with it start over.
const maxSoftRows = 16 * 750

// expand replaces the audio placeholders in ids with the rows of the
// clips, in order, encoding the ones the store has not seen. It reports
// whether the store had to start over, which invalidates any cached
// prefix.
func (a *audioStore) expand(m *qwen, ids []int, clips [][]byte, placeholder int) (out []int, reset bool, err error) {
	var need int
	for _, id := range ids {
		if id == placeholder {
			need++
		}
	}
	if need != len(clips) {
		return nil, false, fmt.Errorf("the conversation holds %d audio clips and %d %s markers", len(clips), need, audioToken)
	}
	// The encoder loads only when a clip is new, and goes as soon as
	// the request's clips are encoded: resident, it would cost 2.5GB.
	var enc *qwenaudio.Encoder
	defer func() {
		if enc != nil {
			debug.FreeOSMemory()
		}
	}()
	spans := make([][2]int, len(clips))
	for i, clip := range clips {
		key := sha256.Sum256(clip)
		if span, ok := a.at[key]; ok {
			spans[i] = span
			continue
		}
		samples, err := qwenaudio.DecodeWAV(clip)
		if err != nil {
			return nil, false, fmt.Errorf("audio clip %d: %w", i+1, err)
		}
		if enc == nil {
			if enc, err = qwenaudio.LoadEncoder(a.weights, 0); err != nil {
				return nil, false, err
			}
		}
		rows := enc.Encode(qwenaudio.LogMel(samples))
		if m.soft != nil && m.soft.Rows+rows.Rows > maxSoftRows {
			m.soft, a.at = nil, nil
			// Spans handed out earlier in this request pointed into
			// what was just dropped, so the request starts again.
			out, _, err := a.expand(m, ids, clips, placeholder)
			return out, true, err
		}
		start := 0
		if m.soft != nil {
			start = m.soft.Rows
		}
		m.soft = appendRows(m.soft, rows)
		if a.at == nil {
			a.at = map[[sha256.Size]byte][2]int{}
		}
		a.at[key] = [2]int{start, rows.Rows}
		spans[i] = a.at[key]
	}
	k := 0
	for _, id := range ids {
		if id != placeholder {
			out = append(out, id)
			continue
		}
		for r := range spans[k][1] {
			out = append(out, m.cfg.Vocab+spans[k][0]+r)
		}
		k++
	}
	return out, reset, nil
}

func appendRows(m, rows *tensai.Matrix) *tensai.Matrix {
	if m == nil {
		return &tensai.Matrix{Rows: rows.Rows, Cols: rows.Cols, Data: append([]tensai.Float(nil), rows.Data...)}
	}
	return &tensai.Matrix{Rows: m.Rows + rows.Rows, Cols: m.Cols, Data: append(m.Data, rows.Data...)}
}

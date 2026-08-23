package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSmolLM2Fuzz(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "ref_smollm2_fuzz.json"))
	if err != nil {
		t.Skip("no fuzz ref")
	}
	var ref struct {
		Corpus []string `json:"corpus"`
		IDs    [][]int  `json:"ids"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	tokRaw, err := os.ReadFile(filepath.Join("testdata", "smollm2.json"))
	if err != nil {
		t.Skip("no smollm2.json")
	}
	tok, err := Parse(tokRaw)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range ref.Corpus {
		got := tok.Encode(s)
		want := ref.IDs[i]
		if len(got) != len(want) {
			t.Fatalf("%q: got %v want %v", s, got, want)
		}
		for j := range got {
			if got[j] != want[j] {
				t.Fatalf("%q: got %v want %v", s, got, want)
			}
		}
	}
}

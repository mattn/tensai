package main

import (
	"bytes"
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Shape flags are package globals, so these tests must not run in parallel.
func smallShape(t *testing.T) {
	t.Helper()
	old := [7]int{dModel, nHeads, nBlocks, seqLen, batchSize, headDim, dFF}
	t.Cleanup(func() {
		dModel, nHeads, nBlocks, seqLen, batchSize, headDim, dFF = old[0], old[1], old[2], old[3], old[4], old[5], old[6]
	})
	dModel, nHeads, nBlocks, seqLen, batchSize = 8, 2, 1, 4, 1
	headDim, dFF = dModel/nHeads, 4*dModel
}

func TestBatchAtWindows(t *testing.T) {
	smallShape(t)
	rng := rand.New(rand.NewPCG(7, 0))
	m := &model{}
	for _, text := range [][]int{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}} {
		seen := map[int]bool{}
		for i := 0; i < 100; i++ {
			tokens, labels := m.batchAt(text, rng)
			p := tokens[0]
			seen[p] = true
			if !reflect.DeepEqual(tokens, text[p:p+seqLen]) || !reflect.DeepEqual(labels, text[p+1:p+seqLen+1]) {
				t.Fatalf("bad window: tokens=%v labels=%v", tokens, labels)
			}
		}
		if len(seen) != len(text)-seqLen {
			t.Fatalf("sampled starts %v, want all %d windows", seen, len(text)-seqLen)
		}
	}
}

func TestInvalidCheckpoint(t *testing.T) {
	for _, tc := range []struct{ name, data, want string }{
		{"missing shape", `{}`, "must be positive"},
		{"zero heads", `{"model":8,"heads":0,"seq":4}`, "must be positive"},
		{"negative width", `{"model":-8,"heads":2,"seq":4}`, "must be positive"},
		{"zero context", `{"model":8,"heads":2,"seq":0}`, "must be positive"},
		{"negative blocks", `{"model":8,"heads":2,"seq":4,"blocks":-1}`, "non-negative"},
		{"incompatible heads", `{"model":8,"heads":3,"seq":4}`, "not divisible"},
		{"empty vocabulary", `{"model":8,"heads":2,"seq":4}`, "vocabulary must not be empty"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			smallShape(t)
			path := filepath.Join(t.TempDir(), "checkpoint.json")
			if err := os.WriteFile(path, []byte(tc.data), 0600); err != nil {
				t.Fatal(err)
			}
			err := run(1, .003, .8, 0, 7, false, "", "", "", path, "")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want error containing %q", err, tc.want)
			}
		})
	}
}

func TestTrainSaveLoad(t *testing.T) {
	smallShape(t)
	dir := t.TempDir()
	data := filepath.Join(dir, "text.txt")
	saved := filepath.Join(dir, "model.json")
	resumed := filepath.Join(dir, "resumed.json")
	// Exactly one valid training window; this used to panic in IntN(0).
	if err := os.WriteFile(data, []byte("abcde"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := run(2, .003, .8, 0, 7, false, data, "", saved, "", ""); err != nil {
		t.Fatal(err)
	}
	before, err := readCheckpoint(saved)
	if err != nil {
		t.Fatal(err)
	}
	// The checkpoint must override shape flags and work without the corpus.
	if err := os.Remove(data); err != nil {
		t.Fatal(err)
	}
	dModel, nHeads, seqLen = 0, 0, 0
	if err := run(1, .003, .8, 4, 7, false, "", "", "", saved, "ab"); err != nil {
		t.Fatal(err)
	}
	if dModel != 8 || nHeads != 2 || seqLen != 4 {
		t.Fatal("checkpoint did not restore its shape")
	}
	if err := os.WriteFile(data, []byte("edcba"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := run(1, .003, .8, 0, 7, false, data, "", resumed, saved, ""); err != nil {
		t.Fatal(err)
	}
	after, err := readCheckpoint(resumed)
	if err != nil {
		t.Fatal(err)
	}
	var a, b bytes.Buffer
	if err := json.Compact(&a, before.Params); err != nil {
		t.Fatal(err)
	}
	if err := json.Compact(&b, after.Params); err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("continued training did not change parameters")
	}
}

func TestShortCorpus(t *testing.T) {
	for _, text := range []string{"", "abcd"} {
		t.Run(text, func(t *testing.T) {
			smallShape(t)
			path := filepath.Join(t.TempDir(), "text.txt")
			if err := os.WriteFile(path, []byte(text), 0600); err != nil {
				t.Fatal(err)
			}
			err := run(1, .003, .8, 0, 7, false, path, "", "", "", "")
			if err == nil || !strings.Contains(err.Error(), "not longer than") {
				t.Fatalf("got %v, want short corpus error", err)
			}
		})
	}
}

package qwenimage

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// readI32 reads a little-endian int32 dump.
func readI32(t *testing.T, path string, want int) []int {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference dump: %v", err)
	}
	if len(b) != 4*want {
		t.Fatalf("%s: %d bytes, want %d values", path, len(b), want)
	}
	out := make([]int, want)
	for i := range out {
		out[i] = int(int32(binary.LittleEndian.Uint32(b[4*i:])))
	}
	return out
}

// TestLayoutMatchesReference checks the rotary angles and the
// block-causal key limits against what diffusers builds for the same
// text-to-image sequence. Needs no weights, only testdata/layout.py.
func TestLayoutMatchesReference(t *testing.T) {
	const (
		textLen = 13
		height  = 6
		width   = 5
	)
	l := NewLayout(textLen, height, width)
	n := l.Tokens()
	suffix := fmt.Sprintf("%d_%d_%d", textLen, height, width)
	name := func(kind, ext string) string {
		return filepath.Join("testdata", "rope_"+kind+"_"+suffix+"."+ext)
	}

	wantLimit := readI32(t, name("limit", "i32"), n)
	for i, w := range wantLimit {
		if l.KeyLimit[i] != w {
			t.Fatalf("query %d reads keys below %d, reference says %d", i, l.KeyLimit[i], w)
		}
	}
	// The reference's target mask marks the tokens that modulate from
	// the sampled timestep, which is row 0 here.
	wantTarget := readI32(t, name("target", "i32"), n)
	for i, w := range wantTarget {
		if row := 1 - w; l.Row[i] != row {
			t.Fatalf("token %d reads modulation row %d, reference says %d", i, l.Row[i], row)
		}
	}

	rope := l.Rope()
	wantCos := readF32(t, name("cos", "f32"), n*ropePairs)
	wantSin := readF32(t, name("sin", "f32"), n*ropePairs)
	var worst float64
	for i := range wantCos {
		worst = math.Max(worst, math.Abs(float64(rope.Cos.Data[i]-wantCos[i])))
		worst = math.Max(worst, math.Abs(float64(rope.Sin.Data[i]-wantSin[i])))
	}
	t.Logf("largest rotary difference %.3g", worst)
	if worst > 1e-6 {
		t.Errorf("rotary angles differ by %g", worst)
	}
}

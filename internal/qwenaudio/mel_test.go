package qwenaudio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// readF32 loads a reference dump, skipping the test when testdata/ref.py
// has not been run: the dumps only mean anything beside the checkpoint.
func readF32(t *testing.T, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Skipf("%s: %v (run testdata/ref.py)", name, err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func maxDiff(a, b []float32) float64 {
	var d float64
	for i := range a {
		d = max(d, math.Abs(float64(a[i])-float64(b[i])))
	}
	return d
}

func TestLogMelMatchesWhisper(t *testing.T) {
	want := readF32(t, "mel.f32")
	ref := readF32(t, "samples.f32")
	samples, err := ReadWAV(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		t.Skip(err)
	}
	if len(samples) != len(ref) || maxDiff(samples, ref) != 0 {
		t.Fatalf("samples differ from soundfile's: %d against %d", len(samples), len(ref))
	}
	mel, frames := LogMel(samples)
	if frames != (len(samples)+hop-1)/hop {
		t.Errorf("frames %d", frames)
	}
	if len(mel.Data) != len(want) {
		t.Fatalf("mel has %d values, want %d", len(mel.Data), len(want))
	}
	if d := maxDiff(mel.Data, want); d > 1e-4 {
		t.Errorf("log-mel differs by %g", d)
	}
}

// A wholly silent clip is the floor everywhere, not NaN.
func TestLogMelSilence(t *testing.T) {
	mel, frames := LogMel(make([]float32, 1600))
	if frames != 10 {
		t.Errorf("frames %d, want 10", frames)
	}
	for _, v := range mel.Data {
		if v != mel.Data[0] || math.IsNaN(float64(v)) {
			t.Fatalf("silence gave %v and %v", mel.Data[0], v)
		}
	}
}

func writeWAV(t *testing.T, format, bits, channels, rate int, frames [][]float64) string {
	t.Helper()
	var data []byte
	for _, fr := range frames {
		for _, v := range fr {
			switch {
			case format == 1 && bits == 16:
				data = binary.LittleEndian.AppendUint16(data, uint16(int16(math.Round(v*(1<<15)))))
			case format == 1 && bits == 24:
				x := int32(math.Round(v * (1 << 23)))
				data = append(data, byte(x), byte(x>>8), byte(x>>16))
			case format == 3 && bits == 32:
				data = binary.LittleEndian.AppendUint32(data, math.Float32bits(float32(v)))
			}
		}
	}
	var b []byte
	b = append(b, "RIFF"...)
	b = binary.LittleEndian.AppendUint32(b, uint32(4+8+16+8+8+len(data)+2))
	b = append(b, "WAVE"...)
	// An unknown chunk before fmt, as some writers put LIST there.
	b = append(b, "LIST"...)
	b = binary.LittleEndian.AppendUint32(b, 1)
	b = append(b, 0, 0)
	b = append(b, "fmt "...)
	b = binary.LittleEndian.AppendUint32(b, 16)
	b = binary.LittleEndian.AppendUint16(b, uint16(format))
	b = binary.LittleEndian.AppendUint16(b, uint16(channels))
	b = binary.LittleEndian.AppendUint32(b, uint32(rate))
	b = binary.LittleEndian.AppendUint32(b, uint32(rate*channels*bits/8))
	b = binary.LittleEndian.AppendUint16(b, uint16(channels*bits/8))
	b = binary.LittleEndian.AppendUint16(b, uint16(bits))
	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(data)))
	b = append(b, data...)
	p := filepath.Join(t.TempDir(), "x.wav")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestReadWAVFormats(t *testing.T) {
	frames := [][]float64{{0.5, -0.5}, {0.25, 0.25}, {-1, 0}}
	want := []float32{0, 0.25, -0.5}
	for _, c := range []struct{ format, bits int }{{1, 16}, {1, 24}, {3, 32}} {
		got, err := ReadWAV(writeWAV(t, c.format, c.bits, 2, SampleRate, frames))
		if err != nil {
			t.Fatalf("format %d/%d: %v", c.format, c.bits, err)
		}
		if len(got) != len(want) || maxDiff(got, want) > 1e-4 {
			t.Errorf("format %d/%d: %v, want %v", c.format, c.bits, got, want)
		}
	}
}

// A tone survives resampling at its frequency and level.
func TestResampleKeepsATone(t *testing.T) {
	const from, hz = 44100, 440.0
	x := make([]float32, from)
	for i := range x {
		x[i] = float32(0.5 * math.Sin(2*math.Pi*hz*float64(i)/from))
	}
	y := Resample(x, from, SampleRate)
	if len(y) != SampleRate {
		t.Fatalf("%d samples, want %d", len(y), SampleRate)
	}
	var worst float64
	for i := 1000; i < len(y)-1000; i++ {
		w := 0.5 * math.Sin(2*math.Pi*hz*float64(i)/SampleRate)
		worst = max(worst, math.Abs(float64(y[i])-w))
	}
	if worst > 5e-3 {
		t.Errorf("resampled tone is off by %g", worst)
	}
}

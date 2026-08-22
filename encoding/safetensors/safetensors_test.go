package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	tensai "github.com/mattn/tensai"
)

func TestRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tensors := map[string]*tensai.Tensor{}
	for _, c := range []struct {
		name  string
		shape []int
	}{
		{"model.embed", []int{7, 5}},
		{"model.layers.0.weight", []int{5, 5}},
		{"bias", []int{5}},
	} {
		x := tensai.NewTensor(c.shape...)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		tensors[c.name] = x
	}
	path := filepath.Join(t.TempDir(), "model.safetensors")
	meta := map[string]string{"format": "pt", "producer": "tensai"}
	if err := SaveFile(path, tensors, meta); err != nil {
		t.Fatalf("save: %v", err)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	if got := f.Names(); len(got) != 3 || got[0] != "bias" || got[1] != "model.embed" {
		t.Fatalf("names: %v", got)
	}
	if f.Metadata()["producer"] != "tensai" {
		t.Fatalf("metadata: %v", f.Metadata())
	}
	dtype, shape, ok := f.Info("model.embed")
	if !ok || dtype != "F32" || len(shape) != 2 || shape[0] != 7 {
		t.Fatalf("info: %v %v %v", dtype, shape, ok)
	}
	for name, want := range tensors {
		got, err := f.Tensor(name)
		if err != nil {
			t.Fatalf("tensor %q: %v", name, err)
		}
		if len(got.Data) != len(want.Data) {
			t.Fatalf("tensor %q: length %d != %d", name, len(got.Data), len(want.Data))
		}
		for i := range want.Data {
			if got.Data[i] != want.Data[i] {
				t.Fatalf("tensor %q element %d: %v != %v", name, i, got.Data[i], want.Data[i])
			}
		}
	}
	if _, err := f.Tensor("missing"); err == nil {
		t.Fatal("expected error for missing tensor")
	}
}

// build assembles a raw safetensors file from a header object and buffer.
func build(t *testing.T, header map[string]any, data []byte) []byte {
	t.Helper()
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(h)))
	out = append(out, h...)
	return append(out, data...)
}

func TestDtypeConversions(t *testing.T) {
	// F16: 1.0, -2.5, smallest subnormal (~5.96e-8), +inf.
	f16 := []uint16{0x3c00, 0xc100, 0x0001, 0x7c00}
	// BF16: 1.0, -3.140625 (0xc049).
	bf16 := []uint16{0x3f80, 0xc049}
	// F64: pi.
	f64 := []uint64{math.Float64bits(math.Pi)}

	var buf []byte
	for _, v := range f16 {
		buf = binary.LittleEndian.AppendUint16(buf, v)
	}
	for _, v := range bf16 {
		buf = binary.LittleEndian.AppendUint16(buf, v)
	}
	for _, v := range f64 {
		buf = binary.LittleEndian.AppendUint64(buf, v)
	}
	raw := build(t, map[string]any{
		"h": map[string]any{"dtype": "F16", "shape": []int{4}, "data_offsets": []int{0, 8}},
		"b": map[string]any{"dtype": "BF16", "shape": []int{2}, "data_offsets": []int{8, 12}},
		"d": map[string]any{"dtype": "F64", "shape": []int{}, "data_offsets": []int{12, 20}},
	}, buf)

	f, err := NewFile(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	h, err := f.Tensor("h")
	if err != nil {
		t.Fatal(err)
	}
	if h.Data[0] != 1 || h.Data[1] != -2.5 {
		t.Fatalf("f16: %v", h.Data)
	}
	if diff := math.Abs(float64(h.Data[2]) - 5.960464477539063e-8); diff > 1e-15 {
		t.Fatalf("f16 subnormal: %v", h.Data[2])
	}
	if !math.IsInf(float64(h.Data[3]), 1) {
		t.Fatalf("f16 inf: %v", h.Data[3])
	}
	b, err := f.Tensor("b")
	if err != nil {
		t.Fatal(err)
	}
	if b.Data[0] != 1 || b.Data[1] != -3.140625 {
		t.Fatalf("bf16: %v", b.Data)
	}
	d, err := f.Tensor("d")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Data) != 1 || math.Abs(float64(d.Data[0])-math.Pi) > 1e-7 {
		t.Fatalf("f64 scalar: %v", d.Data)
	}
}

func TestHeaderErrors(t *testing.T) {
	if _, err := NewFile(bytes.NewReader([]byte{1, 2, 3})); err == nil {
		t.Fatal("expected error for truncated file")
	}
	huge := make([]byte, 8)
	binary.LittleEndian.PutUint64(huge, 1<<40)
	if _, err := NewFile(bytes.NewReader(huge)); err == nil {
		t.Fatal("expected error for implausible header length")
	}
	if _, err := NewFile(bytes.NewReader(build(t, map[string]any{
		"x": map[string]any{"dtype": "F32", "shape": []int{3}, "data_offsets": []int{0, 8}},
	}, make([]byte, 8)))); err == nil {
		t.Fatal("expected error for shape/bytes mismatch")
	}

	// Unsupported dtypes parse but refuse to load.
	f, err := NewFile(bytes.NewReader(build(t, map[string]any{
		"q": map[string]any{"dtype": "I8", "shape": []int{4}, "data_offsets": []int{0, 4}},
	}, make([]byte, 4))))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if _, err := f.Tensor("q"); err == nil {
		t.Fatal("expected unsupported dtype error")
	}

	// Truncated data surfaces as a read error, not garbage.
	f, err = NewFile(bytes.NewReader(build(t, map[string]any{
		"x": map[string]any{"dtype": "F32", "shape": []int{4}, "data_offsets": []int{0, 16}},
	}, make([]byte, 8))))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if _, err := f.Tensor("x"); err == nil {
		t.Fatal("expected error for truncated tensor data")
	}
}

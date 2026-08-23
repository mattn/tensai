package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// writer builds a synthetic GGUF file for the tests.
type writer struct{ buf bytes.Buffer }

func (w *writer) u32(v uint32)  { binary.Write(&w.buf, binary.LittleEndian, v) }
func (w *writer) u64(v uint64)  { binary.Write(&w.buf, binary.LittleEndian, v) }
func (w *writer) f32(v float32) { w.u32(math.Float32bits(v)) }
func (w *writer) str(s string)  { w.u64(uint64(len(s))); w.buf.WriteString(s) }

// f32to16 is enough of a half encoder for the values the tests use.
func f32to16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xff) - 127 + 15
	mant := bits >> 13 & 0x3ff
	if f == 0 {
		return sign
	}
	return sign | uint16(exp)<<10 | uint16(mant)
}

func buildTestFile(t *testing.T) []byte {
	t.Helper()
	var w writer
	w.u32(0x46554747) // GGUF
	w.u32(3)
	w.u64(6) // tensors
	w.u64(6) // metadata keys

	w.str("general.architecture")
	w.u32(8)
	w.str("llama")
	w.str("general.alignment")
	w.u32(4) // u32
	w.u32(64)
	w.str("llama.block_count")
	w.u32(4)
	w.u32(12)
	w.str("llama.rope.freq_base")
	w.u32(6)
	w.f32(10000)
	w.str("tokenizer.ggml.tokens")
	w.u32(9) // array of strings
	w.u32(8)
	w.u64(3)
	w.str("<s>")
	w.str("a")
	w.str("b")
	w.str("flag")
	w.u32(7) // bool
	w.buf.WriteByte(1)

	// Tensor directory: offsets are relative to the aligned data section
	// and themselves aligned.
	type td struct {
		name       string
		dims       []uint64 // ne order: fastest first
		typ        uint32
		byteLength int64
	}
	tds := []td{
		{"f32t", []uint64{4, 2}, typeF32, 8 * 4},
		{"f16t", []uint64{4}, typeF16, 4 * 2},
		{"bf16t", []uint64{2}, typeBF16, 2 * 2},
		{"q8t", []uint64{32}, typeQ8_0, 34},
		{"q4t", []uint64{32}, typeQ4_0, 18},
		{"q41t", []uint64{32}, typeQ4_1, 20},
	}
	off := int64(0)
	offsets := make([]int64, len(tds))
	for i, d := range tds {
		offsets[i] = off
		w.str(d.name)
		w.u32(uint32(len(d.dims)))
		for _, n := range d.dims {
			w.u64(n)
		}
		w.u32(d.typ)
		w.u64(uint64(off))
		off = (off + d.byteLength + 63) &^ 63
	}

	// Pad to the 64-byte alignment the metadata declares.
	for w.buf.Len()%64 != 0 {
		w.buf.WriteByte(0)
	}
	data := w.buf.Len()

	// f32t: 0..7 as [2][4] row-major (ne (4,2) reversed).
	for i := 0; i < 8; i++ {
		w.f32(float32(i))
	}
	pad := func() {
		for (w.buf.Len()-data)%64 != 0 {
			w.buf.WriteByte(0)
		}
	}
	pad()
	for _, v := range []float32{1, -2, 0.5, 0} {
		binary.Write(&w.buf, binary.LittleEndian, f32to16(v))
	}
	pad()
	binary.Write(&w.buf, binary.LittleEndian, uint16(0x3FC0)) // bf16 1.5
	binary.Write(&w.buf, binary.LittleEndian, uint16(0xBF80)) // bf16 -1
	pad()
	// q8t: scale 0.5, values -16..15.
	binary.Write(&w.buf, binary.LittleEndian, f32to16(0.5))
	for i := 0; i < 32; i++ {
		w.buf.WriteByte(byte(int8(i - 16)))
	}
	pad()
	// q4t: scale 2, nibbles i%16 (low half of block) and 15-i%16 (high).
	binary.Write(&w.buf, binary.LittleEndian, f32to16(2))
	for i := 0; i < 16; i++ {
		w.buf.WriteByte(byte(i) | byte(15-i)<<4)
	}
	pad()
	// q41t: scale 0.25, min -1, nibble i for both halves.
	binary.Write(&w.buf, binary.LittleEndian, f32to16(0.25))
	binary.Write(&w.buf, binary.LittleEndian, f32to16(-1))
	for i := 0; i < 16; i++ {
		w.buf.WriteByte(byte(i) | byte(i)<<4)
	}
	return w.buf.Bytes()
}

func TestReadSynthetic(t *testing.T) {
	f, err := NewFile(bytes.NewReader(buildTestFile(t)))
	if err != nil {
		t.Fatal(err)
	}

	if arch, _ := f.String("general.architecture"); arch != "llama" {
		t.Fatalf("architecture: %q", arch)
	}
	if n, _ := f.Int("llama.block_count"); n != 12 {
		t.Fatalf("block_count: %d", n)
	}
	if v, _ := f.Float("llama.rope.freq_base"); v != 10000 {
		t.Fatalf("freq_base: %v", v)
	}
	if b, ok := f.KV("flag"); !ok || b != true {
		t.Fatalf("flag: %v %v", b, ok)
	}
	toks, _ := f.KV("tokenizer.ggml.tokens")
	if arr, ok := toks.([]any); !ok || len(arr) != 3 || arr[0] != "<s>" || arr[2] != "b" {
		t.Fatalf("tokens: %v", toks)
	}

	typ, shape, ok := f.Info("f32t")
	if !ok || typ != "F32" || len(shape) != 2 || shape[0] != 2 || shape[1] != 4 {
		t.Fatalf("f32t info: %v %v %v", typ, shape, ok)
	}

	ft, err := f.Tensor("f32t")
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range ft.Data {
		if v != float32(i) {
			t.Fatalf("f32t[%d] = %v", i, v)
		}
	}

	half, err := f.Tensor("f16t")
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, -2, 0.5, 0} {
		if half.Data[i] != want {
			t.Fatalf("f16t[%d] = %v want %v", i, half.Data[i], want)
		}
	}

	bf, err := f.Tensor("bf16t")
	if err != nil {
		t.Fatal(err)
	}
	if bf.Data[0] != 1.5 || bf.Data[1] != -1 {
		t.Fatalf("bf16t = %v", bf.Data)
	}

	q8, err := f.Tensor("q8t")
	if err != nil {
		t.Fatal(err)
	}
	for i := range q8.Data {
		if want := 0.5 * float32(i-16); q8.Data[i] != want {
			t.Fatalf("q8t[%d] = %v want %v", i, q8.Data[i], want)
		}
	}

	q4, err := f.Tensor("q4t")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if want := 2 * float32(i-8); q4.Data[i] != want {
			t.Fatalf("q4t[%d] = %v want %v", i, q4.Data[i], want)
		}
		if want := 2 * float32(15-i-8); q4.Data[16+i] != want {
			t.Fatalf("q4t[%d] = %v want %v", 16+i, q4.Data[16+i], want)
		}
	}

	q41, err := f.Tensor("q41t")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		want := 0.25*float32(i) - 1
		if q41.Data[i] != want || q41.Data[16+i] != want {
			t.Fatalf("q41t[%d] = %v/%v want %v", i, q41.Data[i], q41.Data[16+i], want)
		}
	}

	if _, err := f.Tensor("missing"); err == nil {
		t.Fatal("expected error for a missing tensor")
	}
}

func TestBadInput(t *testing.T) {
	if _, err := NewFile(bytes.NewReader([]byte("nope"))); err == nil {
		t.Fatal("expected error for a bad magic")
	}
	var w writer
	w.u32(0x46554747)
	w.u32(1) // unsupported version
	w.u64(0)
	w.u64(0)
	if _, err := NewFile(bytes.NewReader(w.buf.Bytes())); err == nil {
		t.Fatal("expected error for version 1")
	}

	// An unsupported tensor encoding surfaces at Tensor, not Open.
	w = writer{}
	w.u32(0x46554747)
	w.u32(3)
	w.u64(1)
	w.u64(0)
	w.str("kq")
	w.u32(1)
	w.u64(256)
	w.u32(11) // Q3_K: real but not decoded here
	w.u64(0)
	f, err := NewFile(bytes.NewReader(w.buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if typ, _, _ := f.Info("kq"); typ != "type11" {
		t.Fatalf("info: %q", typ)
	}
	if _, err := f.Tensor("kq"); err == nil {
		t.Fatal("expected error for Q3_K")
	}
}

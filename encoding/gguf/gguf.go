// Package gguf reads the GGUF model format (llama.cpp's container:
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — typed
// metadata key/values followed by an aligned blob of tensors — with no
// dependencies beyond the standard library.
//
// Reading is lazy: Open parses only the header, and each Tensor call reads
// just that tensor's bytes. F32 comes back as-is; F16 and BF16 convert to
// float32; the block-quantized types Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 and the
// K-quants Q2_K through Q6_K plus IQ4_NL and MXFP4 dequantize to float32 on the way
// out, which covers the encodings llama.cpp's published checkpoints
// usually ship (the Q2_K..Q5_K_M mixes, their Q6_K tensors, and the
// IQ4_NL blocks imatrix mixes lean on). Dimensions arrive in tensai's row-major order (GGUF stores
// them fastest-varying first; this package reverses them), so a
// token-embedding tensor reads as [vocab, hidden] just like the
// safetensors reader would produce.
//
// One conversion artifact to know about: llama.cpp's converter permutes
// the output rows of attention q and k projection weights into its
// interleaved rotary-embedding order (HF row (head, half, i) lands at row
// (head, i, half)). This package returns tensors exactly as stored;
// consumers pairing GGUF weights with half-split RoPE code must undo that
// permutation themselves.
package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/mmapfile"
)

// Tensor encodings from the GGML type enum; only the ones this package
// decodes are named.
const (
	typeF32   = 0
	typeF16   = 1
	typeQ4_0  = 2
	typeQ4_1  = 3
	typeQ5_0  = 6
	typeQ5_1  = 7
	typeQ8_0  = 8
	typeQ2_K  = 10
	typeQ3_K  = 11
	typeQ4_K  = 12
	typeQ5_K  = 13
	typeQ6_K  = 14
	typeIQ4NL = 20
	typeBF16  = 30
	typeMXFP4 = 39
)

var typeNames = map[uint32]string{
	typeF32: "F32", typeF16: "F16", typeQ4_0: "Q4_0",
	typeQ4_1: "Q4_1", typeQ8_0: "Q8_0", typeBF16: "BF16",
	typeQ4_K: "Q4_K", typeQ5_K: "Q5_K", typeQ6_K: "Q6_K",
	typeQ5_0: "Q5_0", typeQ5_1: "Q5_1",
	typeQ2_K: "Q2_K", typeQ3_K: "Q3_K", typeIQ4NL: "IQ4_NL",
	typeMXFP4: "MXFP4",
}

// blockSpec describes one quantization block: how many values it decodes
// to and how many bytes it occupies.
var blockSpec = map[uint32]struct{ values, bytes int64 }{
	typeF32:   {1, 4},
	typeF16:   {1, 2},
	typeBF16:  {1, 2},
	typeQ8_0:  {32, 2 + 32},         // f16 scale + 32 int8
	typeQ4_0:  {32, 2 + 16},         // f16 scale + 32 nibbles
	typeQ4_1:  {32, 2 + 2 + 16},     // f16 scale + f16 min + 32 nibbles
	typeQ5_0:  {32, 2 + 4 + 16},     // f16 scale + high-bit plane + nibbles
	typeQ5_1:  {32, 2 + 2 + 4 + 16}, // + f16 min
	typeIQ4NL: {32, 2 + 16},         // f16 scale + nibbles through a nonlinear table
	typeMXFP4: {32, 1 + 16},         // e8m0 exponent + fp4 codes through a nonlinear table
	// K-quants: 256-value super-blocks with quantized sub-scales.
	typeQ2_K: {256, 16 + 64 + 2 + 2},       // 4-bit scale/min pairs, 2-bit quants
	typeQ3_K: {256, 32 + 64 + 12 + 2},      // high-bit plane, 2-bit quants, 6-bit scales
	typeQ4_K: {256, 2 + 2 + 12 + 128},      // d, dmin, packed scales, nibbles
	typeQ5_K: {256, 2 + 2 + 12 + 32 + 128}, // + high bits
	typeQ6_K: {256, 128 + 64 + 16 + 2},     // ql, qh, int8 scales, d
}

type tensorInfo struct {
	name   string
	shape  []int // row-major (reversed from the file's order)
	typ    uint32
	offset int64 // relative to the data section
}

// File is an open GGUF checkpoint. It keeps the underlying reader open and
// loads tensor data on demand.
type File struct {
	r        io.ReaderAt
	closer   io.Closer
	data     []byte // the whole file when memory-mapped; nil otherwise
	unmap    func() error
	kv       map[string]any
	tensors  map[string]tensorInfo
	names    []string
	dataBase int64
}

// Open opens a GGUF file and parses its metadata and tensor directory.
// The file is memory-mapped where the platform allows, so tensor blocks
// dequantize straight out of the page cache; otherwise reads go through
// ReadAt.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if data, unmap, err := mmapfile.Map(f); err == nil {
		g, err := NewFile(bytes.NewReader(data))
		if err != nil {
			unmap()
			f.Close()
			return nil, err
		}
		g.data = data
		g.unmap = unmap
		g.closer = f
		return g, nil
	}
	g, err := NewFile(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	g.closer = f
	return g, nil
}

// NewFile parses a GGUF checkpoint from a reader.
func NewFile(r io.ReaderAt) (*File, error) {
	d := &decoder{r: io.NewSectionReader(r, 0, 1<<62)}
	if magic := d.u32(); magic != 0x46554747 { // "GGUF"
		return nil, fmt.Errorf("gguf: bad magic 0x%08x", magic)
	}
	version := d.u32()
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}
	nTensors := d.u64()
	nKV := d.u64()
	if d.err != nil {
		return nil, d.err
	}
	if nTensors > 1<<20 || nKV > 1<<20 {
		return nil, fmt.Errorf("gguf: implausible header counts (%d tensors, %d keys)", nTensors, nKV)
	}

	g := &File{r: r, kv: make(map[string]any, nKV), tensors: make(map[string]tensorInfo, nTensors)}
	for i := uint64(0); i < nKV; i++ {
		key := d.str()
		val := d.value(d.u32())
		if d.err != nil {
			return nil, fmt.Errorf("gguf: metadata %q: %w", key, d.err)
		}
		g.kv[key] = val
	}
	for i := uint64(0); i < nTensors; i++ {
		var t tensorInfo
		t.name = d.str()
		nDims := d.u32()
		if d.err == nil && nDims > 8 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dimensions", t.name, nDims)
		}
		dims := make([]int, nDims)
		for j := range dims {
			dims[j] = int(d.u64())
		}
		// GGUF stores ne[0] fastest-varying; tensai shapes are row-major.
		t.shape = make([]int, nDims)
		for j := range dims {
			t.shape[nDims-1-uint32(j)] = dims[j]
		}
		t.typ = d.u32()
		t.offset = int64(d.u64())
		if d.err != nil {
			return nil, fmt.Errorf("gguf: tensor %q: %w", t.name, d.err)
		}
		g.tensors[t.name] = t
		g.names = append(g.names, t.name)
	}
	sort.Strings(g.names)

	align := int64(32)
	if v, ok := g.kv["general.alignment"]; ok {
		if a, ok := toInt64(v); ok && a > 0 {
			align = a
		}
	}
	g.dataBase = (d.off + align - 1) / align * align
	return g, nil
}

// Close unmaps and closes the underlying file when the File was created
// by Open.
func (f *File) Close() error {
	var err error
	if f.unmap != nil {
		err = f.unmap()
		f.unmap = nil
		f.data = nil
	}
	if f.closer != nil {
		if cerr := f.closer.Close(); err == nil {
			err = cerr
		}
	}
	return err
}

// Release hints that a tensor's bytes will not be read again: when the
// file is memory-mapped, the kernel may drop those page-cache pages
// right away instead of letting them crowd out the caller's own
// memory during a large load. Harmless otherwise.
func (f *File) Release(name string) {
	t, ok := f.tensors[name]
	if !ok || f.data == nil {
		return
	}
	spec, ok := blockSpec[t.typ]
	if !ok {
		return
	}
	n := int64(1)
	for _, d := range t.shape {
		n *= int64(d)
	}
	nbytes := n / spec.values * spec.bytes
	lo := f.dataBase + t.offset
	if lo < 0 || lo+nbytes > int64(len(f.data)) {
		return
	}
	mmapfile.Release(f.data[lo : lo+nbytes])
}

// Names returns the tensor names in sorted order.
func (f *File) Names() []string { return append([]string(nil), f.names...) }

// KV returns a metadata value: string, bool, float64, int64, uint64 (and
// float32/uint32/... as stored) or []any for arrays.
func (f *File) KV(key string) (any, bool) {
	v, ok := f.kv[key]
	return v, ok
}

// Keys returns the metadata keys in sorted order.
func (f *File) Keys() []string {
	keys := make([]string, 0, len(f.kv))
	for k := range f.kv {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// String returns a string metadata value.
func (f *File) String(key string) (string, bool) {
	s, ok := f.kv[key].(string)
	return s, ok
}

// Int returns an integer metadata value regardless of its stored width.
func (f *File) Int(key string) (int64, bool) {
	v, ok := f.kv[key]
	if !ok {
		return 0, false
	}
	return toInt64(v)
}

// Float returns a float metadata value (accepting integer storage too).
func (f *File) Float(key string) (float64, bool) {
	switch v := f.kv[key].(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	}
	n, ok := f.Int(key)
	return float64(n), ok
}

// Ints returns an integer array metadata value, whatever width the file
// stored its elements at, or nil when the key is absent or not an array
// of integers. Gemma 4 states its per-layer feed-forward widths this way.
func (f *File) Ints(key string) []int64 {
	arr, ok := f.kv[key].([]any)
	if !ok {
		return nil
	}
	out := make([]int64, len(arr))
	for i, v := range arr {
		n, ok := toInt64(v)
		if !ok {
			return nil
		}
		out[i] = n
	}
	return out
}

// Bools returns a boolean array metadata value, or nil when the key is
// absent or not an array of booleans -- Gemma 4's sliding-window pattern.
func (f *File) Bools(key string) []bool {
	arr, ok := f.kv[key].([]any)
	if !ok {
		return nil
	}
	out := make([]bool, len(arr))
	for i, v := range arr {
		b, ok := v.(bool)
		if !ok {
			return nil
		}
		out[i] = b
	}
	return out
}

func toInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case uint8:
		return int64(n), true
	case int8:
		return int64(n), true
	case uint16:
		return int64(n), true
	case int16:
		return int64(n), true
	case uint32:
		return int64(n), true
	case int32:
		return int64(n), true
	case uint64:
		return int64(n), true
	case int64:
		return n, true
	case bool:
		if n {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// Info reports a tensor's type name and row-major shape.
func (f *File) Info(name string) (typ string, shape []int, ok bool) {
	t, ok := f.tensors[name]
	if !ok {
		return "", nil, false
	}
	typ, named := typeNames[t.typ]
	if !named {
		typ = fmt.Sprintf("type%d", t.typ)
	}
	return typ, append([]int(nil), t.shape...), true
}

// RawTensor returns a tensor's undecoded bytes and type name — the
// quantized blocks exactly as stored, sliced from the mapping when the
// file is memory-mapped (valid only until Close) and read fresh
// otherwise. Consumers that keep quantized weights quantized repack from
// this instead of paying Tensor's float32 detour.
func (f *File) RawTensor(name string) (string, []byte, error) {
	t, ok := f.tensors[name]
	if !ok {
		return "", nil, fmt.Errorf("gguf: no tensor %q", name)
	}
	spec, ok := blockSpec[t.typ]
	if !ok {
		return "", nil, fmt.Errorf("gguf: tensor %q: unsupported encoding type %d", name, t.typ)
	}
	n := int64(1)
	for _, d := range t.shape {
		n *= int64(d)
	}
	nbytes := n / spec.values * spec.bytes
	typ := typeNames[t.typ]
	if f.data != nil {
		lo := f.dataBase + t.offset
		if lo < 0 || lo+nbytes > int64(len(f.data)) {
			return "", nil, fmt.Errorf("gguf: tensor %q extends past the file", name)
		}
		return typ, f.data[lo : lo+nbytes], nil
	}
	raw := make([]byte, nbytes)
	if _, err := f.r.ReadAt(raw, f.dataBase+t.offset); err != nil {
		return "", nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
	}
	return typ, raw, nil
}

// Float16 widens an IEEE 754 half — exported for consumers decoding raw
// quantized blocks, whose scales are stored as f16.
func Float16(h uint16) float32 { return f16to32(h) }

// Tensor reads one tensor, dequantizing to float32.
func (f *File) Tensor(name string) (*tensai.Tensor, error) {
	return f.TensorRows(name, 0, -1)
}

// TensorRows reads and dequantizes a range of a tensor's leading axis --
// rows [from, to), or to the end when to is negative. A quantized tensor
// stores its values in fixed-size blocks, so a row range can only start
// and end on a block boundary; every row length here is a multiple of the
// block size, which is what makes a single row of a per-layer embedding
// table readable without dequantizing the other quarter of a million.
func (f *File) TensorRows(name string, from, to int) (*tensai.Tensor, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %q", name)
	}
	spec, ok := blockSpec[t.typ]
	if !ok {
		typ := typeNames[t.typ]
		if typ == "" {
			typ = fmt.Sprintf("type %d", t.typ)
		}
		return nil, fmt.Errorf("gguf: tensor %q: unsupported encoding %s", name, typ)
	}
	// Only a 2-D or higher tensor has rows to slice; anything flatter is a
	// single row so that a whole-tensor read still works on it.
	rows, axis := 1, 0
	if len(t.shape) > 1 {
		rows, axis = t.shape[0], 1
	}
	if to < 0 || to > rows {
		to = rows
	}
	if from < 0 || from > to {
		return nil, fmt.Errorf("gguf: tensor %q: row range [%d,%d) is not within %d rows", name, from, to, rows)
	}
	rowLen := int64(1)
	for _, d := range t.shape[axis:] {
		rowLen *= int64(d)
	}
	if rowLen%spec.values != 0 {
		return nil, fmt.Errorf("gguf: tensor %q: a row of %d values is not a whole number of blocks", name, rowLen)
	}
	n := rowLen * int64(to-from)
	shape := append([]int(nil), t.shape...)
	if axis == 1 {
		shape[0] = to - from
	}
	var raw []byte
	nbytes := n / spec.values * spec.bytes
	skip := rowLen * int64(from) / spec.values * spec.bytes
	if f.data != nil {
		lo := f.dataBase + t.offset + skip
		if lo < 0 || lo+nbytes > int64(len(f.data)) {
			return nil, fmt.Errorf("gguf: tensor %q extends past the file", name)
		}
		raw = f.data[lo : lo+nbytes]
	} else {
		raw = make([]byte, nbytes)
		if _, err := f.r.ReadAt(raw, f.dataBase+t.offset+skip); err != nil {
			return nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
		}
	}
	out := tensai.NewTensor(shape...)
	dst := out.Data
	switch t.typ {
	case typeF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case typeF16:
		for i := range dst {
			dst[i] = f16to32(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case typeBF16:
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16)
		}
	case typeQ8_0:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*34:]
			s := f16to32(binary.LittleEndian.Uint16(blk))
			for i := 0; i < 32; i++ {
				dst[b*32+int64(i)] = s * float32(int8(blk[2+i]))
			}
		}
	case typeQ4_0:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*18:]
			s := f16to32(binary.LittleEndian.Uint16(blk))
			for i := 0; i < 16; i++ {
				q := blk[2+i]
				dst[b*32+int64(i)] = s * (float32(q&0x0F) - 8)
				dst[b*32+int64(i)+16] = s * (float32(q>>4) - 8)
			}
		}
	case typeQ4_1:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*20:]
			s := f16to32(binary.LittleEndian.Uint16(blk))
			m := f16to32(binary.LittleEndian.Uint16(blk[2:]))
			for i := 0; i < 16; i++ {
				q := blk[4+i]
				dst[b*32+int64(i)] = s*float32(q&0x0F) + m
				dst[b*32+int64(i)+16] = s*float32(q>>4) + m
			}
		}
	case typeQ5_0:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*22:]
			s := f16to32(binary.LittleEndian.Uint16(blk))
			qh := binary.LittleEndian.Uint32(blk[2:])
			for i := 0; i < 16; i++ {
				q := blk[6+i]
				lo := uint32(q&0x0F) | qh>>i<<4&0x10
				hi := uint32(q>>4) | qh>>(i+12)&0x10
				dst[b*32+int64(i)] = s * (float32(lo) - 16)
				dst[b*32+int64(i)+16] = s * (float32(hi) - 16)
			}
		}
	case typeQ5_1:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*24:]
			s := f16to32(binary.LittleEndian.Uint16(blk))
			m := f16to32(binary.LittleEndian.Uint16(blk[2:]))
			qh := binary.LittleEndian.Uint32(blk[4:])
			for i := 0; i < 16; i++ {
				q := blk[8+i]
				lo := uint32(q&0x0F) | qh>>i<<4&0x10
				hi := uint32(q>>4) | qh>>(i+12)&0x10
				dst[b*32+int64(i)] = s*float32(lo) + m
				dst[b*32+int64(i)+16] = s*float32(hi) + m
			}
		}
	case typeIQ4NL:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*18:]
			s := f16to32(binary.LittleEndian.Uint16(blk))
			for i := 0; i < 16; i++ {
				q := blk[2+i]
				dst[b*32+int64(i)] = s * iq4nl[q&0x0F]
				dst[b*32+int64(i)+16] = s * iq4nl[q>>4]
			}
		}
	case typeMXFP4:
		for b := int64(0); b < n/32; b++ {
			blk := raw[b*17:]
			// E8M0 exponent; the table carries twice the FP4 values, so
			// the factor halves.
			s := float32(math.Ldexp(1, int(blk[0])-128))
			for i := 0; i < 16; i++ {
				q := blk[1+i]
				dst[b*32+int64(i)] = s * mxfp4[q&0x0F]
				dst[b*32+int64(i)+16] = s * mxfp4[q>>4]
			}
		}
	case typeQ2_K:
		for b := int64(0); b < n/256; b++ {
			dequantQ2K(raw[b*84:b*84+84], dst[b*256:b*256+256])
		}
	case typeQ3_K:
		for b := int64(0); b < n/256; b++ {
			dequantQ3K(raw[b*110:b*110+110], dst[b*256:b*256+256])
		}
	case typeQ4_K:
		for b := int64(0); b < n/256; b++ {
			dequantQ4K(raw[b*144:b*144+144], dst[b*256:b*256+256])
		}
	case typeQ5_K:
		for b := int64(0); b < n/256; b++ {
			dequantQ5K(raw[b*176:b*176+176], dst[b*256:b*256+256])
		}
	case typeQ6_K:
		for b := int64(0); b < n/256; b++ {
			dequantQ6K(raw[b*210:b*210+210], dst[b*256:b*256+256])
		}
	}
	return out, nil
}

// iq4nl is IQ4_NL's nonlinear codebook: sixteen levels spaced to match
// the empirical weight distribution instead of uniformly.
var iq4nl = [16]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// mxfp4 is the FP4 (E2M1) value table doubled onto an integer grid; the
// per-block scale is halved to compensate.
var mxfp4 = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// scaleMinK4 unpacks the j-th 6-bit scale and min from a K-quant
// super-block's 12 packed bytes (8 pairs: the first four pairs use the low
// 6 bits of bytes 0-7, the last four splice nibbles of bytes 8-11 with the
// top bits of bytes 0-7).
func scaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return q[j+4]&0x0F | q[j-4]>>6<<4, q[j+4]>>4 | q[j]>>6<<4
}

// dequantQ2K expands one 84-byte Q2_K super-block: sixteen 16-value
// groups of 2-bit quants, each with a 4-bit scale and 4-bit min against
// the super-block's two f16 factors.
func dequantQ2K(blk []byte, dst []float32) {
	scales := blk[:16]
	qs := blk[16:80]
	d := f16to32(binary.LittleEndian.Uint16(blk[80:]))
	dmin := f16to32(binary.LittleEndian.Uint16(blk[82:]))
	is := 0
	y := 0
	for n := 0; n < 256; n += 128 {
		q := qs[n/4 : n/4+32]
		for shift := 0; shift < 8; shift += 2 {
			for _, half := range [2]int{0, 16} {
				sc := scales[is]
				is++
				dl := d * float32(sc&0x0F)
				ml := dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					dst[y] = dl*float32(q[half+l]>>shift&3) - ml
					y++
				}
			}
		}
	}
}

// dequantQ3K expands one 110-byte Q3_K super-block: 2-bit quants plus a
// high-bit plane (an unset mask bit means subtract 4), with sixteen 6-bit
// scales packed into twelve bytes.
func dequantQ3K(blk []byte, dst []float32) {
	hm := blk[:32]
	qs := blk[32:96]
	packed := blk[96:108]
	d := f16to32(binary.LittleEndian.Uint16(blk[108:]))

	// Unpack the twelve scale bytes into sixteen int8 values (stored
	// offset by 32): the low 4 bits of bytes 0-7 pair with 2-bit fields of
	// bytes 8-11, and the high 4 bits pair with the remaining fields.
	a0 := binary.LittleEndian.Uint32(packed[0:])
	a1 := binary.LittleEndian.Uint32(packed[4:])
	tmp := binary.LittleEndian.Uint32(packed[8:])
	const kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
	var aux [4]uint32
	aux[0] = a0&kmask2 | tmp>>0&kmask1<<4
	aux[1] = a1&kmask2 | tmp>>2&kmask1<<4
	aux[2] = a0>>4&kmask2 | tmp>>4&kmask1<<4
	aux[3] = a1>>4&kmask2 | tmp>>6&kmask1<<4
	var scales [16]int8
	for i, v := range aux {
		scales[4*i+0] = int8(v)
		scales[4*i+1] = int8(v >> 8)
		scales[4*i+2] = int8(v >> 16)
		scales[4*i+3] = int8(v >> 24)
	}

	is := 0
	y := 0
	m := byte(1)
	for n := 0; n < 256; n += 128 {
		q := qs[n/4 : n/4+32]
		for shift := 0; shift < 8; shift += 2 {
			for _, half := range [2]int{0, 16} {
				dl := d * float32(int32(scales[is])-32)
				is++
				for l := 0; l < 16; l++ {
					v := int32(q[half+l] >> shift & 3)
					if hm[half+l]&m == 0 {
						v -= 4
					}
					dst[y] = dl * float32(v)
					y++
				}
			}
			m <<= 1
		}
	}
}

// dequantQ4K expands one 144-byte Q4_K super-block into 256 floats:
// eight 32-value groups, each with a 6-bit scale and min against the
// super-block's two f16 factors.
func dequantQ4K(blk []byte, dst []float32) {
	d := f16to32(binary.LittleEndian.Uint16(blk))
	dmin := f16to32(binary.LittleEndian.Uint16(blk[2:]))
	scales := blk[4:16]
	qs := blk[16:]
	is := 0
	for j := 0; j < 256; j += 64 {
		sc, mn := scaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(mn)
		sc, mn = scaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(mn)
		q := qs[j/2 : j/2+32]
		for l := 0; l < 32; l++ {
			dst[j+l] = d1*float32(q[l]&0x0F) - m1
			dst[j+32+l] = d2*float32(q[l]>>4) - m2
		}
		is += 2
	}
}

// dequantQ5K is Q4_K plus one high bit per value from the qh plane.
func dequantQ5K(blk []byte, dst []float32) {
	d := f16to32(binary.LittleEndian.Uint16(blk))
	dmin := f16to32(binary.LittleEndian.Uint16(blk[2:]))
	scales := blk[4:16]
	qh := blk[16:48]
	qs := blk[48:]
	is := 0
	u1, u2 := uint8(1), uint8(2)
	for j := 0; j < 256; j += 64 {
		sc, mn := scaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(mn)
		sc, mn = scaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(mn)
		q := qs[j/2 : j/2+32]
		for l := 0; l < 32; l++ {
			hi1, hi2 := float32(0), float32(0)
			if qh[l]&u1 != 0 {
				hi1 = 16
			}
			if qh[l]&u2 != 0 {
				hi2 = 16
			}
			dst[j+l] = d1*(float32(q[l]&0x0F)+hi1) - m1
			dst[j+32+l] = d2*(float32(q[l]>>4)+hi2) - m2
		}
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
}

// dequantQ6K expands one 210-byte Q6_K super-block: 6-bit values split
// across a nibble plane and a 2-bit plane, sixteen int8 sub-scales, one
// f16 super-scale.
func dequantQ6K(blk []byte, dst []float32) {
	ql := blk[:128]
	qh := blk[128:192]
	sc := blk[192:208]
	d := f16to32(binary.LittleEndian.Uint16(blk[208:]))
	for n := 0; n < 256; n += 128 {
		y := dst[n:]
		qln := ql[n/2 : n/2+64]
		qhn := qh[n/4 : n/4+32]
		scn := sc[n/16 : n/16+8]
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8(qln[l]&0x0F|qhn[l]>>0&3<<4) - 32
			q2 := int8(qln[l+32]&0x0F|qhn[l]>>2&3<<4) - 32
			q3 := int8(qln[l]>>4|qhn[l]>>4&3<<4) - 32
			q4 := int8(qln[l+32]>>4|qhn[l]>>6&3<<4) - 32
			y[l] = d * float32(int8(scn[is])) * float32(q1)
			y[l+32] = d * float32(int8(scn[is+2])) * float32(q2)
			y[l+64] = d * float32(int8(scn[is+4])) * float32(q3)
			y[l+96] = d * float32(int8(scn[is+6])) * float32(q4)
		}
	}
}

// f16to32 converts an IEEE 754 half, subnormals and specials included.
func f16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: renormalize.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}

// decoder reads the little-endian header sequentially, latching the first
// error.
type decoder struct {
	r   *io.SectionReader
	off int64
	err error
}

func (d *decoder) read(b []byte) {
	if d.err != nil {
		return
	}
	if _, err := io.ReadFull(io.NewSectionReader(d.r, d.off, int64(len(b))), b); err != nil {
		d.err = err
		return
	}
	d.off += int64(len(b))
}

func (d *decoder) u32() uint32 {
	var b [4]byte
	d.read(b[:])
	return binary.LittleEndian.Uint32(b[:])
}

func (d *decoder) u64() uint64 {
	var b [8]byte
	d.read(b[:])
	return binary.LittleEndian.Uint64(b[:])
}

func (d *decoder) str() string {
	n := d.u64()
	if d.err != nil {
		return ""
	}
	if n > 1<<24 {
		d.err = fmt.Errorf("string of %d bytes", n)
		return ""
	}
	b := make([]byte, n)
	d.read(b)
	return string(b)
}

// value decodes one metadata value of the given type tag.
func (d *decoder) value(typ uint32) any {
	switch typ {
	case 0: // u8
		var b [1]byte
		d.read(b[:])
		return b[0]
	case 1: // i8
		var b [1]byte
		d.read(b[:])
		return int8(b[0])
	case 2: // u16
		var b [2]byte
		d.read(b[:])
		return binary.LittleEndian.Uint16(b[:])
	case 3: // i16
		var b [2]byte
		d.read(b[:])
		return int16(binary.LittleEndian.Uint16(b[:]))
	case 4:
		return d.u32()
	case 5:
		return int32(d.u32())
	case 6:
		return math.Float32frombits(d.u32())
	case 7:
		var b [1]byte
		d.read(b[:])
		return b[0] != 0
	case 8:
		return d.str()
	case 9: // array
		elem := d.u32()
		n := d.u64()
		if d.err != nil {
			return nil
		}
		if n > 1<<24 {
			d.err = fmt.Errorf("array of %d elements", n)
			return nil
		}
		arr := make([]any, n)
		for i := range arr {
			arr[i] = d.value(elem)
			if d.err != nil {
				return nil
			}
		}
		return arr
	case 10:
		return d.u64()
	case 11:
		return int64(d.u64())
	case 12:
		bits := d.u64()
		return math.Float64frombits(bits)
	}
	if d.err == nil {
		d.err = fmt.Errorf("unknown value type %d", typ)
	}
	return nil
}

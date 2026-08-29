// Package safetensors reads and writes the safetensors checkpoint format
// (https://github.com/huggingface/safetensors) — the plain "8-byte header
// length, JSON header, raw little-endian buffer" layout most published
// model weights ship in — with no dependencies beyond the standard
// library.
//
// Reading is lazy: Open parses only the header, and each Tensor call reads
// just that tensor's bytes, so single tensors can be pulled out of a
// multi-gigabyte checkpoint without loading the rest. F32 is returned
// as-is; F16, BF16, and F64 are converted to tensai's float32 on the way
// out. Save writes F32 tensors, which round-trips everything tensai
// produces.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/mmapfile"
)

// maxHeaderSize bounds the JSON header; the reference implementation
// refuses 100MB and larger.
const maxHeaderSize = 100 << 20

type entry struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// File is an open safetensors checkpoint. It keeps the underlying reader
// and loads tensor data on demand.
type File struct {
	r       io.ReaderAt
	closer  io.Closer
	data    []byte // the whole file when memory-mapped; nil otherwise
	unmap   func() error
	entries map[string]entry
	names   []string
	meta    map[string]string
	dataOff int64 // where the byte buffer starts
}

// Open opens a safetensors file, parsing its header. The file is
// memory-mapped where the platform allows, so tensor bytes slice straight
// out of the page cache; otherwise reads go through ReadAt. Close the
// returned File when done.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if data, unmap, err := mmapfile.Map(f); err == nil {
		sf, err := NewFile(bytes.NewReader(data))
		if err != nil {
			unmap()
			f.Close()
			return nil, err
		}
		sf.data = data
		sf.unmap = unmap
		sf.closer = f
		return sf, nil
	}
	sf, err := NewFile(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	sf.closer = f
	return sf, nil
}

// NewFile parses a safetensors header from r. Tensor data is read from r
// on demand.
func NewFile(r io.ReaderAt) (*File, error) {
	var lenBuf [8]byte
	if _, err := r.ReadAt(lenBuf[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: reading header length: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen == 0 || headerLen >= maxHeaderSize {
		return nil, fmt.Errorf("safetensors: implausible header length %d", headerLen)
	}
	header := make([]byte, headerLen)
	if _, err := r.ReadAt(header, 8); err != nil {
		return nil, fmt.Errorf("safetensors: reading header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parsing header: %w", err)
	}
	f := &File{
		r:       r,
		entries: make(map[string]entry, len(raw)),
		dataOff: int64(8 + headerLen),
	}
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &f.meta); err != nil {
				return nil, fmt.Errorf("safetensors: parsing __metadata__: %w", err)
			}
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("safetensors: parsing entry %q: %w", name, err)
		}
		if e.DataOffsets[0] < 0 || e.DataOffsets[1] < e.DataOffsets[0] {
			return nil, fmt.Errorf("safetensors: entry %q has invalid offsets %v", name, e.DataOffsets)
		}
		n := 1
		for _, d := range e.Shape {
			if d < 0 {
				return nil, fmt.Errorf("safetensors: entry %q has negative dimension in %v", name, e.Shape)
			}
			n *= d
		}
		size, err := dtypeSize(e.Dtype)
		if err == nil && int64(n)*size != e.DataOffsets[1]-e.DataOffsets[0] {
			return nil, fmt.Errorf("safetensors: entry %q: shape %v does not match %d bytes",
				name, e.Shape, e.DataOffsets[1]-e.DataOffsets[0])
		}
		f.entries[name] = e
		f.names = append(f.names, name)
	}
	sort.Strings(f.names)
	return f, nil
}

// Close closes the underlying file when the File came from Open; it is a
// no-op otherwise.
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

// Names lists the tensor names, sorted.
func (f *File) Names() []string { return append([]string(nil), f.names...) }

// Metadata returns the free-form __metadata__ strings, or nil.
func (f *File) Metadata() map[string]string { return f.meta }

// Info reports a tensor's dtype and shape without loading its data.
func (f *File) Info(name string) (dtype string, shape []int, ok bool) {
	e, ok := f.entries[name]
	if !ok {
		return "", nil, false
	}
	return e.Dtype, append([]int(nil), e.Shape...), true
}

func dtypeSize(dtype string) (int64, error) {
	switch dtype {
	case "F32":
		return 4, nil
	case "F16", "BF16":
		return 2, nil
	case "F64":
		return 8, nil
	case "U8", "I8":
		return 1, nil
	default:
		return 0, fmt.Errorf("safetensors: unsupported dtype %q", dtype)
	}
}

// Tensor loads one tensor, converting F16, BF16, and F64 to tensai's
// float32. A safetensors scalar (empty shape) comes back as shape [1].
func (f *File) Tensor(name string) (*tensai.Tensor, error) {
	e, ok := f.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	if _, err := dtypeSize(e.Dtype); err != nil {
		return nil, fmt.Errorf("%w (tensor %q)", err, name)
	}
	if e.Dtype == "U8" || e.Dtype == "I8" {
		// These carry packed payloads -- MXFP4 nibbles and their e8m0
		// exponents -- whose meaning lives in the format that wrote them,
		// and widening a quarter of a gigabyte of nibbles to float32 is
		// not what any caller wants. Raw hands back the bytes instead.
		return nil, fmt.Errorf("safetensors: tensor %q is %s; use Raw (tensor %q)", name, e.Dtype, name)
	}
	var buf []byte
	if f.data != nil {
		lo, hi := f.dataOff+e.DataOffsets[0], f.dataOff+e.DataOffsets[1]
		if hi > int64(len(f.data)) || lo < 0 || lo > hi {
			return nil, fmt.Errorf("safetensors: tensor %q extends past the file", name)
		}
		buf = f.data[lo:hi]
	} else {
		buf = make([]byte, e.DataOffsets[1]-e.DataOffsets[0])
		if _, err := f.r.ReadAt(buf, f.dataOff+e.DataOffsets[0]); err != nil {
			return nil, fmt.Errorf("safetensors: reading tensor %q: %w", name, err)
		}
	}
	shape := e.Shape
	if len(shape) == 0 {
		shape = []int{1}
	}
	out := tensai.NewTensor(shape...)
	switch e.Dtype {
	case "F32":
		for i := range out.Data {
			out.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case "F16":
		for i := range out.Data {
			out.Data[i] = f16to32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	case "BF16":
		for i := range out.Data {
			out.Data[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[i*2:])) << 16)
		}
	case "F64":
		for i := range out.Data {
			out.Data[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:])))
		}
	}
	return out, nil
}

// Raw returns a tensor's bytes as they sit in the file, for dtypes whose
// payload is packed rather than numeric -- MXFP4 weights arrive as U8
// blocks of nibbles beside U8 e8m0 scales, and expanding those to float32
// would cost a couple of gigabytes a layer. The slice aliases the mapped
// file when there is one, so treat it as read-only and do not retain it
// past Close.
func (f *File) Raw(name string) (data []byte, shape []int, err error) {
	e, ok := f.entries[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	if f.data != nil {
		lo, hi := f.dataOff+e.DataOffsets[0], f.dataOff+e.DataOffsets[1]
		if hi > int64(len(f.data)) || lo < 0 || lo > hi {
			return nil, nil, fmt.Errorf("safetensors: tensor %q extends past the file", name)
		}
		data = f.data[lo:hi]
	} else {
		data = make([]byte, e.DataOffsets[1]-e.DataOffsets[0])
		if _, err := f.r.ReadAt(data, f.dataOff+e.DataOffsets[0]); err != nil {
			return nil, nil, fmt.Errorf("safetensors: reading tensor %q: %w", name, err)
		}
	}
	return data, append([]int(nil), e.Shape...), nil
}

// f16to32 widens an IEEE 754 binary16 value, including subnormals,
// infinities, and NaNs.
func f16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign) // signed zero
		}
		// Subnormal: normalize into the f32 exponent range.
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (frac&0x3ff)<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13) // inf / NaN
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	}
}

// Save writes tensors as an F32 safetensors checkpoint, names sorted, with
// optional __metadata__ strings.
func Save(w io.Writer, tensors map[string]*tensai.Tensor, meta map[string]string) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if name == "__metadata__" {
			return fmt.Errorf("safetensors: %q is a reserved name", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)

	// Build the JSON header by hand to keep the key order deterministic.
	header := []byte{'{'}
	if len(meta) > 0 {
		m, err := json.Marshal(meta)
		if err != nil {
			return err
		}
		header = append(header, `"__metadata__":`...)
		header = append(header, m...)
		if len(names) > 0 {
			header = append(header, ',')
		}
	}
	var off int64
	for i, name := range names {
		t := tensors[name]
		if err := t.Validate(); err != nil {
			return fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		end := off + int64(len(t.Data))*4
		e, err := json.Marshal(entry{Dtype: "F32", Shape: t.Shape, DataOffsets: [2]int64{off, end}})
		if err != nil {
			return err
		}
		n, err := json.Marshal(name)
		if err != nil {
			return err
		}
		header = append(header, n...)
		header = append(header, ':')
		header = append(header, e...)
		if i < len(names)-1 {
			header = append(header, ',')
		}
		off = end
	}
	header = append(header, '}')
	// Pad with spaces so the byte buffer starts 8-aligned, as the
	// reference implementation does.
	for (8+len(header))%8 != 0 {
		header = append(header, ' ')
	}

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := w.Write(header); err != nil {
		return err
	}
	buf := make([]byte, 0, 1<<16)
	for _, name := range names {
		buf = buf[:0]
		for _, v := range tensors[name].Data {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// Shards is a checkpoint split across several safetensors files, as
// described by a model.safetensors.index.json. It offers the same lazy
// per-tensor access as File.
type Shards struct {
	files  map[string]*File // shard filename -> open file
	byName map[string]*File // tensor name -> its shard
	names  []string
}

// OpenSharded opens a sharded checkpoint via its index file (typically
// model.safetensors.index.json); shard files are resolved relative to it.
// Close the returned Shards when done.
func OpenSharded(indexPath string) (*Shards, error) {
	raw, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("safetensors: parsing index: %w", err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("safetensors: index has no weight_map")
	}
	dir := filepath.Dir(indexPath)
	s := &Shards{files: map[string]*File{}, byName: map[string]*File{}}
	for name, shard := range idx.WeightMap {
		f, ok := s.files[shard]
		if !ok {
			f, err = Open(filepath.Join(dir, shard))
			if err != nil {
				s.Close()
				return nil, err
			}
			s.files[shard] = f
		}
		if _, _, ok := f.Info(name); !ok {
			s.Close()
			return nil, fmt.Errorf("safetensors: index maps %q to %s, which lacks it", name, shard)
		}
		s.byName[name] = f
		s.names = append(s.names, name)
	}
	sort.Strings(s.names)
	return s, nil
}

// Close closes every shard.
func (s *Shards) Close() error {
	var first error
	for _, f := range s.files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Names lists the tensor names, sorted.
func (s *Shards) Names() []string { return append([]string(nil), s.names...) }

// Info reports a tensor's dtype and shape without loading its data.
func (s *Shards) Info(name string) (dtype string, shape []int, ok bool) {
	f, found := s.byName[name]
	if !found {
		return "", nil, false
	}
	return f.Info(name)
}

// Tensor loads one tensor from its shard.
func (s *Shards) Tensor(name string) (*tensai.Tensor, error) {
	f, ok := s.byName[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return f.Tensor(name)
}

// Raw returns one tensor's packed bytes from its shard.
func (s *Shards) Raw(name string) ([]byte, []int, error) {
	f, ok := s.byName[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return f.Raw(name)
}

// SaveFile writes tensors to path via Save.
func SaveFile(path string, tensors map[string]*tensai.Tensor, meta map[string]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Save(f, tensors, meta); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

package qwenimage

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"unsafe"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/mmapfile"
	"github.com/mattn/tensai/quant"
)

// Quantizing seven billion parameters takes about three minutes and
// gives the same answer every time, so the result is written once beside
// the weights and mapped back afterwards. A mapped cache is not read at
// all until a kernel touches a page, which is why the second run starts
// in a second rather than three minutes.
//
// The file is tied to the checkpoint it came from by the names, sizes
// and modification times of the .safetensors beside it. Anything else
// and the cache is ignored and rebuilt.

const (
	cacheMagic   = "tensai-qwenimage\x00"
	cacheVersion = 1
	// Every payload starts on an eight-byte boundary so the mapped bytes
	// can be viewed as floats and int32s where they lie.
	cacheAlign = 8
)

var errCacheStale = errors.New("qwenimage: the cache does not match the checkpoint")

// codec visits a model's weight slots in one fixed order. The writer
// stores what it finds, the reader fills the slots from the file, and
// the order keeps them in step.
type codec interface {
	vec(*[]tensai.Float)
	lin(**linear)
}

// walk visits the denoising transformer's weights.
func (m *Transformer) walk(c codec) {
	for _, l := range []**linear{&m.imgIn, &m.txtIn, &m.txtOut, &m.timeIn, &m.timeUp, &m.modulation, &m.normOut, &m.projOut} {
		c.lin(l)
	}
	c.vec(&m.txtNorm)
	for _, b := range m.blocks {
		for _, l := range []**linear{&b.toQ, &b.toK, &b.toV, &b.toOut, &b.mlpProj, &b.mlpGate, &b.mlpOut} {
			c.lin(l)
		}
		c.vec(&b.normQ)
		c.vec(&b.normK)
	}
}

// walk visits the prompt encoder's weights.
func (t *TextEncoder) walk(c codec) {
	for _, l := range t.layers {
		for _, p := range []**linear{&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down} {
			c.lin(p)
		}
		for _, p := range []*[]tensai.Float{&l.inNorm, &l.postNorm, &l.qNorm, &l.kNorm} {
			c.vec(p)
		}
	}
}

// stamp identifies a checkpoint by what its files look like on disk.
func stamp(dir string) ([]byte, error) {
	names, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("qwenimage: no weights under %s", dir)
	}
	sort.Strings(names)
	var out []byte
	for _, n := range names {
		st, err := os.Stat(n)
		if err != nil {
			return nil, err
		}
		out = append(out, filepath.Base(n)...)
		out = binary.LittleEndian.AppendUint64(out, uint64(st.Size()))
		out = binary.LittleEndian.AppendUint64(out, uint64(st.ModTime().UnixNano()))
	}
	return out, nil
}

// cachePath is where a checkpoint's quantized form lives.
func cachePath(dir string, bits int) string {
	return filepath.Join(dir, fmt.Sprintf("tensai-q%d.cache", bits))
}

// bytesOf views a slice's backing array as bytes.
func bytesOf[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*int(unsafe.Sizeof(s[0])))
}

// sliceOf is the inverse, viewing mapped bytes as a typed slice.
func sliceOf[T any](b []byte) []T {
	if len(b) == 0 {
		return nil
	}
	var z T
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/int(unsafe.Sizeof(z)))
}

// cacheWriter appends each weight it is shown.
type cacheWriter struct {
	w   *bufio.Writer
	n   int64
	err error
}

func (c *cacheWriter) blob(b []byte) {
	if c.err != nil {
		return
	}
	var head [8]byte
	binary.LittleEndian.PutUint64(head[:], uint64(len(b)))
	if _, c.err = c.w.Write(head[:]); c.err != nil {
		return
	}
	c.n += 8
	if _, c.err = c.w.Write(b); c.err != nil {
		return
	}
	c.n += int64(len(b))
	if pad := (cacheAlign - c.n%cacheAlign) % cacheAlign; pad != 0 {
		var zero [cacheAlign]byte
		if _, c.err = c.w.Write(zero[:pad]); c.err != nil {
			return
		}
		c.n += pad
	}
}

func (c *cacheWriter) num(v int) { c.blob(binary.LittleEndian.AppendUint64(nil, uint64(v))) }

func (c *cacheWriter) vec(v *[]tensai.Float) { c.blob(bytesOf(*v)) }

func (c *cacheWriter) lin(l **linear) {
	m := *l
	if m.q == nil {
		c.num(0)
		c.num(m.f.Rows)
		c.num(m.f.Cols)
		c.blob(bytesOf(m.f.Data))
		return
	}
	c.num(1)
	c.num(m.q.Rows)
	c.num(m.q.Cols)
	c.blob(bytesOf(m.q.Q))
	c.blob(bytesOf(m.q.Scale))
	c.blob(bytesOf(m.q.ColSum64))
}

// cacheReader hands slices of the mapped file back to the slots.
type cacheReader struct {
	b   []byte
	off int
	err error
}

func (c *cacheReader) blob() []byte {
	if c.err != nil {
		return nil
	}
	if c.off+8 > len(c.b) {
		c.err = errCacheStale
		return nil
	}
	n := int(binary.LittleEndian.Uint64(c.b[c.off:]))
	c.off += 8
	if n < 0 || c.off+n > len(c.b) {
		c.err = errCacheStale
		return nil
	}
	out := c.b[c.off : c.off+n : c.off+n]
	c.off += n
	c.off += (cacheAlign - c.off%cacheAlign) % cacheAlign
	return out
}

func (c *cacheReader) num() int {
	b := c.blob()
	if len(b) != 8 {
		c.err = errCacheStale
		return 0
	}
	return int(binary.LittleEndian.Uint64(b))
}

func (c *cacheReader) vec(v *[]tensai.Float) { *v = sliceOf[tensai.Float](c.blob()) }

func (c *cacheReader) lin(l **linear) {
	kind := c.num()
	rows, cols := c.num(), c.num()
	if c.err != nil {
		return
	}
	switch kind {
	case 0:
		*l = &linear{f: &tensai.Matrix{Rows: rows, Cols: cols, Data: sliceOf[tensai.Float](c.blob())}}
	case 1:
		q := &quant.QMatrix{Rows: rows, Cols: cols}
		q.Q = sliceOf[int8](c.blob())
		q.Scale = sliceOf[tensai.Float](c.blob())
		q.ColSum64 = sliceOf[int32](c.blob())
		*l = &linear{q: q}
	default:
		c.err = errCacheStale
	}
}

// writeCache stores a quantized model beside its checkpoint. A failure
// only costs the next run the same work, so callers report it and carry
// on rather than giving up on a model they already hold.
func writeCache(dir string, bits int, walk func(codec)) error {
	src, err := stamp(dir)
	if err != nil {
		return err
	}
	path := cachePath(dir, bits)
	tmp, err := os.CreateTemp(dir, "tensai-cache-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	c := &cacheWriter{w: bufio.NewWriterSize(tmp, 1<<20)}
	c.blob([]byte(cacheMagic))
	c.num(cacheVersion)
	c.num(bits)
	c.blob(src)
	walk(c)
	if c.err != nil {
		tmp.Close()
		return c.err
	}
	if err := c.w.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// cacheFiles pins the mapped caches for the life of the process, so the
// finalizer on os.File never closes one under a live mapping.
var cacheFiles []*os.File

// readCache fills a model's weight slots from a mapped cache. A missing,
// stale or corrupt file is reported so the caller can quantize instead.
func readCache(dir string, bits int, walk func(codec)) error {
	src, err := stamp(dir)
	if err != nil {
		return err
	}
	f, err := os.Open(cachePath(dir, bits))
	if err != nil {
		return err
	}
	data, unmap, err := mmapfile.Map(f)
	if err != nil {
		f.Close()
		return err
	}
	bad := func(err error) error {
		unmap()
		f.Close()
		return err
	}
	c := &cacheReader{b: data}
	if string(c.blob()) != cacheMagic || c.num() != cacheVersion || c.num() != bits {
		return bad(errCacheStale)
	}
	if string(c.blob()) != string(src) {
		return bad(errCacheStale)
	}
	walk(c)
	if c.err != nil {
		return bad(c.err)
	}
	cacheFiles = append(cacheFiles, f)
	return nil
}

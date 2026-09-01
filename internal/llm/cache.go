package llm

// Repack caching: the first .gguf load repacks the stored blocks into
// tensai's tile layouts and then writes every weight the model holds
// into one flat file next to the source; later loads mmap that file
// and point the matrices straight into it, skipping the repack. More
// importantly for models near the machine's memory, the weights become
// clean file-backed pages the kernel can drop and re-read at will,
// instead of anonymous memory it can only evict through swap — the
// same structural property llama.cpp gets from running straight off
// its mmap'd file.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"unsafe"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/mmapfile"
	"github.com/mattn/tensai/quant"
)

const cacheMagic = "TSAICCH\x00"

// cacheFormat names the serialized layouts and the walk order; bump it
// whenever either changes so stale caches rewrite instead of decoding
// garbage.
const cacheFormat = 2

// Record kinds, one per weight representation the model can hold.
const (
	kindNil = iota
	kindVec
	kindMat
	kindTensor
	kindQ
	kindQ4
	kindQ8G
	kindMX
)

var errCacheCorrupt = errors.New("repack cache is truncated or corrupt")

func cachePath(gguf string, bits int, direct bool) string {
	mode := fmt.Sprintf("q%d", bits)
	if !direct {
		mode += "r"
	}
	return gguf + ".tensai-" + mode + ".cache"
}

// bytesOf views a slice's backing array as bytes.
func bytesOf[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*int(unsafe.Sizeof(s[0])))
}

// sliceOf is the inverse, viewing a cache blob as a typed slice.
func sliceOf[T any](b []byte) []T {
	if len(b) == 0 {
		return nil
	}
	var z T
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/int(unsafe.Sizeof(z)))
}

// weightCodec visits every weight slot of the model in one fixed
// order; the writer stores what it finds, the reader fills the slots
// from the file.
type weightCodec interface {
	vec(*[]float32)
	mat(**tensai.Matrix)
	tensor(**tensai.Tensor)
	qmat(**qmat)
}

func walkWeights(m *qwen, c weightCodec) {
	c.tensor(&m.embed)
	c.vec(&m.normW)
	c.mat(&m.lmT)
	c.qmat(&m.qLmT)
	// gemma4's per-layer embedding projection. The table itself is not
	// here: it stays in the gguf and is read a row at a time.
	c.mat(&m.wPleIn)
	c.qmat(&m.qPleIn)
	c.vec(&m.pleNorm)
	for i := range m.blocks {
		b := &m.blocks[i]
		c.vec(&b.ln1)
		c.vec(&b.ln2)
		c.vec(&b.qNorm)
		c.vec(&b.kNorm)
		c.vec(&b.postAttn)
		c.vec(&b.postFFN)
		c.vec(&b.sinks)
		c.vec(&b.bo)
		c.vec(&b.bQKV)
		c.mat(&b.wQKV)
		c.qmat(&b.qQKV)
		c.mat(&b.wo)
		c.qmat(&b.qo)
		c.mat(&b.wGU)
		c.qmat(&b.qGU)
		c.mat(&b.wDown)
		c.qmat(&b.qDown)
		c.mat(&b.router)
		c.vec(&b.routerBias)
		c.qmat(&b.sharedGU)
		c.qmat(&b.sharedDown)
		c.vec(&b.sharedGate)
		c.mat(&b.wPleGate)
		c.qmat(&b.qPleGate)
		c.mat(&b.wPleProj)
		c.qmat(&b.qPleProj)
		c.vec(&b.plePost)
		c.vec(&b.outScale)
		c.vec(&b.ropeFF)
		for e := range b.experts {
			x := &b.experts[e]
			c.qmat(&x.qGU)
			c.qmat(&x.qDown)
			c.vec(&x.guBias)
			c.vec(&x.downBias)
		}
	}
}

// cacheHeader carries what must match for a cache to be reused: the
// format, the load mode, and the source file's identity.
func cacheHeader(bits int, direct bool, srcSize int64, srcMtime int64) [64]byte {
	var h [64]byte
	copy(h[:8], cacheMagic)
	binary.LittleEndian.PutUint32(h[8:], cacheFormat)
	binary.LittleEndian.PutUint32(h[12:], uint32(bits))
	var d uint32
	if direct {
		d = 1
	}
	binary.LittleEndian.PutUint32(h[16:], d)
	binary.LittleEndian.PutUint64(h[24:], uint64(srcSize))
	binary.LittleEndian.PutUint64(h[32:], uint64(srcMtime))
	return h
}

type cacheWriter struct {
	w   *bufio.Writer
	off int64
	err error
}

func (w *cacheWriter) raw(b []byte) {
	if w.err != nil {
		return
	}
	if _, err := w.w.Write(b); err != nil {
		w.err = err
		return
	}
	w.off += int64(len(b))
}

func (w *cacheWriter) u64(v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.raw(b[:])
}

// blob writes a length-prefixed byte run whose body starts 64-byte
// aligned in the file, so the mmap'd slices satisfy every element type
// and line up for the kernels' 32-byte loads.
func (w *cacheWriter) blob(b []byte) {
	w.u64(uint64(len(b)))
	if len(b) == 0 {
		return
	}
	var z [64]byte
	w.raw(z[:int(-w.off&63)])
	w.raw(b)
}

func (w *cacheWriter) vec(p *[]float32) {
	if *p == nil {
		w.u64(kindNil)
		return
	}
	w.u64(kindVec)
	w.blob(bytesOf(*p))
}

func (w *cacheWriter) mat(p **tensai.Matrix) {
	m := *p
	if m == nil {
		w.u64(kindNil)
		return
	}
	w.u64(kindMat)
	w.u64(uint64(m.Rows))
	w.u64(uint64(m.Cols))
	w.blob(bytesOf(m.Data))
}

func (w *cacheWriter) tensor(p **tensai.Tensor) {
	t := *p
	if t == nil {
		w.u64(kindNil)
		return
	}
	w.u64(kindTensor)
	w.u64(uint64(len(t.Shape)))
	for _, d := range t.Shape {
		w.u64(uint64(d))
	}
	w.blob(bytesOf(t.Data))
}

func (w *cacheWriter) qmat(p **qmat) {
	q := *p
	switch {
	case q == nil:
		w.u64(kindNil)
	case q.q8 != nil:
		w.u64(kindQ)
		w.u64(uint64(q.q8.Rows))
		w.u64(uint64(q.q8.Cols))
		w.blob(bytesOf(q.q8.Q))
		w.blob(bytesOf(q.q8.Scale))
		w.blob(bytesOf(q.q8.ColSum64))
	case q.q4 != nil:
		w.u64(kindQ4)
		w.u64(uint64(q.q4.Rows))
		w.u64(uint64(q.q4.Cols))
		w.u64(uint64(q.q4.Group))
		w.blob(bytesOf(q.q4.Q))
		w.blob(bytesOf(q.q4.Scale))
		w.blob(bytesOf(q.q4.ScaleMin))
	case q.q8g != nil:
		w.u64(kindQ8G)
		w.u64(uint64(q.q8g.Rows))
		w.u64(uint64(q.q8g.Cols))
		w.u64(uint64(q.q8g.Group))
		w.blob(bytesOf(q.q8g.Q))
		w.blob(bytesOf(q.q8g.Scale))
		w.blob(bytesOf(q.q8g.ColSum64))
	case q.mx != nil:
		w.u64(kindMX)
		w.u64(uint64(q.mx.Rows))
		w.u64(uint64(q.mx.Cols))
		w.blob(bytesOf(q.mx.Q))
		w.blob(bytesOf(q.mx.Scale))
		w.blob(bytesOf(q.mx.ColSum64))
	default:
		w.err = errors.New("qmat carries no backing matrix to cache")
	}
}

type cacheReader struct {
	b   []byte
	off int
	err error
}

func (r *cacheReader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || n > len(r.b)-r.off {
		r.err = errCacheCorrupt
		return nil
	}
	s := r.b[r.off : r.off+n : r.off+n]
	r.off += n
	return s
}

func (r *cacheReader) u64() uint64 {
	b := r.take(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (r *cacheReader) num() int {
	n := r.u64()
	if n > uint64(len(r.b)) {
		r.err = errCacheCorrupt
		return 0
	}
	return int(n)
}

func (r *cacheReader) blob() []byte {
	n := r.num()
	if n == 0 {
		return nil
	}
	r.take(-r.off & 63)
	return r.take(n)
}

func (r *cacheReader) vec(p *[]float32) {
	switch r.u64() {
	case kindNil:
	case kindVec:
		*p = sliceOf[float32](r.blob())
	default:
		r.err = errCacheCorrupt
	}
}

func (r *cacheReader) mat(p **tensai.Matrix) {
	switch r.u64() {
	case kindNil:
	case kindMat:
		rows, cols := r.num(), r.num()
		*p = &tensai.Matrix{Rows: rows, Cols: cols, Data: sliceOf[float32](r.blob())}
	default:
		r.err = errCacheCorrupt
	}
}

func (r *cacheReader) tensor(p **tensai.Tensor) {
	switch r.u64() {
	case kindNil:
	case kindTensor:
		shape := make([]int, r.num())
		for i := range shape {
			shape[i] = r.num()
		}
		*p = &tensai.Tensor{Shape: shape, Data: sliceOf[float32](r.blob())}
	default:
		r.err = errCacheCorrupt
	}
}

func (r *cacheReader) qmat(p **qmat) {
	switch r.u64() {
	case kindNil:
	case kindQ:
		q := &quant.QMatrix{Rows: r.num(), Cols: r.num()}
		q.Q = sliceOf[int8](r.blob())
		q.Scale = sliceOf[float32](r.blob())
		q.ColSum64 = sliceOf[int32](r.blob())
		*p = qmatQ8(q)
	case kindQ4:
		q := &quant.Q4Matrix{Rows: r.num(), Cols: r.num(), Group: r.num()}
		q.Q = sliceOf[uint8](r.blob())
		q.Scale = sliceOf[float32](r.blob())
		q.ScaleMin = sliceOf[uint32](r.blob())
		*p = qmatQ4(q)
	case kindQ8G:
		q := &quant.Q8GMatrix{Rows: r.num(), Cols: r.num(), Group: r.num()}
		q.Q = sliceOf[int8](r.blob())
		q.Scale = sliceOf[float32](r.blob())
		q.ColSum64 = sliceOf[int32](r.blob())
		*p = qmatQ8G(q)
	case kindMX:
		q := &quant.MXFP4Matrix{Rows: r.num(), Cols: r.num()}
		q.Q = sliceOf[uint8](r.blob())
		q.Scale = sliceOf[float32](r.blob())
		q.ColSum64 = sliceOf[int32](r.blob())
		*p = qmatMX(q)
	default:
		r.err = errCacheCorrupt
	}
}

// cacheFiles pins the mmap'd cache files' descriptors for the life of
// the process, so the finalizer on os.File never closes one under a
// live mapping.
var cacheFiles []*os.File

// loadWeightCache maps cpath and rebuilds the model's weights from it.
// Any error means the caller should do the normal load (and rewrite
// the cache); a stale or corrupt file is reported, a missing one is
// just os.IsNotExist.
func loadWeightCache(cpath, src string, bits int, direct bool, cfg config, headSz int) (*qwen, error) {
	st, err := os.Stat(src)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(cpath)
	if err != nil {
		return nil, err
	}
	data, closer, err := mmapfile.Map(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	bad := func(err error) (*qwen, error) {
		closer()
		f.Close()
		return nil, err
	}
	if len(data) < 64 {
		return bad(errCacheCorrupt)
	}
	want := cacheHeader(bits, direct, st.Size(), st.ModTime().UnixNano())
	if [64]byte(data[:64]) != want {
		return bad(errors.New("repack cache is stale"))
	}

	m := &qwen{cfg: cfg, headSz: headSz}
	m.blocks = make([]qblock, cfg.Layers)
	for i := range m.blocks {
		blockShape(&m.blocks[i], cfg, i)
	}
	r := &cacheReader{b: data, off: 64}
	walkWeights(m, r)
	if r.err != nil {
		return bad(r.err)
	}
	if r.off != len(data) {
		return bad(errCacheCorrupt)
	}
	// The per-layer embedding table is never copied into the cache: it
	// is the largest tensor in the file and a step reads one row of it,
	// so the model keeps reading it where it already sits.
	if cfg.PLEDim > 0 {
		m.embScale = float32(math.Sqrt(float64(cfg.HiddenSize)))
		if m.ple, err = newPLETable(src, cfg); err != nil {
			return bad(err)
		}
	}
	m.initRopeFreqs()
	cacheFiles = append(cacheFiles, f)
	return m, nil
}

// writeWeightCache serializes the model's weights next to the source
// file, through a temp name so a crash never leaves a half cache
// behind.
func writeWeightCache(cpath, src string, bits int, direct bool, m *qwen) error {
	st, err := os.Stat(src)
	if err != nil {
		return err
	}
	tmp := cpath + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := &cacheWriter{w: bufio.NewWriterSize(f, 1<<20)}
	h := cacheHeader(bits, direct, st.Size(), st.ModTime().UnixNano())
	w.raw(h[:])
	walkWeights(m, w)
	if w.err == nil {
		w.err = w.w.Flush()
	}
	if err := f.Close(); w.err == nil {
		w.err = err
	}
	if w.err != nil {
		os.Remove(tmp)
		return w.err
	}
	return os.Rename(tmp, cpath)
}

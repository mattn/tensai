package autograd

import (
	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/dims"
)

// Tape recycles the buffers a graph allocates. A training step builds a
// fresh graph every time -- that is what define-by-run means -- and throws
// it away again, so without a tape every step asks the allocator for every
// intermediate value and gradient it touches, and the collector takes them
// back a moment later.
//
// Bind the parameters once, then call Reset at the end of each step:
//
//	tape := autograd.NewTape()
//	tape.Bind(w1, b1, w2, b2)
//	for step := 0; step < steps; step++ {
//		trainer.Step(forward(x).MSELoss(y))
//		tape.Reset()
//	}
//
// Reset hands every buffer from that step back to the pool, so the next
// step reuses the same memory. That makes the rule for using a tape a
// simple one: after Reset, nothing from the finished step may be read --
// no node's Value, no node's Grad. Parameter values are never recycled
// (the tape only owns what operations produce), so the trained weights
// themselves are always safe to keep. Copy anything else you need with
// Clone before resetting.
//
// A Tape is not safe for concurrent use; give each training goroutine its
// own.
type Tape struct {
	free map[int][][]tensai.Float
	used [][]tensai.Float
}

// NewTape returns an empty tape.
func NewTape() *Tape {
	return &Tape{free: map[int][][]tensai.Float{}}
}

// Bind ties nodes to the tape. Every node built from them -- the whole
// graph, since operations inherit the tape from their parents -- takes its
// values and gradients from it. Bind the parameters once, outside the
// training loop.
func (t *Tape) Bind(nodes ...*Node) {
	for _, n := range nodes {
		n.tape = t
	}
}

// Reset returns every buffer handed out since the last Reset to the pool.
// Values and gradients from the finished step must not be read afterwards.
func (t *Tape) Reset() {
	if t == nil {
		return
	}
	for _, buf := range t.used {
		t.free[len(buf)] = append(t.free[len(buf)], buf)
	}
	t.used = t.used[:0]
}

// buffer returns a buffer of n elements. Its contents are whatever the
// previous step left there, so only callers that write every element may
// use it.
func (t *Tape) buffer(n int) []tensai.Float {
	if t == nil {
		return make([]tensai.Float, n)
	}
	pool := t.free[n]
	if len(pool) == 0 {
		buf := make([]tensai.Float, n)
		t.used = append(t.used, buf)
		return buf
	}
	buf := pool[len(pool)-1]
	t.free[n] = pool[:len(pool)-1]
	t.used = append(t.used, buf)
	return buf
}

// tensor returns a tensor of the given shape whose data is not cleared.
// The shape is adopted, not copied: callers pass a shape they own or one
// that belongs to a node whose shape is never mutated.
func (t *Tape) tensor(shape []int) *tensai.Tensor {
	if t == nil {
		return &tensai.Tensor{Shape: shape, Data: make([]tensai.Float, dims.Prod(shape))}
	}
	return &tensai.Tensor{Shape: shape, Data: t.buffer(dims.Prod(shape))}
}

// zeros returns a cleared tensor of the given shape.
func (t *Tape) zeros(shape []int) *tensai.Tensor {
	out := t.tensor(shape)
	clear(out.Data)
	return out
}

// tapeOf returns the tape a result node inherits: the first one any parent
// carries.
func tapeOf(parents ...*Node) *Tape {
	for _, p := range parents {
		// LayerNorm passes an optional gain and bias, either of which may
		// be nil.
		if p != nil && p.tape != nil {
			return p.tape
		}
	}
	return nil
}

// matmulShape returns the shape of the stacked product tensai.MatMul,
// MatMulTN or MatMulNT produces for these operands, so the output buffer
// can come from the tape before the product runs. The kernels validate it
// again, so a mistake here surfaces as an error rather than bad math.
func matmulShape(a, b []int, mode gemmMode) []int {
	na, nb := len(a), len(b)
	var m, n int
	switch mode {
	case gemmTN:
		m, n = a[na-1], b[nb-1]
	case gemmNT:
		m, n = a[na-2], b[nb-2]
	default:
		m, n = a[na-2], b[nb-1]
	}
	if na == 2 && nb == 2 {
		return []int{m, n}
	}
	batch, err := dims.Broadcast(a[:na-2], b[:nb-2])
	if err != nil {
		panic(err.Error())
	}
	return append(batch, m, n)
}

// gemmMode names which operand of a product is transposed, mirroring the
// three MatMul entry points in the root package.
type gemmMode int

const (
	gemmNN gemmMode = iota // a * b
	gemmTN                 // a^T * b
	gemmNT                 // a * b^T
)

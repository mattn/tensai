package llm

// A checkpoint quantized to ternary keeps its weights in a rotated
// basis: each weight matrix was multiplied along its input dimension by
// a blockwise Walsh-Hadamard transform, with a fixed sign flip per input
// channel ahead of it, before its weights were rounded to -1, 0 and +1.
// The rotation spreads every activation's energy across a block, which
// is what makes three levels per weight enough. It is folded into the
// stored weights, so the runtime has to apply the same transform to the
// activations those weights read, and the inverse to what comes out of
// a table that stores rotated rows. PrismML's Bonsai declares all of
// this under prism.hadamard.* in its gguf metadata.

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/gguf"
	"github.com/mattn/tensai/internal/kernels"
)

// hadamard is the transform one weight's input goes through: an optional
// permutation, a sign per channel, then the normalized Sylvester
// Walsh-Hadamard transform of each block of the input.
type hadamard struct {
	block int
	signs []float32 // one per input channel, +1 or -1
	perm  []int     // dst[i] = src[perm[i]] before the signs; nil for none
	pool  sync.Pool // transformed rows, reused across calls
}

// hadamardSpec is what the metadata declares for the whole file.
type hadamardSpec struct {
	block    int
	signs    map[int][]float32 // by input width
	weights  map[string]bool   // tensors whose input is transformed
	inverses map[string]bool   // tables whose rows are transformed back
	vGrouped bool              // ssm_out reads its value heads grouped, not tiled
}

// readHadamardSpec parses prism.hadamard.* and refuses anything it does
// not know how to apply: a file whose transform went unapplied would
// load and produce noise.
func readHadamardSpec(g *gguf.File) (*hadamardSpec, error) {
	version, ok := g.Int("prism.hadamard.version")
	if !ok {
		return nil, nil
	}
	if version != 1 {
		return nil, fmt.Errorf("prism.hadamard.version %d is not understood", version)
	}
	str := func(key string) string {
		s, _ := g.String("prism.hadamard." + key)
		return s
	}
	if t := str("transform"); t != "normalized-sylvester-walsh-hadamard" {
		return nil, fmt.Errorf("prism.hadamard.transform %q is not understood", t)
	}
	if a := str("axis"); a != "input-last-dimension" {
		return nil, fmt.Errorf("prism.hadamard.axis %q is not understood", a)
	}
	if m := str("sign_mode"); m != "explicit" {
		return nil, fmt.Errorf("prism.hadamard.sign_mode %q is not understood", m)
	}
	block, _ := g.Int("prism.hadamard.block_size")
	if block <= 0 || block&(block-1) != 0 {
		return nil, fmt.Errorf("prism.hadamard.block_size %d is not a power of two", block)
	}
	spec := &hadamardSpec{
		block:    int(block),
		signs:    map[int][]float32{},
		weights:  map[string]bool{},
		inverses: map[string]bool{},
	}
	widths := g.Ints("prism.hadamard.sign_widths")
	values := g.Ints("prism.hadamard.sign_values")
	off := 0
	for _, w := range widths {
		width := int(w)
		if width <= 0 || width%spec.block != 0 || off+width > len(values) {
			return nil, errors.New("prism.hadamard.sign_widths do not match sign_values")
		}
		s := make([]float32, width)
		for i := range s {
			switch values[off+i] {
			case 1, -1:
				s[i] = float32(values[off+i])
			default:
				return nil, errors.New("prism.hadamard.sign_values must be +1 or -1")
			}
		}
		spec.signs[width] = s
		off += width
	}
	if off != len(values) {
		return nil, errors.New("prism.hadamard.sign_values has values no width claims")
	}
	for _, n := range g.Strings("prism.hadamard.weight_names") {
		spec.weights[n] = true
	}
	for _, n := range g.Strings("prism.hadamard.inverse_weight_names") {
		spec.inverses[n] = true
	}
	if len(spec.weights) == 0 {
		return nil, errors.New("prism.hadamard.weight_names is empty")
	}
	if v, ok := g.KV("prism.hadamard.gdn_v_grouped"); ok {
		spec.vGrouped, _ = v.(bool)
	}
	return spec, nil
}

// forWidth builds the transform for a weight reading width inputs.
func (s *hadamardSpec) forWidth(width int) (*hadamard, error) {
	if width%s.block != 0 {
		return nil, fmt.Errorf("prism.hadamard: block %d does not divide an input of %d", s.block, width)
	}
	signs, ok := s.signs[width]
	if !ok {
		return nil, fmt.Errorf("prism.hadamard: no signs for an input of %d", width)
	}
	return &hadamard{block: s.block, signs: signs}, nil
}

// tiledToGrouped is the permutation ssm_out's input takes when the
// weight was folded in HF's grouped value-head order while the layer
// produces them tiled: position (j, g, d) moves to (g, j, d) for key
// head g, replica j and value dimension d.
func tiledToGrouped(hd, nk, rep int) []int {
	perm := make([]int, hd*nk*rep)
	for g := 0; g < nk; g++ {
		for j := 0; j < rep; j++ {
			for d := 0; d < hd; d++ {
				perm[(g*rep+j)*hd+d] = (j*nk+g)*hd + d
			}
		}
	}
	return perm
}

// apply returns the transformed copy of x, in a buffer from the pool;
// the caller hands it back with release once the weight has read it.
func (h *hadamard) apply(x []float32) []float32 {
	var y []float32
	if p, ok := h.pool.Get().(*[]float32); ok && len(*p) == len(x) {
		y = *p
	} else {
		y = make([]float32, len(x))
	}
	h.applyInto(y, x)
	return y
}

func (h *hadamard) release(y []float32) {
	h.pool.Put(&y)
}

// applyInto writes the transform of x into y, which must not alias it.
// The normalization of every block is one scale at the end.
func (h *hadamard) applyInto(y, x []float32) {
	if h.perm != nil {
		for i, p := range h.perm {
			y[i] = x[p]
		}
		kernels.MulSlices(y, y, h.signs)
	} else {
		kernels.MulSlices(y, x, h.signs)
	}
	scale := float32(1 / math.Sqrt(float64(h.block)))
	for off := 0; off+h.block <= len(y); off += h.block {
		kernels.Hadamard(y[off:off+h.block], scale)
	}
}

// applyRows transforms every row of a batch.
func (h *hadamard) applyRows(x *tensai.Matrix) *tensai.Matrix {
	y := tensai.NewMatrix(x.Rows, x.Cols)
	for r := 0; r < x.Rows; r++ {
		h.applyInto(y.Data[r*x.Cols:(r+1)*x.Cols], x.Data[r*x.Cols:(r+1)*x.Cols])
	}
	return y
}

// invert applies the inverse in place: the blocks, then the signs. The
// transform is orthogonal and symmetric, so its inverse is itself, and
// the signs undo themselves.
func (h *hadamard) invert(x []float32) {
	scale := float32(1 / math.Sqrt(float64(h.block)))
	for off := 0; off+h.block <= len(x); off += h.block {
		kernels.Hadamard(x[off:off+h.block], scale)
	}
	kernels.MulSlices(x, x, h.signs)
}

// fwht is the normalized Walsh-Hadamard transform of v in place, in the
// Sylvester order the butterfly produces: entry (r, c) of the matrix is
// (-1)^popcount(r&c) / sqrt(n). len(v) must be a power of two.
func fwht(v []float32) {
	kernels.Hadamard(v, float32(1/math.Sqrt(float64(len(v)))))
}

// The rotation kinds a weight records for the cache.
const (
	rotNone    = iota
	rotPlain   // signs and blocks
	rotGrouped // value heads put in grouped order first (ssm_out)
)

// rotate wraps a quantized weight so that what it reads is transformed
// first.
func (q *qmat) rotate(h *hadamard, rot int) {
	q.rot = rot
	f, mm := q.f, q.mm
	q.f = func(x, out []float32) error {
		y := h.apply(x)
		err := f(y, out)
		h.release(y)
		return err
	}
	q.mm = func(x, out *tensai.Matrix) error { return mm(h.applyRows(x), out) }
}

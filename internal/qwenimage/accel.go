package qwenimage

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
)

// The feed-forward is seventy per cent of a block's arithmetic and asks
// nothing of the position scheme or the mask, so it is the part of the
// transformer moved first: three projections and a gate, over weights
// that sit on the device for the whole run. Attention also runs on the
// device after the host applies normalization and rotary embedding.
//
// What decides whether it fits is not the arithmetic but the memory. A
// buffer here is capped at 128MiB, which int8 weights are under and
// float ones are not, and the device is dropped somewhere past five and
// a half gigabytes resident — by the driver, silently, with allocations
// still reporting success afterwards and the process falling over later.
// Nothing reports that, so the only defence is to count what is about to
// be uploaded and refuse before asking.

// deviceWeights is one block's feed-forward, resident.
type deviceWeights struct {
	proj, gate, out *deviceLinear
	// rot is the Hadamard matrix the output projection's input is
	// rotated by, shared by every block, nil when it is not rotated.
	rot *gpu.Tensor
}

// deviceLinear is one weight matrix on the device, at whichever width
// the model was loaded with.
type deviceLinear struct {
	q8 *gpu.QMatrix
	q4 *gpu.Q4Matrix
}

func uploadLinear(g *gpu.Device, l *linear) (*deviceLinear, uint64, error) {
	switch {
	case l.q != nil:
		q, err := g.UploadQ8(l.q)
		return &deviceLinear{q8: q}, uint64(len(l.q.Q)), err
	case l.q4 != nil:
		q, err := g.UploadQ4(l.q4)
		return &deviceLinear{q4: q}, uint64(len(l.q4.Q)), err
	}
	return nil, 0, fmt.Errorf("qwenimage: the device needs quantized weights; this model holds floats")
}

func (d *deviceLinear) matmul(x *gpu.Tensor) (*gpu.Tensor, error) {
	if d.q8 != nil {
		return d.q8.MatMul(x)
	}
	return d.q4.MatMul(x)
}

// bytes is what a block's feed-forward weighs at a given width.
func mlpBytes(b *Block) uint64 {
	n := uint64(0)
	for _, l := range []*linear{b.mlpProj, b.mlpGate, b.mlpOut} {
		switch {
		case l.q != nil:
			n += uint64(len(l.q.Q))
		case l.q4 != nil:
			n += uint64(len(l.q4.Q))
		}
	}
	return n
}

// UseGPU moves every block's feed-forward onto a device, if what that
// weighs fits the budget. It returns what the device is called and how
// many bytes of weights it took, or an error that leaves the model
// running on the CPU exactly as before.
func UseGPU(m *Transformer, budget uint64) (string, uint64, error) {
	if len(m.blocks) == 0 {
		return "", 0, fmt.Errorf("qwenimage: no blocks to move")
	}
	var want uint64
	for _, b := range m.blocks {
		want += mlpBytes(b)
	}
	if want == 0 {
		return "", 0, fmt.Errorf("qwenimage: the device needs quantized weights; this model holds floats")
	}
	if want > budget {
		return "", 0, fmt.Errorf("qwenimage: the feed-forward weights are %.1fGiB and the budget is %.1fGiB; "+
			"draw at four bits or raise the budget",
			float64(want)/(1<<30), float64(budget)/(1<<30))
	}
	g, err := gpu.Open(gpu.HighPerformance)
	if err != nil {
		return "", 0, err
	}
	// A single buffer is capped well below what the whole model weighs,
	// so a width whose one matrix is already too large fails here rather
	// than part way through.
	if lim := g.StorageLimit(); lim > 0 {
		if one := mlpBytes(m.blocks[0]) / 3; one > lim {
			g.Close()
			return "", 0, fmt.Errorf("qwenimage: one feed-forward weight is %dMiB and the device binds at most %dMiB",
				one>>20, lim>>20)
		}
	}
	// The output projection's input, the gated product, is made on the
	// device, so its rotation happens there: one product with H over
	// the rows viewed in groups, a few per cent of the projection.
	var rot *gpu.Tensor
	if r := m.blocks[0].mlpOut.rot; r > 0 {
		h := tensai.NewMatrix(r, r)
		for i := 0; i < r; i++ {
			h.Data[i*r+i] = 1
		}
		rotateRows(h, r) // the rows of I H are H
		var err error
		if rot, err = g.Upload(h.Tensor()); err != nil {
			g.Close()
			return "", 0, err
		}
	}
	for _, b := range m.blocks {
		w := &deviceWeights{rot: rot}
		for _, f := range []struct {
			dst **deviceLinear
			src *linear
		}{{&w.proj, b.mlpProj}, {&w.gate, b.mlpGate}, {&w.out, b.mlpOut}} {
			d, _, err := uploadLinear(g, f.src)
			if err != nil {
				g.Close()
				return "", 0, err
			}
			*f.dst = d
		}
		b.dev, b.g = w, g
	}
	m.dev = g
	return g.Name(), want, nil
}

// mlpOnDevice runs the feed-forward for one block on the device: the
// rows go up once, the three projections and the gate stay there, and
// only the result comes back.
func (b *Block) mlpOnDevice(dst, x *tensai.Matrix) (err error) {
	if err = b.g.BeginBatch(); err != nil {
		return err
	}
	defer func() {
		if e := b.g.Flush(); err == nil {
			err = e
		}
	}()
	// The gate and the projection were quantized rotated, and read the
	// same input; its rotation is done here before it goes up.
	if rot := b.mlpGate.rot; rot > 0 {
		r := rotatedCopy(x, rot)
		defer rotPool.Put(r)
		x = r
	}
	in := &tensai.Tensor{Shape: []int{x.Rows, x.Cols}, Data: x.Data}
	gx, err := b.devOf().Upload(in)
	if err != nil {
		return err
	}
	defer gx.Free()

	gate, err := b.dev.gate.matmul(gx)
	if err != nil {
		return err
	}
	defer gate.Free()
	up, err := b.dev.proj.matmul(gx)
	if err != nil {
		return err
	}
	defer up.Free()
	if err := gate.SiluMul(up); err != nil {
		return err
	}
	in2 := gate
	if b.dev.rot != nil {
		rows, cols, g := x.Rows, b.mlpOut.inputs(), b.mlpOut.rot
		grouped, err := gate.View(0, rows*cols/g, g)
		if err != nil {
			return err
		}
		r, err := grouped.MatMul(b.dev.rot)
		if err != nil {
			return err
		}
		defer r.Free()
		if in2, err = r.View(0, rows, cols); err != nil {
			return err
		}
	}
	out, err := b.dev.out.matmul(in2)
	if err != nil {
		return err
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		return err
	}
	copy(dst.Data, got.Data)
	return nil
}

// attentionOnDevice splits the block-causal mask into a causal text prefix
// and image queries that read every key. RoPE and head normalization have
// already run on the host. No additional model weights live on the device.
func (b *Block) attentionOnDevice(out, q, k, v *tensai.Matrix, l *Layout, s *Scratch) error {
	// Layout is public; preserve arbitrary masks through the host path.
	for i, limit := range l.KeyLimit {
		want := q.Rows
		if i < l.TextLen {
			want = i + 1
		}
		if limit != want {
			return attention(out, q, k, v, l.KeyLimit, s)
		}
	}
	// Every image query tile reads the same keys and values. Upload them
	// once per block; only the query tile and its result cross the bus in
	// the loop. Text queries borrow a view of just the causal prefix.
	gk, err := b.g.Upload(&tensai.Tensor{Shape: []int{k.Rows, k.Cols}, Data: k.Data})
	if err != nil {
		return err
	}
	defer gk.Free()
	gv, err := b.g.Upload(&tensai.Tensor{Shape: []int{v.Rows, v.Cols}, Data: v.Data})
	if err != nil {
		return err
	}
	defer gv.Free()
	for _, part := range []struct {
		lo, hi, keys int
		causal       bool
	}{
		{0, l.TextLen, l.TextLen, true},
		{l.TextLen, q.Rows, q.Rows, false},
	} {
		if part.lo == part.hi {
			continue
		}
		pk, err := gk.View(0, part.keys, k.Cols)
		if err != nil {
			return err
		}
		pv, err := gv.View(0, part.keys, v.Cols)
		if err != nil {
			return err
		}
		chunk := part.hi - part.lo
		if !part.causal {
			// The noncausal kernel materializes scores for every head.
			// Tile queries so larger images stay under the binding limit
			// and do not consume the remaining device memory in scores.
			limit := uint64(16 << 20)
			if storage := b.g.StorageLimit(); storage > 0 && storage < limit {
				limit = storage
			}
			chunk = min(chunk, max(1, int(limit/(uint64(ditHeads)*uint64(part.keys)*4))))
		}
		for lo := part.lo; lo < part.hi; lo += chunk {
			if err := b.attentionPart(out, q, pk, pv, lo, min(lo+chunk, part.hi), part.causal); err != nil {
				return err
			}
		}
	}
	return nil
}

func (b *Block) attentionPart(out, q *tensai.Matrix, gk, gv *gpu.Tensor, lo, hi int, causal bool) (err error) {
	if err = b.g.BeginBatch(); err != nil {
		return err
	}
	defer func() {
		if e := b.g.Flush(); err == nil {
			err = e
		}
	}()
	gq, err := b.g.Upload(&tensai.Tensor{Shape: []int{hi - lo, q.Cols}, Data: q.Data[lo*q.Cols : hi*q.Cols]})
	if err != nil {
		return err
	}
	defer gq.Free()
	var result *gpu.Tensor
	if causal {
		result, err = gq.CausalMultiHeadAttention(gk, gv, ditHeads)
	} else {
		result, err = gq.MultiHeadAttention(gk, gv, ditHeads)
	}
	if err != nil {
		return err
	}
	defer result.Free()
	got, err := result.Download()
	if err != nil {
		return err
	}
	copy(out.Data[lo*out.Cols:hi*out.Cols], got.Data)
	return nil
}

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
	q8   *gpu.QMatrix
	q4   *gpu.Q4Matrix
	lora *lowRank
	g    *gpu.Device
}

// UseGPUProjections streams one attention projection at a time. Its
// weights are released before the next upload, bounding extra residency.
func UseGPUProjections(m *Transformer, budget uint64) error {
	if m.dev == nil {
		return fmt.Errorf("qwenimage: enable the GPU before projections")
	}
	// Only a block whose feed-forward already went up can stream its
	// projections, since that is what put a device within its reach, and
	// what those blocks hold is what the spare budget is measured against.
	var peak uint64
	var on []*Block
	for _, b := range m.blocks {
		if b.dev == nil {
			continue
		}
		for _, l := range []*linear{b.toQ, b.toK, b.toV, b.toOut} {
			if l == nil || (l.q == nil && l.q4 == nil) {
				return fmt.Errorf("qwenimage: projections need quantized weights")
			}
			peak = max(peak, linearGPUBytes(l)+l.lora.bytes())
		}
		on = append(on, b)
	}
	if m.held+peak > budget {
		return fmt.Errorf("qwenimage: GPU projections need %d MiB of spare weight budget", (peak+(1<<20)-1)>>20)
	}
	for _, b := range on {
		b.streamProjections = true
	}
	return nil
}

func (b *Block) project(dst, x *tensai.Matrix, l *linear) error {
	if b.streamProjections && x.Rows >= 256 {
		return streamProjection(b.g, l, dst, x)
	}
	return l.apply(dst, x)
}

func uploadLinear(g *gpu.Device, l *linear) (*deviceLinear, uint64, error) {
	switch {
	case l.q != nil:
		q, err := g.UploadQ8(l.q)
		return &deviceLinear{q8: q, lora: l.lora, g: g}, linearGPUBytes(l), err
	case l.q4 != nil:
		q, err := g.UploadQ4(l.q4)
		return &deviceLinear{q4: q, lora: l.lora, g: g}, linearGPUBytes(l), err
	}
	return nil, 0, fmt.Errorf("qwenimage: the device needs quantized weights; this model holds floats")
}

func (d *deviceLinear) matmul(x *gpu.Tensor) (*gpu.Tensor, error) {
	var out *gpu.Tensor
	var err error
	if d.q8 != nil {
		out, err = d.q8.MatMul(x)
	} else {
		out, err = d.q4.MatMul(x)
	}
	if err != nil {
		return nil, err
	}
	if d.lora != nil {
		if err := d.lora.addGPU(d.g, out, x); err != nil {
			out.Free()
			return nil, err
		}
	}
	return out, nil
}

// linearGPUBytes counts the device's padded weights and scale tables.
func linearGPUBytes(l *linear) uint64 {
	switch {
	case l.q != nil:
		padded := uint64((l.q.Cols + 3) / 4 * 4)
		return uint64(l.q.Rows)*padded + padded*4
	case l.q4 != nil:
		padded := uint64((l.q4.Cols + 3) / 4 * 4)
		return uint64((l.q4.Rows+1)/2)*padded + uint64((l.q4.Rows+63)/64)*padded*4
	}
	return 0
}

func mlpBytes(b *Block) uint64 {
	return linearGPUBytes(b.mlpProj) + linearGPUBytes(b.mlpGate) + linearGPUBytes(b.mlpOut)
}

// UseGPU moves every block's feed-forward onto a device, if what that
// weighs fits the budget. It returns what the device is called and how
// many bytes of weights it took, or an error that leaves the model
// running on the CPU exactly as before.
func UseGPU(m *Transformer, budget uint64) (string, uint64, error) {
	if len(m.blocks) == 0 {
		return "", 0, fmt.Errorf("qwenimage: no blocks to move")
	}
	var want, adapterPeak uint64
	for _, b := range m.blocks {
		want += mlpBytes(b)
		adapterPeak = max(adapterPeak, b.mlpProj.lora.bytes()+b.mlpGate.lora.bytes()+b.mlpOut.lora.bytes())
	}
	if want == 0 {
		return "", 0, fmt.Errorf("qwenimage: the device needs quantized weights; this model holds floats")
	}
	// Blocks are independent -- Forward asks each one whether it has a
	// device -- so a budget that cannot take the whole feed-forward takes
	// as many blocks as it holds and leaves the rest on the CPU. Refusing
	// the device outright would be slower than either.
	room := budget
	if room < adapterPeak {
		room = 0
	} else {
		room -= adapterPeak
	}
	if mlpBytes(m.blocks[0]) > room {
		return "", 0, fmt.Errorf("qwenimage: one block's feed-forward and the adapter scratch need %.2fGiB and the budget is %.2fGiB; "+
			"draw at four bits or raise the budget",
			float64(mlpBytes(m.blocks[0])+adapterPeak)/(1<<30), float64(budget)/(1<<30))
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
	var held uint64
	for _, b := range m.blocks {
		if held+mlpBytes(b) > room {
			break
		}
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
		held += mlpBytes(b)
	}
	m.dev, m.held = g, held
	return g.Name(), held, nil
}

// BlocksOnDevice reports how many of a model's blocks hold their
// feed-forward on the device and how many there are, which differ when
// the budget only stretched to some of them.
func (m *Transformer) BlocksOnDevice() (on, total int) {
	for _, b := range m.blocks {
		if b.dev != nil {
			on++
		}
	}
	return on, len(m.blocks)
}

// mlpOnDevice runs the feed-forward for one block on the device: the
// rows go up once, the three projections and the gate stay there, and
// only the result comes back. The gated product is the widest buffer, so
// rows are tiled to keep it under the device's binding limit.
func (b *Block) mlpOnDevice(dst, x *tensai.Matrix) error {
	chunk := x.Rows
	if limit := b.devOf().StorageLimit(); limit > 0 {
		chunk = max(1, int(limit/(uint64(b.mlpOut.inputs())*4)))
	}
	for lo := 0; lo < x.Rows; lo += chunk {
		hi := min(lo+chunk, x.Rows)
		d := &tensai.Matrix{Rows: hi - lo, Cols: dst.Cols, Data: dst.Data[lo*dst.Cols : hi*dst.Cols]}
		s := &tensai.Matrix{Rows: hi - lo, Cols: x.Cols, Data: x.Data[lo*x.Cols : hi*x.Cols]}
		if err := b.mlpRows(d, s); err != nil {
			return err
		}
	}
	return nil
}

func (b *Block) mlpRows(dst, x *tensai.Matrix) (err error) {
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

// Includes weight packing, upload, input rotation, computation and readback.
func streamProjection(g *gpu.Device, l *linear, dst, x *tensai.Matrix) (err error) {
	w, _, err := uploadLinear(g, l)
	if err != nil {
		return err
	}
	if w.q8 != nil {
		defer w.q8.Free()
	} else {
		defer w.q4.Free()
	}
	if l.rot > 0 {
		r := rotatedCopy(x, l.rot)
		defer rotPool.Put(r)
		x = r
	}
	if err := g.BeginBatch(); err != nil {
		return err
	}
	defer func() {
		if e := g.Flush(); err == nil {
			err = e
		}
	}()
	in, err := g.Upload(x.Tensor())
	if err != nil {
		return err
	}
	defer in.Free()
	out, err := w.matmul(in)
	if err != nil {
		return err
	}
	defer out.Free()
	host, err := out.Download()
	if err != nil {
		return err
	}
	copy(dst.Data, host.Data)
	return nil
}

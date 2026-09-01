package autograd

import (
	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/internal/dims"
)

// Running a graph on a device means the values never come back between
// operations: a node keeps a device buffer and the host copy is only
// materialized when something reads Value or Grad. Every operation below
// mirrors one in ops.go, and each returns ok=false when the device cannot
// express the shape at hand, in which case the caller falls back to the CPU
// path with whatever has to come home first.
//
// The tape owns every device buffer an operation produces, and Reset frees
// them, so device mode requires a tape:
//
//	tape := autograd.NewTape()
//	tape.UseDevice(dev)
//	tape.Bind(params...)

// UseDevice runs the graphs built on this tape on a GPU. Parameters bound
// to the tape upload themselves the first time they are used and stay
// resident; intermediates never leave the device unless they are read.
// Passing nil returns the tape to the CPU.
func (t *Tape) UseDevice(d *gpu.Device) {
	t.dev = d
}

// Device returns the device this tape runs on, or nil.
func (t *Tape) Device() *gpu.Device {
	if t == nil {
		return nil
	}
	return t.dev
}

// openBatch records the dispatches of one step into a single submission.
// A driver charges per submission -- on dozen it is milliseconds -- and a
// training step runs dozens of small kernels, so batching them is the
// difference between residency paying and not. Download flushes on its
// own, which is why the flag is cleared whenever a value comes home.
func (t *Tape) openBatch() {
	if t == nil || t.dev == nil || t.batch {
		return
	}
	if err := t.dev.BeginBatch(); err == nil {
		t.batch = true
	}
}

// flushBatch submits whatever the batch has recorded.
func (t *Tape) flushBatch() {
	if t == nil || t.dev == nil || !t.batch {
		return
	}
	t.batch = false
	t.dev.Flush()
}

// batchClosed notes that something flushed the batch behind the tape's
// back, as Download does before it reads.
func (t *Tape) batchClosed() {
	if t != nil {
		t.batch = false
	}
}

// track hands a device buffer to the tape, which frees it on Reset.
func (t *Tape) track(g *gpu.Tensor) *gpu.Tensor {
	if t != nil && g != nil {
		t.devUsed = append(t.devUsed, g)
	}
	return g
}

// freeDevice releases every device buffer handed out since the last Reset.
// Parameter buffers are not in that list: they stay resident across steps.
func (t *Tape) freeDevice() {
	// Buffers may still be referenced by dispatches the batch has not
	// submitted yet, so the queue drains before they go back to the pool.
	t.flushBatch()
	for _, g := range t.devUsed {
		g.Free()
	}
	t.devUsed = t.devUsed[:0]
}

// device returns the device a node's graph runs on, or nil.
func (n *Node) device() *gpu.Device {
	if n == nil || n.tape == nil {
		return nil
	}
	return n.tape.dev
}

// resident returns the node's value on the device, uploading it the first
// time. A parameter keeps its buffer across steps -- it is the tape that
// would free an intermediate's -- so weights ride the bus once.
func (n *Node) resident(tp *Tape) (*gpu.Tensor, bool) {
	if n.dev != nil {
		return n.dev, true
	}
	if n.tape == nil {
		// A constant fed in per step -- Input(x) -- carries no tape of its
		// own; it joins the graph's.
		n.tape = tp
	}
	d := n.device()
	if d == nil || n.value == nil {
		return nil, false
	}
	g, err := d.Upload(n.value)
	if err != nil {
		return nil, false
	}
	n.dev = g
	if n.op != "" || !n.requiresGrad {
		// Intermediates and constants live for one step; parameters stay.
		n.tape.track(g)
	}
	return g, true
}

// residentGrad returns the node's gradient buffer on the device, allocating
// a zero one the first time.
func (n *Node) residentGrad() (*gpu.Tensor, bool) {
	if n.devGrad != nil {
		return n.devGrad, true
	}
	d := n.device()
	if d == nil {
		return nil, false
	}
	src := n.grad
	if src == nil {
		src = tensai.NewTensor(n.Shape()...)
	}
	g, err := d.Upload(src)
	if err != nil {
		return nil, false
	}
	n.devGrad = g
	n.grad = nil // the device copy is the authoritative one now
	if n.op != "" || !n.requiresGrad {
		n.tape.track(g)
	}
	return g, true
}

// syncValue brings a device-resident value home. Value calls it, so a
// caller that reads a node gets host memory whatever ran the graph.
func (n *Node) syncValue() {
	if n.value != nil || n.dev == nil {
		return
	}
	n.tape.batchClosed()
	v, err := n.dev.Download()
	if err != nil {
		panic("tensai: reading a device value: " + err.Error())
	}
	n.value = v
}

// syncGrad is syncValue for the gradient.
func (n *Node) syncGrad() {
	if n.grad != nil || n.devGrad == nil {
		return
	}
	n.tape.batchClosed()
	g, err := n.devGrad.Download()
	if err != nil {
		panic("tensai: reading a device gradient: " + err.Error())
	}
	n.grad = g
}

// devNode wraps a device buffer as a graph node on the same tape.
func devNode(op string, v *gpu.Tensor, shape []int, parents ...*Node) *Node {
	out := &Node{
		op:      op,
		parents: parents,
		tape:    tapeOf(parents...),
		dev:     v,
		shapeOf: append([]int(nil), shape...),
	}
	for _, p := range parents {
		if p != nil && p.requiresGrad {
			out.requiresGrad = true
		}
	}
	out.tape.track(v)
	return out
}

// devAccum adds g into n's device gradient, summing over a leading axis the
// operand was broadcast along. It reports false when the reduction is not
// one the device kernels can do, which sends the whole op back to the CPU.
func devAccum(n *Node, g *gpu.Tensor, gShape []int) bool {
	dst, ok := n.residentGrad()
	if !ok {
		return false
	}
	src := g
	cols, total := dims.Prod(n.Shape()), dims.Prod(gShape)
	switch {
	case cols == total:
		// Shapes agree; the add below is all that is needed.
	case cols != 0 && total > cols && total%cols == 0:
		// The operand was broadcast over a repeating leading block, so its
		// gradient is the column sums of that block.
		flat, err := g.View(0, total/cols, cols)
		if err != nil {
			return false
		}
		summed, err := flat.SumCols()
		if err != nil {
			return false
		}
		n.tape.track(summed)
		src = summed
	case total != 0 && cols%total == 0:
		// A smaller gradient (a scalar, say) repeats into the destination,
		// which the kernel does on its own.
	default:
		return false
	}
	// Accumulate in place: a gradient is added to many times, and a fresh
	// buffer per addition would double the traffic and the allocations.
	if err := dst.Add(src); err != nil {
		return false
	}
	n.grad = nil
	return true
}

// Resident reports whether the node's value currently lives on a device
// rather than in host memory. Reading Value brings it home.
func (n *Node) Resident() bool { return n.dev != nil }

// devMatMul runs a product on the device, keeping the result there. The
// backward pass stays resident too: both gradients are transposed products
// the device has kernels for.
func devMatMul(n, o *Node, mode gemmMode) (*Node, bool) {
	if tapeOf(n, o).Device() == nil {
		return nil, false
	}
	tapeOf(n, o).openBatch()
	tp := tapeOf(n, o)
	ga, ok := n.resident(tp)
	if !ok {
		return nil, false
	}
	gb, ok := o.resident(tp)
	if !ok {
		return nil, false
	}
	gc, err := devProduct(ga, gb, mode)
	if err != nil {
		return nil, false
	}
	out := devNode("matmul", gc, gc.Shape(), n, o)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device matmul")
		}
		if n.requiresGrad {
			d, err := devProduct(grad, gb, gemmNT)
			if err != nil {
				panic("tensai: device matmul backward: " + err.Error())
			}
			out.tape.track(d)
			if !devAccum(n, d, d.Shape()) {
				panic("tensai: device matmul cannot reduce its input gradient")
			}
		}
		if o.requiresGrad {
			d, err := devProduct(ga, grad, gemmTN)
			if err != nil {
				panic("tensai: device matmul backward: " + err.Error())
			}
			out.tape.track(d)
			if !devAccum(o, d, d.Shape()) {
				panic("tensai: device matmul cannot reduce its weight gradient")
			}
		}
	}), true
}

// devProduct runs one of the three product modes on the device.
func devProduct(a, b *gpu.Tensor, mode gemmMode) (*gpu.Tensor, error) {
	switch mode {
	case gemmTN:
		return a.MatMulTN(b)
	case gemmNT:
		return a.MatMulT(b)
	default:
		return a.MatMul(b)
	}
}

// devBinary runs an element-wise op on the device. The kernels broadcast an
// operand that repeats into the output, which covers a bias row or a
// per-feature scale; anything else goes back to the CPU.
func devBinary(op gpu.BinOp, n, o *Node) (*Node, bool) {
	if tapeOf(n, o).Device() == nil {
		return nil, false
	}
	tapeOf(n, o).openBatch()
	shape, ok := devBinShape(n.Shape(), o.Shape())
	if !ok {
		return nil, false
	}
	tp := tapeOf(n, o)
	ga, ok := n.resident(tp)
	if !ok {
		return nil, false
	}
	gb, ok := o.resident(tp)
	if !ok {
		return nil, false
	}
	if dims.Prod(o.Shape()) > dims.Prod(n.Shape()) {
		return nil, false // only the right operand may repeat
	}
	gc, err := ga.Binary(op, gb)
	if err != nil {
		return nil, false
	}
	out := devNode(opName(op), gc, shape, n, o)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device op")
		}
		switch op {
		case gpu.OpAdd, gpu.OpSub:
			if n.requiresGrad && !devAccum(n, grad, out.Shape()) {
				panic("tensai: device add cannot reduce its gradient")
			}
			if o.requiresGrad {
				g := grad
				if op == gpu.OpSub {
					neg, err := devScaleBuf(out.tape, grad, -1)
					if err != nil {
						panic("tensai: device sub backward: " + err.Error())
					}
					g = neg
				}
				if !devAccum(o, g, out.Shape()) {
					panic("tensai: device sub cannot reduce its gradient")
				}
			}
		default: // OpMul
			if n.requiresGrad {
				d, err := grad.Binary(gpu.OpMul, gb)
				if err != nil {
					panic("tensai: device mul backward: " + err.Error())
				}
				out.tape.track(d)
				if !devAccum(n, d, out.Shape()) {
					panic("tensai: device mul cannot reduce its gradient")
				}
			}
			if o.requiresGrad {
				d, err := grad.Binary(gpu.OpMul, ga)
				if err != nil {
					panic("tensai: device mul backward: " + err.Error())
				}
				out.tape.track(d)
				if !devAccum(o, d, out.Shape()) {
					panic("tensai: device mul cannot reduce its gradient")
				}
			}
		}
	}), true
}

// devScaleBuf returns s*g as a new device buffer.
func devScaleBuf(t *Tape, g *gpu.Tensor, s tensai.Float) (*gpu.Tensor, error) {
	one, err := t.dev.Upload(scalarTensor(s))
	if err != nil {
		return nil, err
	}
	t.track(one)
	out, err := g.Binary(gpu.OpMul, one)
	if err != nil {
		return nil, err
	}
	t.track(out)
	return out, nil
}

// scalarTensor is a single-element tensor, which the kernels broadcast over
// a whole buffer.
func scalarTensor(v tensai.Float) *tensai.Tensor {
	t := tensai.NewTensor(1)
	t.Data[0] = v
	return t
}

// devBinShape returns the output shape of an element-wise op the device can
// run: the shapes must match, or the second must repeat into the first.
func devBinShape(a, b []int) ([]int, bool) {
	na, nb := dims.Prod(a), dims.Prod(b)
	if dims.Same(a, b) {
		return a, true
	}
	if nb == 0 || na%nb != 0 {
		return nil, false
	}
	// Only a trailing block that repeats is expressible; check that b is
	// the tail of a.
	for i := 1; i <= len(b); i++ {
		if i > len(a) || a[len(a)-i] != b[len(b)-i] {
			if b[len(b)-i] != 1 {
				return nil, false
			}
		}
	}
	return a, true
}

// opName labels a device op in ToDot the way its CPU twin does.
func opName(op gpu.BinOp) string {
	switch op {
	case gpu.OpAdd:
		return "add"
	case gpu.OpSub:
		return "sub"
	case gpu.OpMul:
		return "mul"
	default:
		return "div"
	}
}

// devActivate runs an activation and its gradient on the device.
func devActivate(act gpu.Act, name string, n *Node) (*Node, bool) {
	if n.device() == nil {
		return nil, false
	}
	n.tape.openBatch()
	ga, ok := n.resident(n.tape)
	if !ok {
		return nil, false
	}
	gv, err := ga.Activate(act)
	if err != nil {
		return nil, false
	}
	out := devNode(name, gv, n.Shape(), n)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device activation")
		}
		d, err := ga.ActivateGrad(act, grad)
		if err != nil {
			panic("tensai: device activation backward: " + err.Error())
		}
		out.tape.track(d)
		if !devAccum(n, d, n.Shape()) {
			panic("tensai: device activation cannot reduce its gradient")
		}
	}), true
}

// devSum reduces a resident tensor to a scalar without bringing it home:
// the device sums the columns of a wide view, and only that row -- a few
// kilobytes -- crosses the bus. The node itself is an ordinary host-valued
// scalar, so the loss arithmetic above it stays on the CPU where it costs
// nothing, while the gradient it pushes back is a device broadcast.
func devSum(n *Node) (*Node, bool) {
	d := n.device()
	if d == nil {
		return nil, false
	}
	size := dims.Prod(n.Shape())
	cols := devSumWidth(size)
	if cols == 0 {
		return nil, false
	}
	g, ok := n.resident(n.tape)
	if !ok {
		return nil, false
	}
	wide, err := g.View(0, size/cols, cols)
	if err != nil {
		return nil, false
	}
	colSums, err := wide.SumCols()
	if err != nil {
		return nil, false
	}
	n.tape.track(colSums)
	n.tape.batchClosed()
	row, err := colSums.Download()
	if err != nil {
		return nil, false
	}
	total := tensai.NewTensor(1, 1)
	for _, v := range row.Data {
		total.Data[0] += v
	}
	out := newNode("sum", total, n)
	return out.withBack(func(out *Node) {
		// Every element contributed once, so the input gradient is the
		// output gradient broadcast -- a single-element device buffer the
		// kernels repeat over the whole tensor.
		one, err := d.Upload(scalarTensor(out.Grad().Data[0]))
		if err != nil {
			panic("tensai: device sum backward: " + err.Error())
		}
		n.tape.track(one)
		if !devAccum(n, one, []int{1}) {
			panic("tensai: device sum cannot spread its gradient")
		}
	}), true
}

// devSumWidth picks how wide to view a buffer for the column-sum kernel:
// wide enough to keep the device busy, and a divisor of the element count.
func devSumWidth(size int) int {
	for _, w := range []int{4096, 1024, 256, 64} {
		if size >= w && size%w == 0 {
			return w
		}
	}
	return 0
}

// devLayerNorm normalizes on the device, keeping the gradients of the input
// and of the affine parameters there too. A nil gain or bias becomes a
// constant buffer for the kernel, and collects no gradient.
func devLayerNorm(n, gain, bias *Node, eps tensai.Float) (*Node, bool) {
	tp := tapeOf(n, gain, bias)
	if tp.Device() == nil {
		return nil, false
	}
	shape := n.Shape()
	cols := shape[len(shape)-1]
	tp.openBatch()
	gx, ok := n.resident(tp)
	if !ok {
		return nil, false
	}
	gg, ok := devAffine(tp, gain, cols, 1)
	if !ok {
		return nil, false
	}
	gb, ok := devAffine(tp, bias, cols, 0)
	if !ok {
		return nil, false
	}
	gv, err := gx.LayerNorm(gg, gb, eps)
	if err != nil {
		return nil, false
	}
	parents := []*Node{n}
	for _, p := range []*Node{gain, bias} {
		if p != nil {
			parents = append(parents, p)
		}
	}
	out := devNode("layernorm", gv, shape, parents...)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device layernorm")
		}
		if n.requiresGrad {
			dx, err := gx.LayerNormGrad(grad, gg, eps)
			if err != nil {
				panic("tensai: device layernorm backward: " + err.Error())
			}
			out.tape.track(dx)
			if !devAccum(n, dx, shape) {
				panic("tensai: device layernorm cannot reduce its input gradient")
			}
		}
		if gain != nil && gain.requiresGrad {
			// The gain's gradient is the column sums of grad * xhat.
			xhat, err := gx.LayerNormXhat(eps)
			if err != nil {
				panic("tensai: device layernorm backward: " + err.Error())
			}
			out.tape.track(xhat)
			prod, err := grad.Binary(gpu.OpMul, xhat)
			if err != nil {
				panic("tensai: device layernorm backward: " + err.Error())
			}
			out.tape.track(prod)
			if !devAccum(gain, prod, shape) {
				panic("tensai: device layernorm cannot reduce its gain gradient")
			}
		}
		if bias != nil && bias.requiresGrad && !devAccum(bias, grad, shape) {
			panic("tensai: device layernorm cannot reduce its bias gradient")
		}
	}), true
}

// devAffine returns a LayerNorm gain or bias on the device, standing in a
// constant buffer when the node is nil.
func devAffine(t *Tape, p *Node, cols int, fill tensai.Float) (*gpu.Tensor, bool) {
	if p != nil {
		return p.resident(t)
	}
	c := tensai.NewTensor(cols)
	for i := range c.Data {
		c.Data[i] = fill
	}
	g, err := t.dev.Upload(c)
	if err != nil {
		return nil, false
	}
	t.track(g)
	return g, true
}

// devSoftmax normalizes the last axis on the device; the backward pass is
// the softmax Jacobian applied in one kernel.
func devSoftmax(n *Node) (*Node, bool) {
	if n.device() == nil {
		return nil, false
	}
	n.tape.openBatch()
	gx, ok := n.resident(n.tape)
	if !ok {
		return nil, false
	}
	gy, err := gx.Softmax()
	if err != nil {
		return nil, false
	}
	out := devNode("softmax", gy, n.Shape(), n)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device softmax")
		}
		dx, err := gy.SoftmaxGrad(grad)
		if err != nil {
			panic("tensai: device softmax backward: " + err.Error())
		}
		out.tape.track(dx)
		if !devAccum(n, dx, n.Shape()) {
			panic("tensai: device softmax cannot reduce its gradient")
		}
	}), true
}

// devTranspose permutes axes on the device; the gradient permutes back.
func devTranspose(n *Node, perm []int) (*Node, bool) {
	if n.device() == nil {
		return nil, false
	}
	n.tape.openBatch()
	gx, ok := n.resident(n.tape)
	if !ok {
		return nil, false
	}
	gv, err := gx.Permute(perm...)
	if err != nil {
		return nil, false
	}
	inv := make([]int, len(perm))
	for i, p := range perm {
		inv[p] = i
	}
	out := devNode("transpose", gv, gv.Shape(), n)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device transpose")
		}
		d, err := grad.Permute(inv...)
		if err != nil {
			panic("tensai: device transpose backward: " + err.Error())
		}
		out.tape.track(d)
		if !devAccum(n, d, n.Shape()) {
			panic("tensai: device transpose cannot reduce its gradient")
		}
	}), true
}

// devReshape relabels a resident buffer: the data does not move, so this is
// a view on both the way out and the way back.
func devReshape(n *Node, shape []int) (*Node, bool) {
	if n.device() == nil {
		return nil, false
	}
	gx, ok := n.resident(n.tape)
	if !ok {
		return nil, false
	}
	gv, err := gx.View(0, shape...)
	if err != nil {
		return nil, false
	}
	out := devNode("reshape", gv, shape, n)
	return out.withBack(func(out *Node) {
		grad, ok := out.residentGrad()
		if !ok {
			panic("tensai: no device gradient for a device reshape")
		}
		flat, err := grad.View(0, n.Shape()...)
		if err != nil {
			panic("tensai: device reshape backward: " + err.Error())
		}
		out.tape.track(flat)
		if !devAccum(n, flat, n.Shape()) {
			panic("tensai: device reshape cannot reduce its gradient")
		}
	}), true
}

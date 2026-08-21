//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin))

package tensai

// GPUTensor: tensors resident in GPU memory, so a weight is uploaded once
// and reused across calls instead of riding the bus on every MatMul. This
// file is shared between the v22 (wgpu.go) and v24+ (wgpu24.go) bindings —
// it only touches primitives whose shapes agree across the two generations
// (newBuffer, makeBindGroup, mapRead, and the fn* pointers with identical
// signatures).

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// GPUTensor is a tensor whose data lives in GPU memory. Create one with
// GPU.Upload or as the result of GPUTensor.MatMul, read it back with
// Download, and release it with Free — GPU memory is not garbage
// collected.
type GPUTensor struct {
	g     *GPU
	buf   uintptr
	shape []int
	freed bool
}

// Every GPUTensor buffer is storage-usable (matmul input and output),
// copyable in (Upload) and out (Download).
const gpuTensorUsage = wgpuBufferUsageStorage | wgpuBufferUsageCopySrc | wgpuBufferUsageCopyDst

// Shape returns a copy of the tensor's shape.
func (t *GPUTensor) Shape() []int { return append([]int(nil), t.shape...) }

// Size returns the total number of elements.
func (t *GPUTensor) Size() int { return prodDims(t.shape) }

// Upload copies a tensor into GPU memory. The returned GPUTensor is
// independent of t.
func (g *GPU) Upload(t *Tensor) (*GPUTensor, error) {
	if len(t.Data) == 0 {
		return nil, errors.New("tensai: cannot upload an empty tensor")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	buf := g.newBuffer(gpuTensorUsage, uint64(len(t.Data))*4)
	if buf == 0 {
		return nil, errors.New("tensai: gpu buffer allocation failed")
	}
	fnQueueWriteBuffer(g.queue, buf, 0, unsafe.Pointer(&t.Data[0]), uintptr(len(t.Data))*4)
	runtime.KeepAlive(t)
	return &GPUTensor{g: g, buf: buf, shape: append([]int(nil), t.Shape...)}, nil
}

// Download copies the tensor back into host memory.
func (t *GPUTensor) Download() (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	out := NewTensor(t.shape...)
	bytes := uint64(len(out.Data)) * 4
	staging := g.newBuffer(wgpuBufferUsageMapRead|wgpuBufferUsageCopyDst, bytes)
	defer fnBufferRelease(staging)

	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	fnEncoderCopyBuffer(encoder, t.buf, 0, staging, 0, bytes)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)

	src, err := g.mapRead(staging, bytes)
	if err != nil {
		return nil, err
	}
	copy(out.Data, unsafe.Slice((*Float)(src), len(out.Data)))
	fnBufferUnmap(staging)
	return out, nil
}

// Free releases the GPU buffer. The tensor must not be used afterwards;
// calling Free again is a no-op.
func (t *GPUTensor) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if t.freed {
		return
	}
	t.freed = true
	fnBufferRelease(t.buf)
}

// MatMul multiplies two GPU-resident tensors with the same shape and
// broadcasting semantics as the package-level MatMul, returning a new
// GPU-resident tensor without any host transfer. Chain calls freely; only
// Download moves data back.
func (t *GPUTensor) MatMul(o *GPUTensor) (*GPUTensor, error) {
	if t.freed || o.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if t.g != o.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	na, nb := len(t.shape), len(o.shape)
	if na < 2 || nb < 2 {
		return nil, fmt.Errorf("tensai: matmul needs at least 2 axes: %v * %v", t.shape, o.shape)
	}
	m, k := t.shape[na-2], t.shape[na-1]
	if o.shape[nb-2] != k {
		return nil, fmt.Errorf("tensai: matmul shape mismatch: %v * %v", t.shape, o.shape)
	}
	n := o.shape[nb-1]
	batch, err := broadcastShapes(t.shape[:na-2], o.shape[:nb-2])
	if err != nil {
		return nil, err
	}
	batches := prodDims(batch)
	if batches > 65535 {
		return nil, fmt.Errorf("tensai: gpu matmul batch count %d exceeds 65535", batches)
	}
	outShape := append(append([]int(nil), batch...), m, n)

	// Element offsets of each batch's matrix in t and o.
	as := broadcastStrides(t.shape[:na-2], batch)
	bs := broadcastStrides(o.shape[:nb-2], batch)
	offs := make([]uint32, 2*batches)
	for bi := 0; bi < batches; bi++ {
		offA, offB := 0, 0
		for d, rem := len(batch)-1, bi; d >= 0; d-- {
			i := rem % batch[d]
			rem /= batch[d]
			offA += i * as[d]
			offB += i * bs[d]
		}
		offs[2*bi] = uint32(offA * m * k)
		offs[2*bi+1] = uint32(offB * k * n)
	}
	params := [4]uint32{uint32(m), uint32(k), uint32(n), uint32(batches)}

	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(prodDims(outShape)) * 4
	bufParams := g.newBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufOffs := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.newBuffer(gpuTensorUsage, outBytes)
	defer fnBufferRelease(bufParams)
	defer fnBufferRelease(bufOffs)

	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	entries := [5]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, size: 16},
		{binding: 1, buffer: t.buf, size: uint64(t.Size()) * 4},
		{binding: 2, buffer: o.buf, size: uint64(o.Size()) * 4},
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 4, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.makeBindGroup(entries[:])
	runtime.KeepAlive(&entries)
	defer fnBindGroupRelease(bindGroup)

	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	pass := fnEncoderBeginComputePass(encoder, nil)
	fnPassSetPipeline(pass, g.pipeline)
	fnPassSetBindGroup(pass, 0, bindGroup, 0, nil)
	fnPassDispatch(pass, uint32((n+7)/8), uint32((m+7)/8), uint32(batches))
	fnPassEnd(pass)
	fnPassRelease(pass)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)

	// Validation errors surface through the uncaptured-error callback,
	// which wgpu-native fires during submission; pump once without
	// blocking on the actual compute work.
	fnDevicePoll(g.device, 0, nil)
	if uncapturedCB != "" {
		fnBufferRelease(bufOut)
		return nil, fmt.Errorf("tensai: gpu matmul failed: %s", uncapturedCB)
	}
	return &GPUTensor{g: g, buf: bufOut, shape: outShape}, nil
}

// MatMul is the GPU version of the package-level MatMul: identical shape
// and broadcasting semantics, executed as one compute dispatch. Inputs and
// the result live in host memory; each call uploads the operands and reads
// the product back, so it only pays off for large products — keep operands
// resident with Upload / GPUTensor.MatMul to amortize the transfers.
func (g *GPU) MatMul(a, b *Tensor) (*Tensor, error) {
	ga, err := g.Upload(a)
	if err != nil {
		return nil, err
	}
	defer ga.Free()
	gb, err := g.Upload(b)
	if err != nil {
		return nil, err
	}
	defer gb.Free()
	gc, err := ga.MatMul(gb)
	if err != nil {
		return nil, err
	}
	defer gc.Free()
	return gc.Download()
}

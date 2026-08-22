//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package tensai

// GPUTensor: tensors resident in GPU memory, so a weight is uploaded once
// and reused across calls instead of riding the bus on every MatMul, and
// chains of operations keep their intermediates on the device. This file is
// shared between the v22 (wgpu.go) and v24+ (wgpu24.go) bindings — it only
// touches primitives whose shapes agree across the two generations
// (newBuffer, makeBindGroup, makePipeline, mapRead, and the fn* pointers
// with identical signatures).

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// gpuBlock is the square output-block edge one matmul workgroup produces;
// the WGSL constants and the dispatches below must agree with it.
const gpuBlock = 64

// gpuKernelWG is the workgroup width of the 1-D kernels (scale, softmax).
const gpuKernelWG = 256

// matmulWGSL holds every compute kernel as one module. Each entry point
// declares its own bindings on distinct slots, so the auto layouts stay
// independent.
//
// main / matmul_t multiply one (m x k) x (k x n) — respectively (n x k),
// read transposed — pair per z-slice; per-batch input offsets come from a
// lookup table so broadcast batches share data. Each 16x16 workgroup
// produces a 64x64 output block: it streams 64x16 tiles of a and 16x64
// tiles of b through workgroup memory (shared-memory tiling), and each
// thread accumulates a 4x4 register tile, so a value read from workgroup
// memory feeds four multiplies instead of one (register tiling). The tile
// loop count is uniform across the workgroup, so the barriers inside are
// in uniform control flow; out-of-range lanes load zeros and only the
// final store is guarded.
//
// scale_ip multiplies a buffer by a scalar in place. softmax_last runs one
// workgroup per row: a strided max reduction, then exp and a sum
// reduction, then the divide — the numerically stable softmax over the
// last axis.
const matmulWGSL = `
struct Params {
    m: u32, k: u32, n: u32, batches: u32,
    lda: u32, ldb: u32, ldc: u32, pad: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> offs: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;

const TILE = 16u; // k-depth of one shared tile
const BLK = 64u;  // output block edge per workgroup
const TT = 4u;    // per-thread register tile edge

var<workgroup> tileA: array<f32, 1024>; // BLK x TILE
var<workgroup> tileB: array<f32, 1024>; // TILE x BLK

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc: array<f32, 16>;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        // 256 threads cooperatively load the 1024-element a and b tiles,
        // four elements each.
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = li * 4u + i;
            let ar = idx / TILE;
            let ac = idx % TILE;
            let gr = wid.y * BLK + ar;
            let gc = t * TILE + ac;
            if (gr < p.m && gc < p.k) {
                tileA[idx] = a[offA + gr * p.lda + gc];
            } else {
                tileA[idx] = 0.0;
            }
            let br = idx / BLK;
            let bc = idx % BLK;
            let gkr = t * TILE + br;
            let gbc = wid.x * BLK + bc;
            if (gkr < p.k && gbc < p.n) {
                tileB[idx] = b[offB + gkr * p.ldb + gbc];
            } else {
                tileB[idx] = 0.0;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            var af: array<f32, 4>;
            var bf: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                af[i] = tileA[(lid.y * TT + i) * TILE + kk];
                bf[i] = tileB[kk * BLK + lid.x * TT + i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + af[i] * bf[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let r = rowBase + i;
            let c = colBase + j;
            if (r < p.m && c < p.n) {
                outv[offC + r * p.ldc + c] = acc[i * 4u + j];
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn matmul_t(@builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc: array<f32, 16>;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = li * 4u + i;
            let ar = idx / TILE;
            let ac = idx % TILE;
            let gr = wid.y * BLK + ar;
            let gc = t * TILE + ac;
            if (gr < p.m && gc < p.k) {
                tileA[idx] = a[offA + gr * p.lda + gc];
            } else {
                tileA[idx] = 0.0;
            }
            // b holds n rows of length k (row stride ldb); transpose the
            // tile while loading so the inner loop below is shared.
            let br = idx / BLK;
            let bc = idx % BLK;
            let gkr = t * TILE + br;
            let gbc = wid.x * BLK + bc;
            if (gkr < p.k && gbc < p.n) {
                tileB[idx] = b[offB + gbc * p.ldb + gkr];
            } else {
                tileB[idx] = 0.0;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            var af: array<f32, 4>;
            var bf: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                af[i] = tileA[(lid.y * TT + i) * TILE + kk];
                bf[i] = tileB[kk * BLK + lid.x * TT + i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + af[i] * bf[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let r = rowBase + i;
            let c = colBase + j;
            if (r < p.m && c < p.n) {
                outv[offC + r * p.ldc + c] = acc[i * 4u + j];
            }
        }
    }
}

// matmul_s / matmul_ts are the plain 16x16 shared-memory variants (one
// output per thread). They win on small or skinny products, where 64x64
// blocks would leave most of the workgroup idle; stridedMatMul picks the
// kernel by shape.
var<workgroup> tileSA: array<f32, 256>;
var<workgroup> tileSB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul_s(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let batch = gid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    var sum = 0.0;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ak = t * TILE + lid.x;
        if (row < p.m && ak < p.k) {
            tileSA[lid.y * TILE + lid.x] = a[offA + row * p.lda + ak];
        } else {
            tileSA[lid.y * TILE + lid.x] = 0.0;
        }
        let bk = t * TILE + lid.y;
        if (bk < p.k && col < p.n) {
            tileSB[lid.y * TILE + lid.x] = b[offB + bk * p.ldb + col];
        } else {
            tileSB[lid.y * TILE + lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            sum = sum + tileSA[lid.y * TILE + i] * tileSB[i * TILE + lid.x];
        }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) {
        outv[offC + row * p.ldc + col] = sum;
    }
}

@compute @workgroup_size(16, 16, 1)
fn matmul_ts(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let batch = gid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    var sum = 0.0;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ak = t * TILE + lid.x;
        if (row < p.m && ak < p.k) {
            tileSA[lid.y * TILE + lid.x] = a[offA + row * p.lda + ak];
        } else {
            tileSA[lid.y * TILE + lid.x] = 0.0;
        }
        let bcol = wid.x * TILE + lid.y;
        let bk = t * TILE + lid.x;
        if (bcol < p.n && bk < p.k) {
            tileSB[lid.x * TILE + lid.y] = b[offB + bcol * p.ldb + bk];
        } else {
            tileSB[lid.x * TILE + lid.y] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            sum = sum + tileSA[lid.y * TILE + i] * tileSB[i * TILE + lid.x];
        }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) {
        outv[offC + row * p.ldc + col] = sum;
    }
}

struct ScaleParams { count: u32, s: f32 }
@group(0) @binding(5) var<uniform> scp: ScaleParams;
@group(0) @binding(6) var<storage, read_write> sbuf: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn scale_ip(@builtin(workgroup_id) wg: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    // Workgroups are laid out on a 2-D grid to escape the 65535 per-axis
    // dispatch limit.
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < scp.count) {
        sbuf[idx] = sbuf[idx] * scp.s;
    }
}

struct SoftParams { rows: u32, cols: u32, qmod: u32, off: u32 }
@group(0) @binding(7) var<uniform> sp: SoftParams;
@group(0) @binding(8) var<storage, read> sx: array<f32>;
@group(0) @binding(9) var<storage, read_write> sy: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax_last(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    // One workgroup per row, on a 2-D grid to escape the 65535 per-axis
    // dispatch limit. row is workgroup-uniform, so returning here keeps
    // the barriers below in uniform control flow.
    let row = wid.y * nwg.x + wid.x;
    if (row >= sp.rows) {
        return;
    }
    let base = row * sp.cols;
    // Causal masking: with qmod > 0, row's query index is row % qmod and
    // it may attend to the first (query index + off + 1) columns; the rest
    // get probability zero. limit is workgroup-uniform, so the barriers
    // below stay in uniform control flow.
    var limit = sp.cols;
    if (sp.qmod > 0u) {
        limit = min(sp.cols, (row % sp.qmod) + sp.off + 1u);
    }
    var m = -3.40282e38;
    for (var i = lid.x; i < limit; i = i + 256u) {
        m = max(m, sx[base + i]);
    }
    red[lid.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
        workgroupBarrier();
    }
    let rowMax = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var i = lid.x; i < limit; i = i + 256u) {
        let e = exp(sx[base + i] - rowMax);
        sy[base + i] = e;
        sum = sum + e;
    }
    for (var i = limit + lid.x; i < sp.cols; i = i + 256u) {
        sy[base + i] = 0.0;
    }
    red[lid.x] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
        workgroupBarrier();
    }
    let total = red[0];
    for (var i = lid.x; i < limit; i = i + 256u) {
        sy[base + i] = sy[base + i] / total;
    }
}
`

// gpuPipelines holds one compute pipeline (and its auto bind-group layout)
// per kernel entry point. It is embedded in each binding generation's GPU
// struct.
type gpuPipelines struct {
	matmul, matmulT, matmulS, matmulTS             uintptr
	scale, softmax                                 uintptr
	layMatmul, layMatmulT, layMatmulS, layMatmulTS uintptr
	layScale, laySoftmax                           uintptr
}

// initPipelines compiles every kernel from g.module; the caller holds
// wgpuMu.
func (g *GPU) initPipelines() error {
	for _, x := range []struct {
		pipe, lay *uintptr
		entry     string
	}{
		{&g.pipes.matmul, &g.pipes.layMatmul, "main"},
		{&g.pipes.matmulT, &g.pipes.layMatmulT, "matmul_t"},
		{&g.pipes.matmulS, &g.pipes.layMatmulS, "matmul_s"},
		{&g.pipes.matmulTS, &g.pipes.layMatmulTS, "matmul_ts"},
		{&g.pipes.scale, &g.pipes.layScale, "scale_ip"},
		{&g.pipes.softmax, &g.pipes.laySoftmax, "softmax_last"},
	} {
		*x.pipe = g.makePipeline(x.entry)
		if *x.pipe == 0 || uncapturedCB != "" {
			return fmt.Errorf("tensai: wgpu pipeline %q creation failed: %s", x.entry, uncapturedCB)
		}
		*x.lay = fnPipelineGetLayout(*x.pipe, 0)
	}
	return nil
}

// releasePipelines drops every pipeline and layout; the caller holds
// wgpuMu.
func (g *GPU) releasePipelines() {
	for _, h := range []uintptr{
		g.pipes.layMatmul, g.pipes.layMatmulT, g.pipes.layMatmulS,
		g.pipes.layMatmulTS, g.pipes.layScale, g.pipes.laySoftmax,
	} {
		if h != 0 {
			fnLayoutRelease(h)
		}
	}
	for _, h := range []uintptr{
		g.pipes.matmul, g.pipes.matmulT, g.pipes.matmulS,
		g.pipes.matmulTS, g.pipes.scale, g.pipes.softmax,
	} {
		if h != 0 {
			fnPipelineRelease(h)
		}
	}
}

// dispatch runs one compute pass and pumps the error callback once; the
// caller holds wgpuMu.
func (g *GPU) dispatch(pipe, bindGroup uintptr, x, y, z uint32) error {
	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	pass := fnEncoderBeginComputePass(encoder, nil)
	fnPassSetPipeline(pass, pipe)
	fnPassSetBindGroup(pass, 0, bindGroup, 0, nil)
	fnPassDispatch(pass, x, y, z)
	fnPassEnd(pass)
	fnPassRelease(pass)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)
	fnDevicePoll(g.device, 0, nil)
	if uncapturedCB != "" {
		return fmt.Errorf("tensai: gpu dispatch failed: %s", uncapturedCB)
	}
	return nil
}

// GPUTensor is a tensor whose data lives in GPU memory. Create one with
// GPU.Upload or as the result of a GPUTensor operation, read it back with
// Download, and release it with Free — GPU memory is not garbage
// collected.
type GPUTensor struct {
	g     *GPU
	buf   uintptr
	shape []int
	freed bool
}

// Every GPUTensor buffer is storage-usable (kernel input and output),
// copyable in (Upload) and out (Download).
const gpuTensorUsage = wgpuBufferUsageStorage | wgpuBufferUsageCopySrc | wgpuBufferUsageCopyDst

// gpuReadbackBuffer is the reusable MapRead staging buffer. Downloads are
// serialized by GPU.mu and wgpuMu, and mapRead waits for the copy to finish,
// so the buffer is idle again before it is returned to this slot.
type gpuReadbackBuffer struct {
	buf  uintptr
	size uint64
}

// takeReadback returns an unmapped staging buffer at least bytes large. The
// single retained buffer grows to the largest download seen by this GPU.
// The caller holds GPU.mu and wgpuMu.
func (g *GPU) takeReadback(bytes uint64) (uintptr, uint64) {
	if g.readback.buf != 0 && g.readback.size >= bytes {
		buf, capacity := g.readback.buf, g.readback.size
		g.readback = gpuReadbackBuffer{}
		return buf, capacity
	}
	if g.readback.buf != 0 {
		fnBufferRelease(g.readback.buf)
		g.readback = gpuReadbackBuffer{}
	}
	return g.newBuffer(wgpuBufferUsageMapRead|wgpuBufferUsageCopyDst, bytes), bytes
}

// putReadback retains a successfully unmapped staging buffer for the next
// download. The caller holds GPU.mu and wgpuMu.
func (g *GPU) putReadback(buf uintptr, size uint64) {
	if g.readback.buf != 0 {
		fnBufferRelease(g.readback.buf)
	}
	g.readback = gpuReadbackBuffer{buf: buf, size: size}
}

// releaseReadback drops the retained staging buffer during GPU.Close. The
// caller holds wgpuMu.
func (g *GPU) releaseReadback() {
	if g.readback.buf != 0 {
		fnBufferRelease(g.readback.buf)
	}
	g.readback = gpuReadbackBuffer{}
}

// StorageLimit reports how many bytes a single GPU buffer may hold under
// the device limits negotiated at OpenGPU time (0 when unknown). Tensor
// operations return an error instead of touching the driver when a buffer
// would exceed it.
func (g *GPU) StorageLimit() uint64 { return g.maxStorage }

func (g *GPU) checkSize(bytes uint64) error {
	if g.maxStorage != 0 && bytes > g.maxStorage {
		return fmt.Errorf("tensai: gpu buffer of %d bytes exceeds the device storage limit of %d", bytes, g.maxStorage)
	}
	return nil
}

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
	if err := g.checkSize(uint64(len(t.Data)) * 4); err != nil {
		return nil, err
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
	staging, stagingSize := g.takeReadback(bytes)
	if staging == 0 {
		return nil, errors.New("tensai: gpu readback buffer allocation failed")
	}
	reuse := false
	defer func() {
		if reuse {
			g.putReadback(staging, stagingSize)
		} else {
			fnBufferRelease(staging)
		}
	}()

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
	reuse = true
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
	return t.matmul(o, false)
}

// MatMulT multiplies t by o with o's last two axes read transposed: a
// (batch..., m, k) tensor times a (batch..., n, k) tensor yields
// (batch..., m, n), without materializing the transpose. This is the
// attention pattern q @ k^T.
func (t *GPUTensor) MatMulT(o *GPUTensor) (*GPUTensor, error) {
	return t.matmul(o, true)
}

func (t *GPUTensor) matmul(o *GPUTensor, transB bool) (*GPUTensor, error) {
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
	var n int
	if transB {
		if o.shape[nb-1] != k {
			return nil, fmt.Errorf("tensai: matmul-t shape mismatch: %v * %v^T", t.shape, o.shape)
		}
		n = o.shape[nb-2]
	} else {
		if o.shape[nb-2] != k {
			return nil, fmt.Errorf("tensai: matmul shape mismatch: %v * %v", t.shape, o.shape)
		}
		n = o.shape[nb-1]
	}
	batch, err := broadcastShapes(t.shape[:na-2], o.shape[:nb-2])
	if err != nil {
		return nil, err
	}
	batches := prodDims(batch)
	outShape := append(append([]int(nil), batch...), m, n)

	// Element offsets of each batch's (contiguous) matrix in t, o, out.
	as := broadcastStrides(t.shape[:na-2], batch)
	bs := broadcastStrides(o.shape[:nb-2], batch)
	offs := make([]uint32, 4*batches)
	for bi := 0; bi < batches; bi++ {
		offA, offB := 0, 0
		for d, rem := len(batch)-1, bi; d >= 0; d-- {
			i := rem % batch[d]
			rem /= batch[d]
			offA += i * as[d]
			offB += i * bs[d]
		}
		offs[4*bi] = uint32(offA * m * k)
		offs[4*bi+1] = uint32(offB * k * n)
		offs[4*bi+2] = uint32(bi * m * n)
	}
	ldb := n
	if transB {
		ldb = k
	}
	return t.g.stridedMatMul(t, o, outShape, transB, m, k, n, batches, k, ldb, n, offs)
}

// stridedMatMul runs `batches` independent (m x k) x (k x n) products —
// x (n x k) read transposed when transB — where offs holds per-batch
// element offsets (offA, offB, offC, 0) into a, b, and the freshly
// allocated output, and lda/ldb/ldc are the row strides. Explicit strides
// let callers carve sub-matrices out of a wider layout, which is how
// multi-head attention splits heads without materializing a permute.
func (g *GPU) stridedMatMul(a, b *GPUTensor, outShape []int, transB bool, m, k, n, batches, lda, ldb, ldc int, offs []uint32) (*GPUTensor, error) {
	if batches > 65535 {
		return nil, fmt.Errorf("tensai: gpu matmul batch count %d exceeds 65535", batches)
	}
	params := [8]uint32{
		uint32(m), uint32(k), uint32(n), uint32(batches),
		uint32(lda), uint32(ldb), uint32(ldc), 0,
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(prodDims(outShape)) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams := g.newBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOffs := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.newBuffer(gpuTensorUsage, outBytes)
	defer fnBufferRelease(bufParams)
	defer fnBufferRelease(bufOffs)

	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	// The 64x64 register-tiled kernels win on big products. Skinny ones
	// (m or n under half a block) would leave most of each workgroup
	// idle, and products with only a handful of 64x64 blocks cannot fill
	// the GPU at all — both take the plain 16x16 variants, whose grid has
	// sixteen times the workgroups.
	block := gpuBlock
	pipe, lay := g.pipes.matmul, g.pipes.layMatmul
	if transB {
		pipe, lay = g.pipes.matmulT, g.pipes.layMatmulT
	}
	blocks := ((m + gpuBlock - 1) / gpuBlock) * ((n + gpuBlock - 1) / gpuBlock) * batches
	if m < 32 || n < 32 || blocks < 16 {
		block = 16
		pipe, lay = g.pipes.matmulS, g.pipes.layMatmulS
		if transB {
			pipe, lay = g.pipes.matmulTS, g.pipes.layMatmulTS
		}
	}
	entries := [5]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, size: 32},
		{binding: 1, buffer: a.buf, size: uint64(a.Size()) * 4},
		{binding: 2, buffer: b.buf, size: uint64(b.Size()) * 4},
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 4, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.makeBindGroup(lay, entries[:])
	runtime.KeepAlive(&entries)
	defer fnBindGroupRelease(bindGroup)

	err := g.dispatch(pipe, bindGroup,
		uint32((n+block-1)/block), uint32((m+block-1)/block), uint32(batches))
	if err != nil {
		fnBufferRelease(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: outShape}, nil
}

// split2D spreads n workgroups over a 2-D dispatch grid, since a single
// axis is capped at 65535.
func split2D(n int) (x, y uint32) {
	if n <= 65535 {
		return uint32(n), 1
	}
	return 65535, uint32((n + 65534) / 65535)
}

// Scale multiplies every element by s, in place — the GPU counterpart of
// Tensor.Scale.
func (t *GPUTensor) Scale(s Float) error {
	if t.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	count := t.Size()
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	params := [4]uint32{uint32(count), math.Float32bits(float32(s))}
	bufParams := g.newBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	defer fnBufferRelease(bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [2]wgpuBindGroupEntry{
		{binding: 5, buffer: bufParams, size: 16},
		{binding: 6, buffer: t.buf, size: uint64(count) * 4},
	}
	bindGroup := g.makeBindGroup(g.pipes.layScale, entries[:])
	runtime.KeepAlive(&entries)
	defer fnBindGroupRelease(bindGroup)

	x, y := split2D((count + gpuKernelWG - 1) / gpuKernelWG)
	return g.dispatch(g.pipes.scale, bindGroup, x, y, 1)
}

// Softmax applies a numerically stable softmax over the last axis,
// returning a new GPU-resident tensor.
func (t *GPUTensor) Softmax() (*GPUTensor, error) {
	return t.softmax(0, 0)
}

// softmax optionally applies a causal mask: with qmod > 0, row r is query
// index r%qmod and attends to the first r%qmod+off+1 columns; masked
// columns come out as exactly zero.
func (t *GPUTensor) softmax(qmod, off int) (*GPUTensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if len(t.shape) == 0 {
		return nil, errors.New("tensai: gpu softmax needs at least 1 axis")
	}
	cols := t.shape[len(t.shape)-1]
	rows := t.Size() / cols
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	bytes := uint64(t.Size()) * 4
	params := [4]uint32{uint32(rows), uint32(cols), uint32(qmod), uint32(off)}
	bufParams := g.newBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufOut := g.newBuffer(gpuTensorUsage, bytes)
	defer fnBufferRelease(bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [3]wgpuBindGroupEntry{
		{binding: 7, buffer: bufParams, size: 16},
		{binding: 8, buffer: t.buf, size: bytes},
		{binding: 9, buffer: bufOut, size: bytes},
	}
	bindGroup := g.makeBindGroup(g.pipes.laySoftmax, entries[:])
	runtime.KeepAlive(&entries)
	defer fnBindGroupRelease(bindGroup)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.softmax, bindGroup, x, y, 1); err != nil {
		fnBufferRelease(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, nil
}

// Attention computes scaled dot-product attention softmax(q*k^T/sqrt(d))*v
// entirely on the GPU — the resident counterpart of the autograd Attention.
// q, k, v are (batch..., seqLen, d) tensors; nothing touches host memory.
func (q *GPUTensor) Attention(k, v *GPUTensor) (*GPUTensor, error) {
	return q.attention(k, v, false)
}

// CausalAttention is Attention with a causal mask: query i attends only to
// key positions 0..i+(seqKV-seqQ), so k and v may hold seqKV >= seqQ
// positions with the queries aligned to their end — the prompt-prefill
// pattern of autoregressive models.
func (q *GPUTensor) CausalAttention(k, v *GPUTensor) (*GPUTensor, error) {
	return q.attention(k, v, true)
}

func (q *GPUTensor) attention(k, v *GPUTensor, causal bool) (*GPUTensor, error) {
	nq := len(q.shape)
	if nq < 2 {
		return nil, fmt.Errorf("tensai: attention needs at least 2 axes: %v", q.shape)
	}
	seq := q.shape[nq-2]
	seqKV := k.shape[len(k.shape)-2]
	if causal && seqKV < seq {
		return nil, fmt.Errorf("tensai: causal attention needs seqKV >= seqQ, got %d < %d", seqKV, seq)
	}
	scores, err := q.MatMulT(k)
	if err != nil {
		return nil, err
	}
	defer scores.Free()
	if err := scores.Scale(1 / sqrtF(Float(q.shape[nq-1]))); err != nil {
		return nil, err
	}
	qmod, off := 0, 0
	if causal {
		qmod, off = seq, seqKV-seq
	}
	weights, err := scores.softmax(qmod, off)
	if err != nil {
		return nil, err
	}
	defer weights.Free()
	return weights.MatMul(v)
}

// MultiHeadAttention computes multi-head scaled dot-product attention
// entirely on the GPU. q is (batch..., seqQ, d) and k, v are
// (batch..., seqKV, d) in the usual packed head layout, d = heads * dh;
// the result is (batch..., seqQ, d). Heads are carved out of the packed
// layout with strided kernels, so no permute is ever materialized.
func (q *GPUTensor) MultiHeadAttention(k, v *GPUTensor, heads int) (*GPUTensor, error) {
	return q.multiHeadAttention(k, v, heads, false)
}

// CausalMultiHeadAttention is MultiHeadAttention with a causal mask: query
// i attends only to key positions 0..i+(seqKV-seqQ), so k and v may hold
// seqKV >= seqQ positions with the queries aligned to their end — the
// prompt-prefill pattern of autoregressive models.
func (q *GPUTensor) CausalMultiHeadAttention(k, v *GPUTensor, heads int) (*GPUTensor, error) {
	return q.multiHeadAttention(k, v, heads, true)
}

func (q *GPUTensor) multiHeadAttention(k, v *GPUTensor, heads int, causal bool) (*GPUTensor, error) {
	if q.freed || k.freed || v.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if q.g != k.g || q.g != v.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	nq := len(q.shape)
	if nq < 2 {
		return nil, fmt.Errorf("tensai: attention needs at least 2 axes: %v", q.shape)
	}
	d := q.shape[nq-1]
	if heads <= 0 || d%heads != 0 {
		return nil, fmt.Errorf("tensai: %d heads do not divide model dimension %d", heads, d)
	}
	if !sameDims(k.shape, v.shape) {
		return nil, fmt.Errorf("tensai: attention k and v shapes differ: %v vs %v", k.shape, v.shape)
	}
	if len(k.shape) != nq || k.shape[nq-1] != d || !sameDims(k.shape[:nq-2], q.shape[:nq-2]) {
		return nil, fmt.Errorf("tensai: attention shape mismatch: q %v, k %v", q.shape, k.shape)
	}
	seq := q.shape[nq-2]
	seqKV := k.shape[nq-2]
	if causal && seqKV < seq {
		return nil, fmt.Errorf("tensai: causal attention needs seqKV >= seqQ, got %d < %d", seqKV, seq)
	}
	batch := prodDims(q.shape[:nq-2])
	dh := d / heads
	bh := batch * heads

	// scores (batch*heads, seq, seqKV) = q_head @ k_head^T, each head a
	// (seq x dh) sub-matrix with row stride d inside the packed layout.
	offs := make([]uint32, 4*bh)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			i := b*heads + h
			offs[4*i] = uint32(b*seq*d + h*dh)
			offs[4*i+1] = uint32(b*seqKV*d + h*dh)
			offs[4*i+2] = uint32(i * seq * seqKV)
		}
	}
	scores, err := q.g.stridedMatMul(q, k, []int{bh, seq, seqKV}, true,
		seq, dh, seqKV, bh, d, d, seqKV, offs)
	if err != nil {
		return nil, err
	}
	defer scores.Free()
	if err := scores.Scale(1 / sqrtF(Float(dh))); err != nil {
		return nil, err
	}
	qmod, off := 0, 0
	if causal {
		qmod, off = seq, seqKV-seq
	}
	weights, err := scores.softmax(qmod, off)
	if err != nil {
		return nil, err
	}
	defer weights.Free()

	// out (batch..., seq, d) = weights @ v_head, written back into the
	// packed head layout.
	offs2 := make([]uint32, 4*bh)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			i := b*heads + h
			offs2[4*i] = uint32(i * seq * seqKV)
			offs2[4*i+1] = uint32(b*seqKV*d + h*dh)
			offs2[4*i+2] = uint32(b*seq*d + h*dh)
		}
	}
	outShape := append(append([]int(nil), q.shape[:nq-2]...), seq, d)
	return q.g.stridedMatMul(weights, v, outShape, false,
		seq, seqKV, dh, bh, seqKV, d, d, offs2)
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

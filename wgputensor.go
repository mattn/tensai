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

// attn_causal fuses q@k^T, the causal softmax, and the value mix with an
// online (flash-attention style) softmax, so the scores matrix is never
// materialized. One 64-lane workgroup handles one (batch*head, query)
// pair: per 64-wide kv tile the lanes first each compute one score and
// reduce its max and sum, then switch axes and each accumulate one or two
// output channels (dh up to 128), rescaling the running state as the max
// grows. offs (binding 3) holds per-batch*head element offsets
// (q, kv, out); rows in q and out stride by ap.d and rows in k and v by
// ap.dkv, so heads stay packed and k/v may pack fewer heads than q
// (grouped-query attention).
struct AttnParams {
    seqQ: u32, seqKV: u32, dh: u32, d: u32,
    rows: u32, off: u32, dkv: u32, window: u32,
}
@group(0) @binding(10) var<uniform> ap: AttnParams;
@group(0) @binding(11) var<storage, read> aq: array<f32>;
@group(0) @binding(12) var<storage, read> ak: array<f32>;
@group(0) @binding(13) var<storage, read> av: array<f32>;
@group(0) @binding(14) var<storage, read_write> aout: array<f32>;

const AT = 64u;
var<workgroup> qrow: array<f32, 256>;
var<workgroup> ap_sc: array<f32, 64>;
var<workgroup> ap_red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn attn_causal(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= ap.rows) {
        return;
    }
    let bh = row / ap.seqQ;
    let qi = row % ap.seqQ;
    let offQ = offs[bh].x;
    let offKV = offs[bh].y;
    let offO = offs[bh].z;
    let t = lid.x;
    for (var c = t; c < ap.dh; c = c + 64u) {
        qrow[c] = aq[offQ + qi * ap.d + c];
    }
    workgroupBarrier();
    let limit = qi + ap.off + 1u;
    // Sliding-window layers (Gemma) see only the last window positions.
    var start = 0u;
    if (ap.window > 0u && limit > ap.window) {
        start = limit - ap.window;
    }
    let scale = inverseSqrt(f32(ap.dh));
    var m = -3.40282e38;
    var l = 0.0;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    let tiles = (limit + AT - 1u) / AT;
    for (var tt = start / AT; tt < tiles; tt = tt + 1u) {
        // Lane t scores kv position tt*64+t.
        let j = tt * AT + t;
        var s = -3.40282e38;
        if (j >= start && j < limit) {
            var dot = 0.0;
            for (var c = 0u; c < ap.dh; c = c + 1u) {
                dot = dot + qrow[c] * ak[offKV + j * ap.dkv + c];
            }
            s = dot * scale;
        }
        ap_red[t] = s;
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) { ap_red[t] = max(ap_red[t], ap_red[t + r]); }
            workgroupBarrier();
        }
        let mNew = max(m, ap_red[0]);
        workgroupBarrier();
        var p = 0.0;
        if (j >= start && j < limit) {
            p = exp(s - mNew);
        }
        ap_sc[t] = p;
        ap_red[t] = p;
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) { ap_red[t] = ap_red[t] + ap_red[t + r]; }
            workgroupBarrier();
        }
        let tileSum = ap_red[0];
        // exp underflows to zero on the first tile, where m is -inf-like.
        let rescale = exp(m - mNew);
        l = l * rescale + tileSum;
        m = mNew;
        // Lane t now accumulates output channels t, t+64, t+128, t+192.
        acc0 = acc0 * rescale;
        acc1 = acc1 * rescale;
        acc2 = acc2 * rescale;
        acc3 = acc3 * rescale;
        let jEnd = min(limit, tt * AT + AT);
        for (var jj = max(start, tt * AT); jj < jEnd; jj = jj + 1u) {
            let pj = ap_sc[jj - tt * AT];
            if (t < ap.dh) {
                acc0 = acc0 + pj * av[offKV + jj * ap.dkv + t];
            }
            if (64u + t < ap.dh) {
                acc1 = acc1 + pj * av[offKV + jj * ap.dkv + 64u + t];
            }
            if (128u + t < ap.dh) {
                acc2 = acc2 + pj * av[offKV + jj * ap.dkv + 128u + t];
            }
            if (192u + t < ap.dh) {
                acc3 = acc3 + pj * av[offKV + jj * ap.dkv + 192u + t];
            }
        }
        workgroupBarrier();
    }
    if (t < ap.dh) {
        aout[offO + qi * ap.d + t] = acc0 / l;
    }
    if (64u + t < ap.dh) {
        aout[offO + qi * ap.d + 64u + t] = acc1 / l;
    }
    if (128u + t < ap.dh) {
        aout[offO + qi * ap.d + 128u + t] = acc2 / l;
    }
    if (192u + t < ap.dh) {
        aout[offO + qi * ap.d + 192u + t] = acc3 / l;
    }
}

// qmatmul multiplies f32 activation rows by an int8 weight matrix that
// stays packed in GPU memory, four weights per u32, dequantizing in
// registers: the weight traffic — which dominates a decode matvec — is a
// quarter of the f32 kernel's. A 256-lane workgroup covers 16 packed words
// (64 output columns) of one activation row, with each word's rows split
// 16 ways: lane (rsub, wsub) strides the rows by 16 reading 16 adjacent
// words per step (coalesced, 64 bytes), and a shared-memory reduction
// folds the row splits before the guarded store. The split keeps a decode
// matvec — parallelism = output columns only — wide enough to fill the
// device.
struct QMParams { rows: u32, cols: u32, words: u32, m: u32 }
@group(0) @binding(15) var<uniform> qp: QMParams;
@group(0) @binding(16) var<storage, read> qwt: array<u32>;
@group(0) @binding(17) var<storage, read> qsc: array<f32>;
@group(0) @binding(18) var<storage, read> qxv: array<f32>;
@group(0) @binding(19) var<storage, read_write> qov: array<f32>;

const QWG = 16u;    // packed words per workgroup
const QSPLIT = 16u; // row splits sharing one word
var<workgroup> qred: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn qmatmul(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = wid.x * QWG + lid.x % QWG;
    let rsub = lid.x / QWG;
    let r = wid.y;
    let xbase = r * qp.rows;
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (w < qp.words) {
        for (var i = rsub; i < qp.rows; i = i + QSPLIT) {
            let xv = qxv[xbase + i];
            let pw = qwt[i * qp.words + w];
            acc = acc + xv * vec4<f32>(
                f32(i32(pw << 24u) >> 24u),
                f32(i32(pw << 16u) >> 24u),
                f32(i32(pw << 8u) >> 24u),
                f32(i32(pw) >> 24u));
        }
    }
    qred[lid.x] = acc;
    workgroupBarrier();
    // Lanes 0..15 each fold one word's sixteen partials; their w above is
    // already wid.x*QWG + lid.x.
    if (lid.x < QWG && w < qp.words) {
        var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var v = 0u; v < QSPLIT; v = v + 1u) {
            sum = sum + qred[v * QWG + lid.x];
        }
        let j = w * 4u;
        for (var l = 0u; l < 4u; l = l + 1u) {
            if (j + l < qp.cols) {
                qov[r * qp.cols + j + l] = sum[l] * qsc[j + l];
            }
        }
    }
}

// rmsnorm_row: one workgroup per row computes out = x * w / rms(x).
struct NormParams { rows: u32, n: u32, eps: f32, pad: u32 }
@group(0) @binding(20) var<uniform> np: NormParams;
@group(0) @binding(21) var<storage, read> nx: array<f32>;
@group(0) @binding(22) var<storage, read> nw: array<f32>;
@group(0) @binding(23) var<storage, read_write> nout: array<f32>;

var<workgroup> nred: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_row(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= np.rows) {
        return;
    }
    let base = row * np.n;
    var ss = 0.0;
    for (var i = lid.x; i < np.n; i = i + 256u) {
        let v = nx[base + i];
        ss = ss + v * v;
    }
    nred[lid.x] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { nred[lid.x] = nred[lid.x] + nred[lid.x + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(nred[0] / f32(np.n) + np.eps);
    for (var i = lid.x; i < np.n; i = i + 256u) {
        nout[base + i] = nx[base + i] * inv * nw[i];
    }
}

// rope_rows rotates every head of every row in place, half-split style:
// element pair (c, c+headSz/2) of each head rotates by pos*theta^(-2c/headSz),
// where row r sits at position pos0 + r.
struct RopeParams { rows: u32, d: u32, headSz: u32, pos0: u32, theta: f32 }
@group(0) @binding(24) var<uniform> rp: RopeParams;
@group(0) @binding(25) var<storage, read_write> rx: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn rope_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = rp.headSz / 2u;
    let pairs = (rp.d / rp.headSz) * half;
    if (gid.x >= pairs || gid.y >= rp.rows) {
        return;
    }
    let h = gid.x / half;
    let c = gid.x % half;
    let base = gid.y * rp.d + h * rp.headSz + c;
    let freq = pow(rp.theta, -2.0 * f32(c) / f32(rp.headSz));
    let ang = f32(rp.pos0 + gid.y) * freq;
    let sn = sin(ang);
    let cs = cos(ang);
    let a = rx[base];
    let b = rx[base + half];
    rx[base] = a * cs - b * sn;
    rx[base + half] = b * cs + a * sn;
}

// add_ip: dst += src, with src repeating when shorter (a row bias applied
// to every row). silu_mul_ip: dst = silu(dst) * src, the SwiGLU joint.
struct EltParams { count: u32, srcCount: u32 }
@group(0) @binding(26) var<uniform> ep: EltParams;
@group(0) @binding(27) var<storage, read_write> edst: array<f32>;
@group(0) @binding(28) var<storage, read> esrc: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn add_ip(@builtin(workgroup_id) wg: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < ep.count) {
        edst[idx] = edst[idx] + esrc[idx % ep.srcCount];
    }
}

@compute @workgroup_size(256, 1, 1)
fn silu_mul_ip(@builtin(workgroup_id) wg: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < ep.count) {
        let g = edst[idx];
        edst[idx] = g / (1.0 + exp(-g)) * esrc[idx % ep.srcCount];
    }
}

// gelu_mul_ip: dst = gelu_tanh(dst) * src — Gemma's gate.
@compute @workgroup_size(256, 1, 1)
fn gelu_mul_ip(@builtin(workgroup_id) wg: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < ep.count) {
        let g = edst[idx];
        let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
        edst[idx] = 0.5 * g * (1.0 + tanh(inner)) * esrc[idx % ep.srcCount];
    }
}

// q4matmul multiplies f32 activation rows by an int4 weight matrix packed
// four row-pair bytes per u32 (a byte holds one column's even row in the
// low nibble and odd row in the high one, offset-binary). The -8 offset
// subtracts in registers — no correction plane needed — and each group's
// running partial folds with its (group, column) scale at the boundary.
// Same shape as qmatmul: 16 words x 16 row-splits per 256-lane workgroup
// with a shared-memory reduction.
struct Q4MParams { rows: u32, cols: u32, words: u32, m: u32, groups: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(29) var<uniform> q4p: Q4MParams;
@group(0) @binding(30) var<storage, read> q4w: array<u32>;
@group(0) @binding(31) var<storage, read> q4sc: array<f32>;
@group(0) @binding(32) var<storage, read> q4x: array<f32>;
@group(0) @binding(33) var<storage, read_write> q4out: array<f32>;

var<workgroup> q4red: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn q4matmul(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = wid.x * 16u + lid.x % 16u;
    let rsub = lid.x / 16u;
    let r = wid.y;
    let xbase = r * q4p.rows;
    var total = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (w < q4p.words) {
        let pairs = (q4p.rows + 1u) / 2u;
        var run = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var g = 0xffffffffu;
        for (var i2 = rsub; i2 < pairs; i2 = i2 + 16u) {
            let gi = i2 / 32u; // q4Group/2 pair rows per group
            if (gi != g) {
                if (g != 0xffffffffu) {
                    let sb = g * q4p.words * 4u + w * 4u;
                    total = total + run * vec4<f32>(q4sc[sb], q4sc[sb + 1u], q4sc[sb + 2u], q4sc[sb + 3u]);
                }
                run = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                g = gi;
            }
            let x0 = q4x[xbase + 2u * i2];
            var x1 = 0.0;
            if (2u * i2 + 1u < q4p.rows) {
                x1 = q4x[xbase + 2u * i2 + 1u];
            }
            let pw = q4w[i2 * q4p.words + w];
            let lo = vec4<f32>(
                f32(pw & 0xFu), f32(pw >> 8u & 0xFu),
                f32(pw >> 16u & 0xFu), f32(pw >> 24u & 0xFu)) - vec4<f32>(8.0, 8.0, 8.0, 8.0);
            let hi = vec4<f32>(
                f32(pw >> 4u & 0xFu), f32(pw >> 12u & 0xFu),
                f32(pw >> 20u & 0xFu), f32(pw >> 28u & 0xFu)) - vec4<f32>(8.0, 8.0, 8.0, 8.0);
            run = run + x0 * lo + x1 * hi;
        }
        if (g != 0xffffffffu) {
            let sb = g * q4p.words * 4u + w * 4u;
            total = total + run * vec4<f32>(q4sc[sb], q4sc[sb + 1u], q4sc[sb + 2u], q4sc[sb + 3u]);
        }
    }
    q4red[lid.x] = total;
    workgroupBarrier();
    if (lid.x < 16u && w < q4p.words) {
        var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var v = 0u; v < 16u; v = v + 1u) {
            sum = sum + q4red[v * 16u + lid.x];
        }
        let j = w * 4u;
        for (var l = 0u; l < 4u; l = l + 1u) {
            if (j + l < q4p.cols) {
                q4out[r * q4p.cols + j + l] = sum[l];
            }
        }
    }
}
`

// gpuPipelines holds one compute pipeline (and its auto bind-group layout)
// per kernel entry point. It is embedded in each binding generation's GPU
// struct.
type gpuPipelines struct {
	matmul, matmulT, matmulS, matmulTS             uintptr
	scale, softmax, attn, qmatmul                  uintptr
	rmsnorm, rope, addIP, siluMulIP, q4matmul      uintptr
	geluMulIP                                      uintptr
	layMatmul, layMatmulT, layMatmulS, layMatmulTS uintptr
	layScale, laySoftmax, layAttn, layQmatmul      uintptr
	layRmsnorm, layRope, layAddIP, laySiluMulIP    uintptr
	layQ4matmul, layGeluMulIP                      uintptr
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
		{&g.pipes.attn, &g.pipes.layAttn, "attn_causal"},
		{&g.pipes.qmatmul, &g.pipes.layQmatmul, "qmatmul"},
		{&g.pipes.rmsnorm, &g.pipes.layRmsnorm, "rmsnorm_row"},
		{&g.pipes.rope, &g.pipes.layRope, "rope_rows"},
		{&g.pipes.addIP, &g.pipes.layAddIP, "add_ip"},
		{&g.pipes.siluMulIP, &g.pipes.laySiluMulIP, "silu_mul_ip"},
		{&g.pipes.q4matmul, &g.pipes.layQ4matmul, "q4matmul"},
		{&g.pipes.geluMulIP, &g.pipes.layGeluMulIP, "gelu_mul_ip"},
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
		g.pipes.layMatmulTS, g.pipes.layScale, g.pipes.laySoftmax, g.pipes.layAttn,
		g.pipes.layQmatmul, g.pipes.layRmsnorm, g.pipes.layRope,
		g.pipes.layAddIP, g.pipes.laySiluMulIP, g.pipes.layQ4matmul,
		g.pipes.layGeluMulIP,
	} {
		if h != 0 {
			fnLayoutRelease(h)
		}
	}
	for _, h := range []uintptr{
		g.pipes.matmul, g.pipes.matmulT, g.pipes.matmulS,
		g.pipes.matmulTS, g.pipes.scale, g.pipes.softmax, g.pipes.attn,
		g.pipes.qmatmul, g.pipes.rmsnorm, g.pipes.rope,
		g.pipes.addIP, g.pipes.siluMulIP, g.pipes.q4matmul,
		g.pipes.geluMulIP,
	} {
		if h != 0 {
			fnPipelineRelease(h)
		}
	}
}

// dispatch runs one compute pass and pumps the error callback once; the
// caller holds wgpuMu. Inside a batch (see BeginBatch) the pass is only
// recorded — submission waits for Flush.
func (g *GPU) dispatch(pipe, bindGroup uintptr, x, y, z uint32) error {
	encoder := g.batchEnc
	if encoder == 0 {
		encoder = fnDeviceCreateCmdEncoder(g.device, nil)
	}
	pass := fnEncoderBeginComputePass(encoder, nil)
	fnPassSetPipeline(pass, pipe)
	fnPassSetBindGroup(pass, 0, bindGroup, 0, nil)
	fnPassDispatch(pass, x, y, z)
	fnPassEnd(pass)
	fnPassRelease(pass)
	if g.batchEnc != 0 {
		if uncapturedCB != "" {
			return fmt.Errorf("tensai: gpu dispatch failed: %s", uncapturedCB)
		}
		return nil
	}
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

// BeginBatch opens a command encoder that collects every subsequent
// operation on this GPU — compute dispatches and buffer copies — into one
// submission, which Flush sends. One submission instead of hundreds is
// the difference between a usable and an unusable decode step on drivers
// with per-submit overhead. Operations inside a batch report validation
// errors at Flush; Download flushes an open batch implicitly.
func (g *GPU) BeginBatch() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return errors.New("tensai: gpu is closed")
	}
	if g.batchEnc != 0 {
		return errors.New("tensai: gpu batch already open")
	}
	uncapturedCB = ""
	g.batchEnc = fnDeviceCreateCmdEncoder(g.device, nil)
	return nil
}

// Flush submits the batch opened by BeginBatch. Safe to call with no batch
// open.
func (g *GPU) Flush() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	return g.flushLocked()
}

// flushLocked submits and closes an open batch encoder; the caller holds
// g.mu and wgpuMu.
func (g *GPU) flushLocked() error {
	if g.batchEnc == 0 {
		return nil
	}
	encoder := g.batchEnc
	g.batchEnc = 0
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)
	fnDevicePoll(g.device, 0, nil)
	g.drainPending()
	if uncapturedCB != "" {
		return fmt.Errorf("tensai: gpu batch failed: %s", uncapturedCB)
	}
	return nil
}

// gpuBufferPool recycles transient buffers — op parameter uniforms and
// intermediate tensors — keyed by exact (usage, size). A decode step
// allocates the same shapes every token, so after the first token every
// request hits. While a batch is open, returned buffers wait in pending:
// reuse writes go through fnQueueWriteBuffer, which jumps ahead of the
// still-unsubmitted encoder, so a buffer must not be handed out again
// until the batch that references it is flushed.
type gpuBufferPool struct {
	free    map[[2]uint64][]uintptr
	pending []pooledBuf
	bytes   uint64
}

type pooledBuf struct {
	usage, size uint64
	buf         uintptr
}

// gpuPoolMaxBytes caps the pooled total; beyond it buffers just release.
const gpuPoolMaxBytes = 512 << 20

// takeBuffer returns a pooled buffer of exactly this usage and size, or a
// fresh one. The caller holds g.mu and wgpuMu.
func (g *GPU) takeBuffer(usage, size uint64) uintptr {
	key := [2]uint64{usage, size}
	if l := g.pool.free[key]; len(l) > 0 {
		buf := l[len(l)-1]
		g.pool.free[key] = l[:len(l)-1]
		g.pool.bytes -= size
		return buf
	}
	return g.newBuffer(usage, size)
}

// putBuffer returns a buffer to the pool (or releases it past the cap).
// The caller holds wgpuMu.
func (g *GPU) putBuffer(usage, size uint64, buf uintptr) {
	if g.closed || g.pool.bytes+size > gpuPoolMaxBytes {
		g.dropBuffer(buf)
		return
	}
	if g.batchEnc != 0 {
		g.pool.pending = append(g.pool.pending, pooledBuf{usage, size, buf})
		return
	}
	if g.pool.free == nil {
		g.pool.free = make(map[[2]uint64][]uintptr)
	}
	key := [2]uint64{usage, size}
	g.pool.free[key] = append(g.pool.free[key], buf)
	g.pool.bytes += size
}

// drainPending moves batch-held buffers into the free lists once their
// batch has been submitted. The caller holds wgpuMu.
func (g *GPU) drainPending() {
	for _, p := range g.pool.pending {
		if g.pool.bytes+p.size > gpuPoolMaxBytes {
			g.dropBuffer(p.buf)
			continue
		}
		if g.pool.free == nil {
			g.pool.free = make(map[[2]uint64][]uintptr)
		}
		key := [2]uint64{p.usage, p.size}
		g.pool.free[key] = append(g.pool.free[key], p.buf)
		g.pool.bytes += p.size
	}
	g.pool.pending = g.pool.pending[:0]
}

// releasePool drops every pooled buffer during Close. The caller holds
// wgpuMu.
func (g *GPU) releasePool() {
	for _, l := range g.pool.free {
		for _, buf := range l {
			fnBufferRelease(buf)
		}
	}
	for _, p := range g.pool.pending {
		fnBufferRelease(p.buf)
	}
	g.pool = gpuBufferPool{}
	g.invalidateBindGroups()
}

// bgKey identifies a bind group by its layout and bound buffers. Decode
// runs the same operations against the same (pooled, so handle-stable)
// buffers every token, so almost every lookup hits after the first token.
type bgKey struct {
	layout uintptr
	n      int
	e      [6]struct {
		binding uint32
		buf     uintptr
		size    uint64
	}
}

// bgCacheMax caps the cache; overflow drops everything (simple, and far
// beyond what a decode loop creates).
const bgCacheMax = 8192

// cachedBindGroup returns a bind group for the entries, creating and
// retaining it on first use. Cached groups are owned by the cache — the
// caller must not release them. The caller holds g.mu and wgpuMu.
func (g *GPU) cachedBindGroup(layout uintptr, entries []wgpuBindGroupEntry) uintptr {
	var key bgKey
	key.layout = layout
	key.n = len(entries)
	for i, e := range entries {
		key.e[i].binding = e.binding
		key.e[i].buf = e.buffer
		key.e[i].size = e.size
	}
	if bg, ok := g.bgCache[key]; ok {
		return bg
	}
	if len(g.bgCache) >= bgCacheMax {
		g.invalidateBindGroups()
	}
	bg := g.makeBindGroup(layout, entries)
	if g.bgCache == nil {
		g.bgCache = make(map[bgKey]uintptr)
	}
	g.bgCache[key] = bg
	return bg
}

// invalidateBindGroups releases every cached bind group. It must run
// whenever a buffer is truly released (dropBuffer), since a later
// allocation may reuse the handle address and silently collide with a
// stale cache entry. The caller holds wgpuMu.
func (g *GPU) invalidateBindGroups() {
	for _, bg := range g.bgCache {
		fnBindGroupRelease(bg)
	}
	g.bgCache = nil
}

// dropBuffer releases a buffer for real (not into the pool) and
// invalidates the bind-group cache. The caller holds wgpuMu.
func (g *GPU) dropBuffer(buf uintptr) {
	fnBufferRelease(buf)
	g.invalidateBindGroups()
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
		g.dropBuffer(g.readback.buf)
		g.readback = gpuReadbackBuffer{}
	}
	return g.newBuffer(wgpuBufferUsageMapRead|wgpuBufferUsageCopyDst, bytes), bytes
}

// putReadback retains a successfully unmapped staging buffer for the next
// download. The caller holds GPU.mu and wgpuMu.
func (g *GPU) putReadback(buf uintptr, size uint64) {
	if g.readback.buf != 0 {
		g.dropBuffer(g.readback.buf)
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
	buf := g.takeBuffer(gpuTensorUsage, uint64(len(t.Data))*4)
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
	if err := g.flushLocked(); err != nil {
		return nil, err
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

// DownloadRange copies elements [off, off+n) back into host memory as a
// flat tensor — reading freshly appended rows out of a resident cache
// without moving the whole buffer. Like Download it flushes an open
// batch first.
func (t *GPUTensor) DownloadRange(off, n int) (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if off < 0 || n <= 0 || off+n > t.Size() {
		return nil, fmt.Errorf("tensai: download of %d elements at %d overflows %d", n, off, t.Size())
	}
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	if err := g.flushLocked(); err != nil {
		return nil, err
	}
	out := NewTensor(n)
	bytes := uint64(n) * 4
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
	fnEncoderCopyBuffer(encoder, t.buf, uint64(off)*4, staging, 0, bytes)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)
	src, err := g.mapRead(staging, bytes)
	if err != nil {
		return nil, err
	}
	copy(out.Data, unsafe.Slice((*Float)(src), n))
	fnBufferUnmap(staging)
	reuse = true
	return out, nil
}

// Free releases the GPU buffer (into the transient pool, from which the
// next same-sized tensor will reuse it). The tensor must not be used
// afterwards; calling Free again is a no-op.
func (t *GPUTensor) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if t.freed {
		return
	}
	t.freed = true
	t.g.putBuffer(gpuTensorUsage, uint64(t.Size())*4, t.buf)
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
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOffs := g.takeBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.takeBuffer(gpuTensorUsage, outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	defer g.putBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4, bufOffs)

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
	bindGroup := g.cachedBindGroup(lay, entries[:])
	runtime.KeepAlive(&entries)

	err := g.dispatch(pipe, bindGroup,
		uint32((n+block-1)/block), uint32((m+block-1)/block), uint32(batches))
	if err != nil {
		g.dropBuffer(bufOut)
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
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [2]wgpuBindGroupEntry{
		{binding: 5, buffer: bufParams, size: 16},
		{binding: 6, buffer: t.buf, size: uint64(count) * 4},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layScale, entries[:])
	runtime.KeepAlive(&entries)

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
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufOut := g.takeBuffer(gpuTensorUsage, bytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [3]wgpuBindGroupEntry{
		{binding: 7, buffer: bufParams, size: 16},
		{binding: 8, buffer: t.buf, size: bytes},
		{binding: 9, buffer: bufOut, size: bytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.laySoftmax, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.softmax, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
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
	if causal && dh <= 128 {
		return q.fusedCausalMHA(k, v, heads, batch, seq, seqKV, d, dh, bh)
	}

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

// fusedCausalMHA runs causal multi-head attention as one flash-attention
// style dispatch: the scores matrix is never materialized, so memory use
// is just q, k, v, and the output, independent of sequence length.
func (q *GPUTensor) fusedCausalMHA(k, v *GPUTensor, heads, batch, seq, seqKV, d, dh, bh int) (*GPUTensor, error) {
	offs := make([]uint32, 4*bh)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			i := b*heads + h
			offs[4*i] = uint32(b*seq*d + h*dh)
			offs[4*i+1] = uint32(b*seqKV*d + h*dh)
			offs[4*i+2] = uint32(b*seq*d + h*dh)
		}
	}
	rows := bh * seq
	params := [8]uint32{
		uint32(seq), uint32(seqKV), uint32(dh), uint32(d),
		uint32(rows), uint32(seqKV - seq), uint32(d), 0,
	}
	outShape := append(append([]int(nil), q.shape[:len(q.shape)-2]...), seq, d)

	g := q.g
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
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOffs := g.takeBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.takeBuffer(gpuTensorUsage, outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	defer g.putBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4, bufOffs)

	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	entries := [6]wgpuBindGroupEntry{
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 10, buffer: bufParams, size: 32},
		{binding: 11, buffer: q.buf, size: uint64(q.Size()) * 4},
		{binding: 12, buffer: k.buf, size: uint64(k.Size()) * 4},
		{binding: 13, buffer: v.buf, size: uint64(v.Size()) * 4},
		{binding: 14, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layAttn, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.attn, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
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

// GPUQMatrix is an int8-quantized weight matrix resident in GPU memory:
// the packed weights of a QMatrix, four per u32 in plain row-major order,
// plus the per-column scales. Upload one with UploadQ8 and multiply
// activations into it with MatMul; the float32 weights never reach the
// device.
type GPUQMatrix struct {
	g           *GPU
	buf, scales uintptr
	rows, cols  int
	words       int // u32 words per weight row: ceil(cols/4)
	freed       bool
}

// UploadQ8 packs a quantized matrix into GPU memory. The QMatrix's
// interleaved row-quad layout (an AVX2 artifact) flattens back to
// row-major on the way in.
func (g *GPU) UploadQ8(q *QMatrix) (*GPUQMatrix, error) {
	if q.Rows == 0 || q.Cols == 0 {
		return nil, errors.New("tensai: cannot upload an empty matrix")
	}
	words := (q.Cols + 3) / 4
	packed := make([]uint32, q.Rows*words)
	for i := 0; i < q.Rows; i++ {
		base := (i / 4) * 4 * q.Cols
		for j := 0; j < q.Cols; j++ {
			b := uint32(uint8(q.Q[base+4*j+i%4]))
			packed[i*words+j/4] |= b << (8 * (j % 4))
		}
	}
	scales := make([]float32, words*4)
	for j, s := range q.Scale {
		scales[j] = float32(s)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	if err := g.checkSize(uint64(len(packed)) * 4); err != nil {
		return nil, err
	}
	buf := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(packed))*4)
	sbuf := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(scales))*4)
	if buf == 0 || sbuf == 0 {
		if buf != 0 {
			fnBufferRelease(buf)
		}
		if sbuf != 0 {
			fnBufferRelease(sbuf)
		}
		return nil, errors.New("tensai: gpu buffer allocation failed")
	}
	fnQueueWriteBuffer(g.queue, buf, 0, unsafe.Pointer(&packed[0]), uintptr(len(packed))*4)
	fnQueueWriteBuffer(g.queue, sbuf, 0, unsafe.Pointer(&scales[0]), uintptr(len(scales))*4)
	runtime.KeepAlive(&packed)
	runtime.KeepAlive(&scales)
	return &GPUQMatrix{g: g, buf: buf, scales: sbuf, rows: q.Rows, cols: q.Cols, words: words}, nil
}

// Shape returns (rows, cols).
func (q *GPUQMatrix) Shape() (int, int) { return q.rows, q.cols }

// Free releases the GPU buffers. Calling Free again is a no-op.
func (q *GPUQMatrix) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if q.freed {
		return
	}
	q.freed = true
	q.g.dropBuffer(q.buf)
	q.g.dropBuffer(q.scales)
}

// MatMul computes x @ Q on the GPU: x's last axis must equal the weight
// rows, and every leading axis is a batch of activation rows. Weights are
// dequantized in registers, so only a quarter of the f32 weight bytes
// cross the memory bus.
func (q *GPUQMatrix) MatMul(x *GPUTensor) (*GPUTensor, error) {
	if q.freed || x.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if q.g != x.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	n := len(x.shape)
	if n == 0 || x.shape[n-1] != q.rows {
		return nil, fmt.Errorf("tensai: gpu qmatmul shape mismatch: %v @ %dx%d", x.shape, q.rows, q.cols)
	}
	m := x.Size() / q.rows
	if m > 65535 {
		return nil, fmt.Errorf("tensai: gpu qmatmul batch of %d rows exceeds 65535", m)
	}
	outShape := append(append([]int(nil), x.shape[:n-1]...), q.cols)

	g := q.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(m*q.cols) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	params := [4]uint32{uint32(q.rows), uint32(q.cols), uint32(q.words), uint32(m)}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufOut := g.takeBuffer(gpuTensorUsage, outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [5]wgpuBindGroupEntry{
		{binding: 15, buffer: bufParams, size: 16},
		{binding: 16, buffer: q.buf, size: uint64(q.rows*q.words) * 4},
		{binding: 17, buffer: q.scales, size: uint64(q.words*4) * 4},
		{binding: 18, buffer: x.buf, size: uint64(x.Size()) * 4},
		{binding: 19, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layQmatmul, entries[:])
	runtime.KeepAlive(&entries)

	err := g.dispatch(g.pipes.qmatmul, bindGroup,
		uint32((q.words+15)/16), uint32(m), 1)
	if err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: outShape}, nil
}

// RMSNorm normalizes each row of the last axis by its root mean square and
// multiplies by the weight vector w (length = last axis), the pre-norm of
// Llama-family transformer blocks. Returns a new GPU-resident tensor.
func (t *GPUTensor) RMSNorm(w *GPUTensor, eps float64) (*GPUTensor, error) {
	if t.freed || w.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	nd := len(t.shape)
	if nd == 0 {
		return nil, errors.New("tensai: rmsnorm needs at least 1 axis")
	}
	n := t.shape[nd-1]
	if w.Size() != n {
		return nil, fmt.Errorf("tensai: rmsnorm weight length %d != last axis %d", w.Size(), n)
	}
	rows := t.Size() / n
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	params := [4]uint32{uint32(rows), uint32(n), math.Float32bits(float32(eps)), 0}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufOut := g.takeBuffer(gpuTensorUsage, uint64(t.Size())*4)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [4]wgpuBindGroupEntry{
		{binding: 20, buffer: bufParams, size: 16},
		{binding: 21, buffer: t.buf, size: uint64(t.Size()) * 4},
		{binding: 22, buffer: w.buf, size: uint64(w.Size()) * 4},
		{binding: 23, buffer: bufOut, size: uint64(t.Size()) * 4},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRmsnorm, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.rmsnorm, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, nil
}

// RMSNormEach is RMSNorm over consecutive groups of len(w) elements
// instead of whole rows of the last axis — Qwen3's per-head QK-norm,
// where every head of a packed projection normalizes against the same
// per-channel weights.
func (t *GPUTensor) RMSNormEach(w *GPUTensor, eps float64) (*GPUTensor, error) {
	if t.freed || w.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	n := w.Size()
	if n == 0 || t.Size()%n != 0 {
		return nil, fmt.Errorf("tensai: rmsnorm group of %d does not divide %d elements", n, t.Size())
	}
	rows := t.Size() / n
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	params := [4]uint32{uint32(rows), uint32(n), math.Float32bits(float32(eps)), 0}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufOut := g.takeBuffer(gpuTensorUsage, uint64(t.Size())*4)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [4]wgpuBindGroupEntry{
		{binding: 20, buffer: bufParams, size: 16},
		{binding: 21, buffer: t.buf, size: uint64(t.Size()) * 4},
		{binding: 22, buffer: w.buf, size: uint64(n) * 4},
		{binding: 23, buffer: bufOut, size: uint64(t.Size()) * 4},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRmsnorm, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.rmsnorm, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, nil
}

// RoPE applies rotary position embeddings in place, half-split style: the
// last axis divides into heads of headSz, and element pair (c, c+headSz/2)
// of each head in row r rotates by (pos0+r) * theta^(-2c/headSz).
func (t *GPUTensor) RoPE(headSz, pos0 int, theta float64) error {
	if t.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	nd := len(t.shape)
	if nd == 0 {
		return errors.New("tensai: rope needs at least 1 axis")
	}
	d := t.shape[nd-1]
	if headSz <= 0 || headSz%2 != 0 || d%headSz != 0 {
		return fmt.Errorf("tensai: rope head size %d does not divide axis %d", headSz, d)
	}
	rows := t.Size() / d
	if rows > 65535 {
		return fmt.Errorf("tensai: rope batch of %d rows exceeds 65535", rows)
	}
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	params := [8]uint32{
		uint32(rows), uint32(d), uint32(headSz), uint32(pos0),
		math.Float32bits(float32(theta)), 0, 0, 0,
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)

	entries := [2]wgpuBindGroupEntry{
		{binding: 24, buffer: bufParams, size: 32},
		{binding: 25, buffer: t.buf, size: uint64(t.Size()) * 4},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRope, entries[:])
	runtime.KeepAlive(&entries)

	pairs := (d / headSz) * (headSz / 2)
	return g.dispatch(g.pipes.rope, bindGroup, uint32((pairs+63)/64), uint32(rows), 1)
}

// eltwiseIP dispatches one of the two in-place elementwise kernels with o
// as the second operand, repeating o when it is shorter (a row bias
// against a batch of rows).
func (t *GPUTensor) eltwiseIP(pipe, lay uintptr, o *GPUTensor) error {
	if t.freed || o.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	if t.g != o.g {
		return errors.New("tensai: gpu tensors belong to different GPUs")
	}
	if o.Size() == 0 || t.Size()%o.Size() != 0 {
		return fmt.Errorf("tensai: elementwise size mismatch: %d vs %d", t.Size(), o.Size())
	}
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	params := [4]uint32{uint32(t.Size()), uint32(o.Size()), 0, 0}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [3]wgpuBindGroupEntry{
		{binding: 26, buffer: bufParams, size: 16},
		{binding: 27, buffer: t.buf, size: uint64(t.Size()) * 4},
		{binding: 28, buffer: o.buf, size: uint64(o.Size()) * 4},
	}
	bindGroup := g.cachedBindGroup(lay, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((t.Size() + gpuKernelWG - 1) / gpuKernelWG)
	return g.dispatch(pipe, bindGroup, x, y, 1)
}

// Add computes t += o elementwise in place. o may be shorter as long as
// its size divides t's: it repeats, which applies a row bias to every row.
func (t *GPUTensor) Add(o *GPUTensor) error {
	return t.eltwiseIP(t.g.pipes.addIP, t.g.pipes.layAddIP, o)
}

// SiluMul computes t = silu(t) * o elementwise in place — the SwiGLU
// joint, with t the gate projection and o the up projection.
func (t *GPUTensor) SiluMul(o *GPUTensor) error {
	return t.eltwiseIP(t.g.pipes.siluMulIP, t.g.pipes.laySiluMulIP, o)
}

// GeluMul computes t = gelu_tanh(t) * o elementwise in place — Gemma's
// gate.
func (t *GPUTensor) GeluMul(o *GPUTensor) error {
	return t.eltwiseIP(t.g.pipes.geluMulIP, t.g.pipes.layGeluMulIP, o)
}

// CopyRowsInto copies t's whole buffer into dst starting at element offset
// off — appending freshly projected k/v rows to a preallocated cache
// without leaving the device.
func (t *GPUTensor) CopyRowsInto(dst *GPUTensor, off int) error {
	if t.freed || dst.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	if t.g != dst.g {
		return errors.New("tensai: gpu tensors belong to different GPUs")
	}
	if off < 0 || off+t.Size() > dst.Size() {
		return fmt.Errorf("tensai: copy of %d elements at %d overflows %d", t.Size(), off, dst.Size())
	}
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""
	if g.batchEnc != 0 {
		fnEncoderCopyBuffer(g.batchEnc, t.buf, 0, dst.buf, uint64(off)*4, uint64(t.Size())*4)
		return nil
	}
	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	fnEncoderCopyBuffer(encoder, t.buf, 0, dst.buf, uint64(off)*4, uint64(t.Size())*4)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)
	if uncapturedCB != "" {
		return fmt.Errorf("tensai: gpu copy failed: %s", uncapturedCB)
	}
	return nil
}

// GroupedCausalAttention is causal multi-head attention against a
// KV cache that may pack fewer heads than the queries (grouped-query
// attention) and hold more positions than are valid: q is (seq, heads*dh),
// k and v hold at least seqKV rows of kvHeads*dh (extra cache capacity
// beyond seqKV is ignored), and query i attends to positions
// 0..i+(seqKV-seq), floored to the last `window` positions when window is
// positive (Gemma's sliding attention). Head h reads kv head
// h/(heads/kvHeads). One fused dispatch, dh at most 128.
func (q *GPUTensor) GroupedCausalAttention(k, v *GPUTensor, heads, kvHeads, seqKV, window int) (*GPUTensor, error) {
	if q.freed || k.freed || v.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if q.g != k.g || q.g != v.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	nd := len(q.shape)
	if nd == 0 {
		return nil, errors.New("tensai: attention needs at least 1 axis")
	}
	d := q.shape[nd-1]
	if heads <= 0 || kvHeads <= 0 || d%heads != 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("tensai: %d query heads / %d kv heads do not divide dimension %d", heads, kvHeads, d)
	}
	dh := d / heads
	if dh > 256 {
		return nil, fmt.Errorf("tensai: grouped attention head dimension %d exceeds 256", dh)
	}
	seq := q.Size() / d
	kvDim := kvHeads * dh
	if seqKV < seq {
		return nil, fmt.Errorf("tensai: grouped attention needs seqKV >= seq, got %d < %d", seqKV, seq)
	}
	if k.Size() < seqKV*kvDim || v.Size() < seqKV*kvDim {
		return nil, fmt.Errorf("tensai: kv cache of %d/%d elements is smaller than %d x %d", k.Size(), v.Size(), seqKV, kvDim)
	}
	group := heads / kvHeads
	offs := make([]uint32, 4*heads)
	for h := 0; h < heads; h++ {
		offs[4*h] = uint32(h * dh)
		offs[4*h+1] = uint32((h / group) * dh)
		offs[4*h+2] = uint32(h * dh)
	}
	rows := heads * seq
	params := [8]uint32{
		uint32(seq), uint32(seqKV), uint32(dh), uint32(d),
		uint32(rows), uint32(seqKV - seq), uint32(kvDim), uint32(window),
	}

	g := q.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(q.Size()) * 4
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOffs := g.takeBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.takeBuffer(gpuTensorUsage, outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	defer g.putBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4, bufOffs)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	entries := [6]wgpuBindGroupEntry{
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 10, buffer: bufParams, size: 32},
		{binding: 11, buffer: q.buf, size: uint64(q.Size()) * 4},
		{binding: 12, buffer: k.buf, size: uint64(k.Size()) * 4},
		{binding: 13, buffer: v.buf, size: uint64(v.Size()) * 4},
		{binding: 14, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layAttn, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.attn, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: append([]int(nil), q.shape...)}, nil
}

// GPUQ4Matrix is an int4-quantized weight matrix resident in GPU memory:
// the nibbles of a Q4Matrix packed four row-pair bytes per u32 with the
// per-(group, column) scales alongside. The kernel subtracts the nibble
// offset in registers and folds group scales at group boundaries, so a
// matvec streams an eighth of the f32 weight bytes.
type GPUQ4Matrix struct {
	g           *GPU
	buf, scales uintptr
	rows, cols  int
	words       int // u32 words per pair row: ceil(cols/4)
	freed       bool
}

// UploadQ4 packs a 4-bit quantized matrix into GPU memory. The kernel
// folds scales on 64-row groups, so other group lengths are rejected.
func (g *GPU) UploadQ4(q *Q4Matrix) (*GPUQ4Matrix, error) {
	if q.Rows == 0 || q.Cols == 0 {
		return nil, errors.New("tensai: cannot upload an empty matrix")
	}
	if q.Group != 0 && q.Group != 64 {
		return nil, fmt.Errorf("tensai: gpu int4 kernel folds 64-row groups, not %d", q.Group)
	}
	if q.ScaleMin != nil {
		return nil, fmt.Errorf("tensai: gpu int4 kernel is symmetric; min-form matrices are cpu-only")
	}
	pairs := (q.Rows + 1) / 2
	words := (q.Cols + 3) / 4
	// The CPU layout stores row quads (two nibble bytes per column); the
	// kernel walks row pairs, so repack nibble pairs on the way in.
	nib := func(i, j int) uint32 {
		if i >= q.Rows {
			return 8 // zero pad row
		}
		return uint32(q.Q[q.Index(i, j)]>>(4*(i%2))) & 0x0F
	}
	packed := make([]uint32, pairs*words)
	for i2 := 0; i2 < pairs; i2++ {
		for j := 0; j < q.Cols; j++ {
			b := nib(2*i2, j) | nib(2*i2+1, j)<<4
			packed[i2*words+j/4] |= b << (8 * (j % 4))
		}
	}
	groups := (q.Rows + 63) / 64 // q4Group
	scales := make([]float32, groups*words*4)
	for gi := 0; gi < groups; gi++ {
		for j := 0; j < q.Cols; j++ {
			scales[gi*words*4+j] = q.Scale[q.TableIndex(gi, j)]
		}
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	if err := g.checkSize(uint64(len(packed)) * 4); err != nil {
		return nil, err
	}
	buf := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(packed))*4)
	sbuf := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(scales))*4)
	if buf == 0 || sbuf == 0 {
		if buf != 0 {
			fnBufferRelease(buf)
		}
		if sbuf != 0 {
			fnBufferRelease(sbuf)
		}
		return nil, errors.New("tensai: gpu buffer allocation failed")
	}
	fnQueueWriteBuffer(g.queue, buf, 0, unsafe.Pointer(&packed[0]), uintptr(len(packed))*4)
	fnQueueWriteBuffer(g.queue, sbuf, 0, unsafe.Pointer(&scales[0]), uintptr(len(scales))*4)
	runtime.KeepAlive(&packed)
	runtime.KeepAlive(&scales)
	return &GPUQ4Matrix{g: g, buf: buf, scales: sbuf, rows: q.Rows, cols: q.Cols, words: words}, nil
}

// Shape returns (rows, cols).
func (q *GPUQ4Matrix) Shape() (int, int) { return q.rows, q.cols }

// Free releases the GPU buffers. Calling Free again is a no-op.
func (q *GPUQ4Matrix) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if q.freed {
		return
	}
	q.freed = true
	q.g.dropBuffer(q.buf)
	q.g.dropBuffer(q.scales)
}

// MatMul computes x @ Q on the GPU: x's last axis must equal the weight
// rows, and every leading axis is a batch of activation rows.
func (q *GPUQ4Matrix) MatMul(x *GPUTensor) (*GPUTensor, error) {
	if q.freed || x.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if q.g != x.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	n := len(x.shape)
	if n == 0 || x.shape[n-1] != q.rows {
		return nil, fmt.Errorf("tensai: gpu q4matmul shape mismatch: %v @ %dx%d", x.shape, q.rows, q.cols)
	}
	m := x.Size() / q.rows
	if m > 65535 {
		return nil, fmt.Errorf("tensai: gpu q4matmul batch of %d rows exceeds 65535", m)
	}
	outShape := append(append([]int(nil), x.shape[:n-1]...), q.cols)

	g := q.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(m*q.cols) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	groups := (q.rows + 63) / 64
	params := [8]uint32{
		uint32(q.rows), uint32(q.cols), uint32(q.words), uint32(m),
		uint32(groups), 0, 0, 0,
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOut := g.takeBuffer(gpuTensorUsage, outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)

	pairs := (q.rows + 1) / 2
	entries := [5]wgpuBindGroupEntry{
		{binding: 29, buffer: bufParams, size: 32},
		{binding: 30, buffer: q.buf, size: uint64(pairs*q.words) * 4},
		{binding: 31, buffer: q.scales, size: uint64(groups*q.words*4) * 4},
		{binding: 32, buffer: x.buf, size: uint64(x.Size()) * 4},
		{binding: 33, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layQ4matmul, entries[:])
	runtime.KeepAlive(&entries)

	err := g.dispatch(g.pipes.q4matmul, bindGroup,
		uint32((q.words+15)/16), uint32(m), 1)
	if err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &GPUTensor{g: g, buf: bufOut, shape: outShape}, nil
}

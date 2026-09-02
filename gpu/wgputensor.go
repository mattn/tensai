//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

// Tensor: tensors resident in Device memory, so a weight is uploaded once
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

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/dims"
	"github.com/mattn/tensai/internal/kernels"
	"github.com/mattn/tensai/internal/workpool"
	"github.com/mattn/tensai/quant"
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

// matmul_l is the 64x64 block again, but with the accumulators as four
// vec4 registers instead of an indexed array. A private array that the
// compiler cannot fully unroll lands in scratch memory rather than
// registers, and at 16 accumulators per thread that is the difference
// between the inner loop running out of registers and not.
@compute @workgroup_size(16, 16, 1)
fn matmul_l(@builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
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
            let aBase = lid.y * TT;
            let bBase = kk * BLK + lid.x * TT;
            let bv = vec4<f32>(tileB[bBase], tileB[bBase + 1u], tileB[bBase + 2u], tileB[bBase + 3u]);
            acc0 = fma(vec4<f32>(tileA[aBase * TILE + kk]), bv, acc0);
            acc1 = fma(vec4<f32>(tileA[(aBase + 1u) * TILE + kk]), bv, acc1);
            acc2 = fma(vec4<f32>(tileA[(aBase + 2u) * TILE + kk]), bv, acc2);
            acc3 = fma(vec4<f32>(tileA[(aBase + 3u) * TILE + kk]), bv, acc3);
        }
        workgroupBarrier();
    }
    storeRow(offC, rowBase, colBase, acc0);
    storeRow(offC, rowBase + 1u, colBase, acc1);
    storeRow(offC, rowBase + 2u, colBase, acc2);
    storeRow(offC, rowBase + 3u, colBase, acc3);
}

// matmul_v4 is matmul_l with the tile loads widened: the same buffers are
// bound again as vec4 arrays, so each thread brings in four values with one
// instruction instead of four. It needs k and n to be multiples of four and
// the row strides and batch offsets to be too, which the host checks before
// picking it.
@group(0) @binding(56) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(57) var<storage, read> b4: array<vec4<f32>>;

@compute @workgroup_size(16, 16, 1)
fn matmul_v4(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    // Each of the 256 threads fills one vec4 of each tile: a is four k
    // values of one row, b is four n values of one k.
    let aRow = li / 4u;          // 0..63 within the block
    let aQuad = (li % 4u) * 4u;  // k offset within the tile
    let bRow = li / 16u;         // 0..15, the k index within the tile
    let bQuad = (li % 16u) * 4u; // n offset within the block
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let gr = wid.y * BLK + aRow;
        let gc = t * TILE + aQuad;
        var av = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gr < p.m && gc < p.k) {
            av = a4[(offA + gr * p.lda + gc) / 4u];
        }
        tileA[aRow * TILE + aQuad] = av.x;
        tileA[aRow * TILE + aQuad + 1u] = av.y;
        tileA[aRow * TILE + aQuad + 2u] = av.z;
        tileA[aRow * TILE + aQuad + 3u] = av.w;

        let gkr = t * TILE + bRow;
        let gbc = wid.x * BLK + bQuad;
        var bvv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gkr < p.k && gbc < p.n) {
            bvv = b4[(offB + gkr * p.ldb + gbc) / 4u];
        }
        tileB[bRow * BLK + bQuad] = bvv.x;
        tileB[bRow * BLK + bQuad + 1u] = bvv.y;
        tileB[bRow * BLK + bQuad + 2u] = bvv.z;
        tileB[bRow * BLK + bQuad + 3u] = bvv.w;
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            let aBase = lid.y * TT;
            let bBase = kk * BLK + lid.x * TT;
            let bv = vec4<f32>(tileB[bBase], tileB[bBase + 1u], tileB[bBase + 2u], tileB[bBase + 3u]);
            acc0 = fma(vec4<f32>(tileA[aBase * TILE + kk]), bv, acc0);
            acc1 = fma(vec4<f32>(tileA[(aBase + 1u) * TILE + kk]), bv, acc1);
            acc2 = fma(vec4<f32>(tileA[(aBase + 2u) * TILE + kk]), bv, acc2);
            acc3 = fma(vec4<f32>(tileA[(aBase + 3u) * TILE + kk]), bv, acc3);
        }
        workgroupBarrier();
    }
    storeRow(offC, rowBase, colBase, acc0);
    storeRow(offC, rowBase + 1u, colBase, acc1);
    storeRow(offC, rowBase + 2u, colBase, acc2);
    storeRow(offC, rowBase + 3u, colBase, acc3);
}

// matmul_v4t and matmul_v4tn widen the loads of the transposed modes. The
// operand that is read transposed is fetched four values along k (or along
// m) at a time and scattered into the tile, which is what makes the wide
// load legal there.
@compute @workgroup_size(16, 16, 1)
fn matmul_v4t(@builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let aRow = li / 4u;
    let aQuad = (li % 4u) * 4u;
    let bCol = li % BLK;        // the n index within the block
    let bQuad = (li / BLK) * 4u; // four k values per thread
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let gr = wid.y * BLK + aRow;
        let gc = t * TILE + aQuad;
        var av = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gr < p.m && gc < p.k) {
            av = a4[(offA + gr * p.lda + gc) / 4u];
        }
        tileA[aRow * TILE + aQuad] = av.x;
        tileA[aRow * TILE + aQuad + 1u] = av.y;
        tileA[aRow * TILE + aQuad + 2u] = av.z;
        tileA[aRow * TILE + aQuad + 3u] = av.w;

        // b holds n rows of length k: four k values of one row, scattered
        // down the tile's k axis.
        let gbc = wid.x * BLK + bCol;
        let gkr = t * TILE + bQuad;
        var bw = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gbc < p.n && gkr < p.k) {
            bw = b4[(offB + gbc * p.ldb + gkr) / 4u];
        }
        tileB[bQuad * BLK + bCol] = bw.x;
        tileB[(bQuad + 1u) * BLK + bCol] = bw.y;
        tileB[(bQuad + 2u) * BLK + bCol] = bw.z;
        tileB[(bQuad + 3u) * BLK + bCol] = bw.w;
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            let aBase = lid.y * TT;
            let bBase = kk * BLK + lid.x * TT;
            let bv = vec4<f32>(tileB[bBase], tileB[bBase + 1u], tileB[bBase + 2u], tileB[bBase + 3u]);
            acc0 = fma(vec4<f32>(tileA[aBase * TILE + kk]), bv, acc0);
            acc1 = fma(vec4<f32>(tileA[(aBase + 1u) * TILE + kk]), bv, acc1);
            acc2 = fma(vec4<f32>(tileA[(aBase + 2u) * TILE + kk]), bv, acc2);
            acc3 = fma(vec4<f32>(tileA[(aBase + 3u) * TILE + kk]), bv, acc3);
        }
        workgroupBarrier();
    }
    storeRow(offC, rowBase, colBase, acc0);
    storeRow(offC, rowBase + 1u, colBase, acc1);
    storeRow(offC, rowBase + 2u, colBase, acc2);
    storeRow(offC, rowBase + 3u, colBase, acc3);
}

@compute @workgroup_size(16, 16, 1)
fn matmul_v4tn(@builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let aCol = li % TILE;         // the k index within the tile
    let aQuad = (li / TILE) * 4u; // four output rows per thread
    let bRow = li / 16u;
    let bQuad = (li % 16u) * 4u;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        // a holds k rows of length m: four m values of one k, scattered
        // down the tile's m axis.
        let gc = t * TILE + aCol;
        let gr = wid.y * BLK + aQuad;
        var av = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gc < p.k && gr < p.m) {
            av = a4[(offA + gc * p.lda + gr) / 4u];
        }
        tileA[aQuad * TILE + aCol] = av.x;
        tileA[(aQuad + 1u) * TILE + aCol] = av.y;
        tileA[(aQuad + 2u) * TILE + aCol] = av.z;
        tileA[(aQuad + 3u) * TILE + aCol] = av.w;

        let gkr = t * TILE + bRow;
        let gbc = wid.x * BLK + bQuad;
        var bvv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gkr < p.k && gbc < p.n) {
            bvv = b4[(offB + gkr * p.ldb + gbc) / 4u];
        }
        tileB[bRow * BLK + bQuad] = bvv.x;
        tileB[bRow * BLK + bQuad + 1u] = bvv.y;
        tileB[bRow * BLK + bQuad + 2u] = bvv.z;
        tileB[bRow * BLK + bQuad + 3u] = bvv.w;
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            let aBase = lid.y * TT;
            let bBase = kk * BLK + lid.x * TT;
            let bv = vec4<f32>(tileB[bBase], tileB[bBase + 1u], tileB[bBase + 2u], tileB[bBase + 3u]);
            acc0 = fma(vec4<f32>(tileA[aBase * TILE + kk]), bv, acc0);
            acc1 = fma(vec4<f32>(tileA[(aBase + 1u) * TILE + kk]), bv, acc1);
            acc2 = fma(vec4<f32>(tileA[(aBase + 2u) * TILE + kk]), bv, acc2);
            acc3 = fma(vec4<f32>(tileA[(aBase + 3u) * TILE + kk]), bv, acc3);
        }
        workgroupBarrier();
    }
    storeRow(offC, rowBase, colBase, acc0);
    storeRow(offC, rowBase + 1u, colBase, acc1);
    storeRow(offC, rowBase + 2u, colBase, acc2);
    storeRow(offC, rowBase + 3u, colBase, acc3);
}

// matmul_lt and matmul_ltn are the same register-tiled body with one
// operand read transposed while its tile is loaded, so the inner loop is
// shared with matmul_l.
@compute @workgroup_size(16, 16, 1)
fn matmul_lt(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
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
            // b holds n rows of length k; transpose it into the tile.
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
            let aBase = lid.y * TT;
            let bBase = kk * BLK + lid.x * TT;
            let bv = vec4<f32>(tileB[bBase], tileB[bBase + 1u], tileB[bBase + 2u], tileB[bBase + 3u]);
            acc0 = fma(vec4<f32>(tileA[aBase * TILE + kk]), bv, acc0);
            acc1 = fma(vec4<f32>(tileA[(aBase + 1u) * TILE + kk]), bv, acc1);
            acc2 = fma(vec4<f32>(tileA[(aBase + 2u) * TILE + kk]), bv, acc2);
            acc3 = fma(vec4<f32>(tileA[(aBase + 3u) * TILE + kk]), bv, acc3);
        }
        workgroupBarrier();
    }
    storeRow(offC, rowBase, colBase, acc0);
    storeRow(offC, rowBase + 1u, colBase, acc1);
    storeRow(offC, rowBase + 2u, colBase, acc2);
    storeRow(offC, rowBase + 3u, colBase, acc3);
}

@compute @workgroup_size(16, 16, 1)
fn matmul_ltn(@builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let batch = wid.z;
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    let offC = offs[batch].z;
    let rowBase = wid.y * BLK + lid.y * TT;
    let colBase = wid.x * BLK + lid.x * TT;
    let li = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = li * 4u + i;
            let ar = idx / TILE;
            let ac = idx % TILE;
            let gr = wid.y * BLK + ar;
            let gc = t * TILE + ac;
            // a holds k rows of length m; transpose it into the tile.
            if (gr < p.m && gc < p.k) {
                tileA[idx] = a[offA + gc * p.lda + gr];
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
            let aBase = lid.y * TT;
            let bBase = kk * BLK + lid.x * TT;
            let bv = vec4<f32>(tileB[bBase], tileB[bBase + 1u], tileB[bBase + 2u], tileB[bBase + 3u]);
            acc0 = fma(vec4<f32>(tileA[aBase * TILE + kk]), bv, acc0);
            acc1 = fma(vec4<f32>(tileA[(aBase + 1u) * TILE + kk]), bv, acc1);
            acc2 = fma(vec4<f32>(tileA[(aBase + 2u) * TILE + kk]), bv, acc2);
            acc3 = fma(vec4<f32>(tileA[(aBase + 3u) * TILE + kk]), bv, acc3);
        }
        workgroupBarrier();
    }
    storeRow(offC, rowBase, colBase, acc0);
    storeRow(offC, rowBase + 1u, colBase, acc1);
    storeRow(offC, rowBase + 2u, colBase, acc2);
    storeRow(offC, rowBase + 3u, colBase, acc3);
}

// storeRow writes four contiguous outputs, skipping the ones a ragged edge
// puts past the end.
fn storeRow(offC: u32, r: u32, c: u32, v: vec4<f32>) {
    if (r >= p.m) {
        return;
    }
    let base = offC + r * p.ldc + c;
    if (c + 3u < p.n) {
        outv[base] = v.x;
        outv[base + 1u] = v.y;
        outv[base + 2u] = v.z;
        outv[base + 3u] = v.w;
        return;
    }
    if (c < p.n) { outv[base] = v.x; }
    if (c + 1u < p.n) { outv[base + 1u] = v.y; }
    if (c + 2u < p.n) { outv[base + 2u] = v.z; }
    if (c + 3u < p.n) { outv[base + 3u] = v.w; }
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

// matmul_tn / matmul_tns read a transposed instead: a holds k rows of
// length m (row stride lda), so the product is a^T * b. Training needs it
// for the weight gradient, which is the input activation transposed times
// the output gradient.

@compute @workgroup_size(16, 16, 1)
fn matmul_tn(@builtin(local_invocation_id) lid: vec3<u32>,
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
            // a holds k rows of length m; transpose while loading.
            if (gr < p.m && gc < p.k) {
                tileA[idx] = a[offA + gc * p.lda + gr];
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
fn matmul_tns(@builtin(global_invocation_id) gid: vec3<u32>,
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
            tileSA[lid.y * TILE + lid.x] = a[offA + ak * p.lda + row];
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
// One query head, staged for the whole workgroup. Gemma 4's global
// layers are 512 wide, which is the widest head this kernel takes; the
// grouped kernel below stages a whole group instead and so tops out at
// a narrower head.
var<workgroup> qrow: array<f32, 512>;
var<workgroup> ap_sc: array<f32, 64>;
var<workgroup> ap_red: array<f32, 64>;

// attn_causal_g is attn_causal for one query row against one KV head,
// producing every query head of the group that shares it: the K and V
// rows stream once per group instead of once per head. Loops over the
// group run a fixed eight iterations masked by the real group (grp =
// d/dkv) so per-head state keeps static indices, and each head repeats
// attn_causal's arithmetic in its exact order, so outputs match it bit
// for bit.
var<workgroup> qrow_g: array<f32, 2048>;
var<workgroup> ap_sc_g: array<f32, 512>;
var<workgroup> ap_red_g: array<f32, 512>;

@compute @workgroup_size(64, 1, 1)
fn attn_causal_g(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= ap.rows) {
        return;
    }
    let grp = ap.d / ap.dkv;
    let bh = row / ap.seqQ;
    let qi = row % ap.seqQ;
    let offQ = offs[bh].x;
    let offKV = offs[bh].y;
    let offO = offs[bh].z;
    let t = lid.x;
    for (var c = t; c < grp * ap.dh; c = c + 64u) {
        qrow_g[c] = aq[offQ + qi * ap.d + c];
    }
    workgroupBarrier();
    let limit = qi + ap.off + 1u;
    // Sliding-window layers (Gemma) see only the last window positions.
    var start = 0u;
    if (ap.window > 0u && limit > ap.window) {
        start = limit - ap.window;
    }
    let scale = inverseSqrt(f32(ap.dh));
    var m: array<f32, 8>;
    var l: array<f32, 8>;
    var acc: array<vec4<f32>, 8>;
    for (var h = 0u; h < 8u; h = h + 1u) {
        m[h] = -3.40282e38;
    }
    let tiles = (limit + AT - 1u) / AT;
    for (var tt = start / AT; tt < tiles; tt = tt + 1u) {
        // Lane t scores kv position tt*64+t for every head at once.
        let j = tt * AT + t;
        let valid = j >= start && j < limit;
        var dot: array<f32, 8>;
        for (var h = 0u; h < 8u; h = h + 1u) {
            dot[h] = 0.0;
        }
        if (valid) {
            for (var c = 0u; c < ap.dh; c = c + 1u) {
                let kv = ak[offKV + j * ap.dkv + c];
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        dot[h] = dot[h] + qrow_g[h * ap.dh + c] * kv;
                    }
                }
            }
        }
        for (var h = 0u; h < 8u; h = h + 1u) {
            if (h < grp) {
                var s = -3.40282e38;
                if (valid) {
                    s = dot[h] * scale;
                }
                dot[h] = s;
                ap_red_g[h * 64u + t] = s;
            }
        }
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) {
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        ap_red_g[h * 64u + t] = max(ap_red_g[h * 64u + t], ap_red_g[h * 64u + t + r]);
                    }
                }
            }
            workgroupBarrier();
        }
        var mNew: array<f32, 8>;
        var p: array<f32, 8>;
        for (var h = 0u; h < 8u; h = h + 1u) {
            p[h] = 0.0;
            if (h < grp) {
                mNew[h] = max(m[h], ap_red_g[h * 64u]);
                if (valid) {
                    p[h] = exp(dot[h] - mNew[h]);
                }
            }
        }
        workgroupBarrier();
        for (var h = 0u; h < 8u; h = h + 1u) {
            if (h < grp) {
                ap_sc_g[h * 64u + t] = p[h];
                ap_red_g[h * 64u + t] = p[h];
            }
        }
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) {
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        ap_red_g[h * 64u + t] = ap_red_g[h * 64u + t] + ap_red_g[h * 64u + t + r];
                    }
                }
            }
            workgroupBarrier();
        }
        for (var h = 0u; h < 8u; h = h + 1u) {
            if (h < grp) {
                // exp underflows to zero on the first tile, where m is -inf-like.
                let rescale = exp(m[h] - mNew[h]);
                l[h] = l[h] * rescale + ap_red_g[h * 64u];
                m[h] = mNew[h];
                acc[h] = acc[h] * rescale;
            }
        }
        // Lane t accumulates output channels t, t+64, t+128, t+192; the
        // V row loads once and feeds every head.
        let jEnd = min(limit, tt * AT + AT);
        for (var jj = max(start, tt * AT); jj < jEnd; jj = jj + 1u) {
            var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (t < ap.dh) {
                vv.x = av[offKV + jj * ap.dkv + t];
            }
            if (64u + t < ap.dh) {
                vv.y = av[offKV + jj * ap.dkv + 64u + t];
            }
            if (128u + t < ap.dh) {
                vv.z = av[offKV + jj * ap.dkv + 128u + t];
            }
            if (192u + t < ap.dh) {
                vv.w = av[offKV + jj * ap.dkv + 192u + t];
            }
            for (var h = 0u; h < 8u; h = h + 1u) {
                if (h < grp) {
                    acc[h] = acc[h] + ap_sc_g[h * 64u + jj - tt * AT] * vv;
                }
            }
        }
        workgroupBarrier();
    }
    for (var h = 0u; h < 8u; h = h + 1u) {
        if (h < grp) {
            let o = offO + qi * ap.d + h * ap.dh;
            if (t < ap.dh) {
                aout[o + t] = acc[h].x / l[h];
            }
            if (64u + t < ap.dh) {
                aout[o + 64u + t] = acc[h].y / l[h];
            }
            if (128u + t < ap.dh) {
                aout[o + 128u + t] = acc[h].z / l[h];
            }
            if (192u + t < ap.dh) {
                aout[o + 192u + t] = acc[h].w / l[h];
            }
        }
    }
}

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
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
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
        // Lane t now accumulates output channels t, t+64, and so on to
        // t+448: eight of them, which is what carries a 512-wide head.
        acc0 = acc0 * rescale;
        acc1 = acc1 * rescale;
        acc2 = acc2 * rescale;
        acc3 = acc3 * rescale;
        acc4 = acc4 * rescale;
        acc5 = acc5 * rescale;
        acc6 = acc6 * rescale;
        acc7 = acc7 * rescale;
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
            if (256u + t < ap.dh) {
                acc4 = acc4 + pj * av[offKV + jj * ap.dkv + 256u + t];
            }
            if (320u + t < ap.dh) {
                acc5 = acc5 + pj * av[offKV + jj * ap.dkv + 320u + t];
            }
            if (384u + t < ap.dh) {
                acc6 = acc6 + pj * av[offKV + jj * ap.dkv + 384u + t];
            }
            if (448u + t < ap.dh) {
                acc7 = acc7 + pj * av[offKV + jj * ap.dkv + 448u + t];
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
    if (256u + t < ap.dh) {
        aout[offO + qi * ap.d + 256u + t] = acc4 / l;
    }
    if (320u + t < ap.dh) {
        aout[offO + qi * ap.d + 320u + t] = acc5 / l;
    }
    if (384u + t < ap.dh) {
        aout[offO + qi * ap.d + 384u + t] = acc6 / l;
    }
    if (448u + t < ap.dh) {
        aout[offO + qi * ap.d + 448u + t] = acc7 / l;
    }
}

// qmatmul multiplies f32 activation rows by an int8 weight matrix that
// stays packed in Device memory, four weights per u32, dequantizing in
// registers: the weight traffic — which dominates a decode matvec — is a
// quarter of the f32 kernel's. A 256-lane workgroup covers 16 packed words
// (64 output columns) of one activation row, with each word's rows split
// 16 ways: lane (rsub, wsub) strides the rows by 16 reading 16 adjacent
// words per step (coalesced, 64 bytes), and a shared-memory reduction
// folds the row splits before the guarded store. The split keeps a decode
// matvec — parallelism = output columns only — wide enough to fill the
// device.
struct QMParams { rows: u32, cols: u32, words: u32, m: u32, flags: u32, eps: f32, pad1: u32, pad2: u32 }
@group(0) @binding(15) var<uniform> qp: QMParams;
@group(0) @binding(16) var<storage, read> qwt: array<u32>;
@group(0) @binding(17) var<storage, read> qsc: array<f32>;
@group(0) @binding(18) var<storage, read> qxv: array<f32>;
@group(0) @binding(19) var<storage, read_write> qov: array<f32>;
@group(0) @binding(34) var<storage, read> qbias: array<f32>;
@group(0) @binding(39) var<storage, read> qnormw: array<f32>;
@group(0) @binding(45) var<storage, read> qascr: array<f32>;

const QWG = 16u;    // packed words per workgroup
const QSPLIT = 16u; // row splits sharing one word
const QXS = 2048u;  // widest activation row the norm prologue stages
var<workgroup> qred: array<vec4<f32>, 256>;
var<workgroup> qxs: array<f32, QXS>;

@compute @workgroup_size(256, 1, 1)
fn qmatmul(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = wid.x * QWG + lid.x % QWG;
    let rsub = lid.x / QWG;
    let r = wid.y;
    let xbase = r * qp.rows;
    // Flag 4 folds the row's RMS normalization in as a prologue: every
    // workgroup redundantly computes the same inverse rms — a pass over
    // one row is nothing next to the weight stream — and stages the
    // normalized row in shared memory, sparing decode a dependent
    // dispatch per norm without touching the streaming loop's flops.
    // The stride and the reduction tree mirror rmsnorm_row exactly, so
    // the products match the standalone dispatch bit for bit. Callers
    // set the flag only when the row fits QXS.
    if ((qp.flags & 4u) != 0u) {
        var ss = 0.0;
        for (var i = lid.x; i < qp.rows; i = i + 256u) {
            let v = qxv[xbase + i];
            ss = ss + v * v;
        }
        qred[lid.x].x = ss;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (lid.x < s) { qred[lid.x].x = qred[lid.x].x + qred[lid.x + s].x; }
            workgroupBarrier();
        }
        let inv = inverseSqrt(qred[0].x / f32(qp.rows) + qp.eps);
        workgroupBarrier(); // qred is reused by the column reduction below
        for (var i = lid.x; i < qp.rows; i = i + 256u) {
            qxs[i] = qxv[xbase + i] * inv * qnormw[i];
        }
        workgroupBarrier();
    }
    // Flag 8 stages attn_split_gh's slabs instead of an activation row:
    // the output projection folds the softmax combine that attn_reduce_g
    // used to run as its own dependent dispatch. pad1 carries the slab
    // count, pad2 the head size and group; every max, rescale, and sum
    // runs in the same order as the standalone reduce, so the staged row
    // matches it bit for bit.
    if ((qp.flags & 8u) != 0u) {
        let dh = qp.pad2 & 0xffffu;
        let grp = qp.pad2 >> 16u;
        let slabs = qp.pad1;
        let stride = 8u * dh + 16u;
        for (var i = lid.x; i < qp.rows; i = i + 256u) {
            let h = i / dh;
            let c = i - h * dh;
            let kvh = h / grp;
            let hg = h - kvh * grp;
            let base = kvh * slabs * stride;
            var mh = -3.40282e38;
            for (var s = 0u; s < slabs; s = s + 1u) {
                mh = max(mh, qascr[base + s * stride + 8u * dh + hg]);
            }
            var lh = 0.0;
            for (var s = 0u; s < slabs; s = s + 1u) {
                let ms = qascr[base + s * stride + 8u * dh + hg];
                lh = lh + qascr[base + s * stride + 8u * dh + 8u + hg] * exp(ms - mh);
            }
            var o = 0.0;
            for (var s = 0u; s < slabs; s = s + 1u) {
                let ms = qascr[base + s * stride + 8u * dh + hg];
                o = o + qascr[base + s * stride + hg * dh + c] * exp(ms - mh);
            }
            qxs[i] = o / lh;
        }
        workgroupBarrier();
    }
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (w < qp.words) {
        if ((qp.flags & 12u) != 0u) { // a prologue staged the row
            for (var i = rsub; i < qp.rows; i = i + QSPLIT) {
                let xv = qxs[i];
                let pw = qwt[i * qp.words + w];
                acc = acc + xv * vec4<f32>(
                    f32(i32(pw << 24u) >> 24u),
                    f32(i32(pw << 16u) >> 24u),
                    f32(i32(pw << 8u) >> 24u),
                    f32(i32(pw) >> 24u));
            }
        } else {
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
                var o = sum[l] * qsc[j + l];
                if ((qp.flags & 1u) != 0u) {
                    o = o + qbias[j + l];
                }
                if ((qp.flags & 2u) != 0u) {
                    o = o + qov[r * qp.cols + j + l];
                }
                qov[r * qp.cols + j + l] = o;
            }
        }
    }
}

// qmatmul_b is qmatmul's batched twin for prefill: a workgroup still
// covers 16 packed words split 16 ways, but accumulates QROWS activation
// rows per weight word, so the weight stream a matvec pays once per row
// amortizes QROWS-fold. Each row keeps qmatmul's stride-16 accumulation
// and reduction order, so its output matches the matvec bit for bit; the
// shared reduction reuses qred once per row.
const QROWS = 8u;

@compute @workgroup_size(256, 1, 1)
fn qmatmul_b(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = wid.x * QWG + lid.x % QWG;
    let rsub = lid.x / QWG;
    let r0 = wid.y * QROWS;
    let jm = min(QROWS, qp.m - r0);
    var acc: array<vec4<f32>, 8>;
    if (w < qp.words) {
        for (var i = rsub; i < qp.rows; i = i + QSPLIT) {
            let pw = qwt[i * qp.words + w];
            let wv = vec4<f32>(
                f32(i32(pw << 24u) >> 24u),
                f32(i32(pw << 16u) >> 24u),
                f32(i32(pw << 8u) >> 24u),
                f32(i32(pw) >> 24u));
            for (var j = 0u; j < jm; j = j + 1u) {
                acc[j] = acc[j] + qxv[(r0 + j) * qp.rows + i] * wv;
            }
        }
    }
    for (var j = 0u; j < QROWS; j = j + 1u) {
        qred[lid.x] = acc[j];
        workgroupBarrier();
        if (lid.x < QWG && w < qp.words && j < jm) {
            var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            for (var v = 0u; v < QSPLIT; v = v + 1u) {
                sum = sum + qred[v * QWG + lid.x];
            }
            let jc = w * 4u;
            for (var l = 0u; l < 4u; l = l + 1u) {
                if (jc + l < qp.cols) {
                    var o = sum[l] * qsc[jc + l];
                    if ((qp.flags & 1u) != 0u) {
                        o = o + qbias[jc + l];
                    }
                    if ((qp.flags & 2u) != 0u) {
                        o = o + qov[(r0 + j) * qp.cols + jc + l];
                    }
                    qov[(r0 + j) * qp.cols + jc + l] = o;
                }
            }
        }
        workgroupBarrier();
    }
}


// qmatmul_t is the tiled GEMM form for large batches: a workgroup
// produces a 64-column x 32-row output tile, staging both operands in
// shared memory per 64-deep K slice — activations cooperatively, the
// packed weights dequantized once into a vec4 tile — so each weight
// byte streams from memory rows/32 times instead of rows/8 and no
// cross-lane reduction is needed. Accumulation is k-sequential per
// output, so results sit within fp32 rounding of the split kernels
// rather than matching them bit for bit.
const QTR = 64u; // output rows per workgroup
const QTK = 32u; // K slice depth
var<workgroup> qta: array<f32, 2048>;      // QTR x QTK activations
var<workgroup> qtw: array<vec4<f32>, 512>; // QTK x 16 dequantized words

@compute @workgroup_size(256, 1, 1)
fn qmatmul_t(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let wsub = lid.x % QWG;
    let rsub = lid.x / QWG;
    let w = wid.x * QWG + wsub;
    let r0 = wid.y * QTR;
    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let ktiles = (qp.rows + QTK - 1u) / QTK;
    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {
        let kbase = kt * QTK;
        // Stage the activation tile, each lane eight consecutive k.
        for (var s = 0u; s < 8u; s = s + 1u) {
            let idx = lid.x * 8u + s;
            let rr = idx / QTK;
            let kk = idx % QTK;
            var a = 0.0;
            if (r0 + rr < qp.m && kbase + kk < qp.rows) {
                a = qxv[(r0 + rr) * qp.rows + kbase + kk];
            }
            qta[idx] = a;
        }
        // Stage the weight tile, each lane expanding two packed words.
        for (var s = 0u; s < 2u; s = s + 1u) {
            let idx = lid.x * 2u + s;
            let kk = idx / QWG;
            let ww = wid.x * QWG + idx % QWG;
            var wv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (kbase + kk < qp.rows && ww < qp.words) {
                let pw = qwt[(kbase + kk) * qp.words + ww];
                wv = vec4<f32>(
                    f32(i32(pw << 24u) >> 24u),
                    f32(i32(pw << 16u) >> 24u),
                    f32(i32(pw << 8u) >> 24u),
                    f32(i32(pw) >> 24u));
            }
            qtw[idx] = wv;
        }
        workgroupBarrier();
        let kmax = min(QTK, qp.rows - kbase);
        let ra = rsub * 4u;
        for (var i = 0u; i < kmax; i = i + 1u) {
            let wv = qtw[i * QWG + wsub];
            acc0 = acc0 + qta[ra * QTK + i] * wv;
            acc1 = acc1 + qta[(ra + 1u) * QTK + i] * wv;
            acc2 = acc2 + qta[(ra + 2u) * QTK + i] * wv;
            acc3 = acc3 + qta[(ra + 3u) * QTK + i] * wv;
        }
        workgroupBarrier();
    }
    if (w < qp.words) {
        let j = w * 4u;
        let rA = r0 + rsub * 4u;
        for (var l = 0u; l < 4u; l = l + 1u) {
            if (j + l < qp.cols) {
                let sc = qsc[j + l];
                var bv = 0.0;
                if ((qp.flags & 1u) != 0u) {
                    bv = qbias[j + l];
                }
                let vv = vec4<f32>(acc0[l], acc1[l], acc2[l], acc3[l]) * sc + vec4<f32>(bv, bv, bv, bv);
                for (var r = 0u; r < 4u; r = r + 1u) {
                    if (rA + r < qp.m) {
                        let idx = (rA + r) * qp.cols + j + l;
                        var o = vv[r];
                        if ((qp.flags & 2u) != 0u) {
                            o = o + qov[idx];
                        }
                        qov[idx] = o;
                    }
                }
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

// The training kernels. A backward pass needs the element-wise arithmetic
// and activations an inference pass never runs on their own, the column
// sums a broadcast operand's gradient reduces to, and the optimizer update
// itself -- without them a training graph has to come back to the host
// between every product.
//
// Operands broadcast cyclically: an operand shorter than the output
// repeats, which covers the trailing-axis broadcasts training uses (a bias
// over rows, a per-feature gain over a batch). Anything else is rejected
// on the Go side.
struct TrainParams {
    count: u32, aCount: u32, bCount: u32, mode: u32,
    rows: u32, lr: f32, beta1: f32, beta2: f32,
    rc1: f32, rc2: f32, eps: f32, decay: f32,
}
@group(0) @binding(46) var<uniform> tp: TrainParams;
@group(0) @binding(47) var<storage, read> ta: array<f32>;
@group(0) @binding(48) var<storage, read> tb: array<f32>;
@group(0) @binding(49) var<storage, read_write> tout: array<f32>;
@group(0) @binding(50) var<storage, read_write> tm: array<f32>;
@group(0) @binding(51) var<storage, read_write> tv: array<f32>;

// erff matches the CPU kernels, which use the exact error function rather
// than the tanh approximation the inference GELU takes: Abramowitz and
// Stegun 7.1.26, good to about 1.5e-7.
fn erff(x: f32) -> f32 {
    let s = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
}

fn act_of(x: f32, mode: u32) -> f32 {
    switch mode {
        case 0u: { return max(x, 0.0); }                        // relu
        case 1u: { return tanh(x); }
        case 2u: { return 1.0 / (1.0 + exp(-x)); }              // sigmoid
        default: { return 0.5 * x * (1.0 + erff(x * 0.7071067811865476)); } // gelu
    }
}

fn act_grad_of(x: f32, mode: u32) -> f32 {
    switch mode {
        case 0u: {
            if (x > 0.0) { return 1.0; }
            return 0.0;
        }
        case 1u: {
            let y = tanh(x);
            return 1.0 - y * y;
        }
        case 2u: {
            let y = 1.0 / (1.0 + exp(-x));
            return y * (1.0 - y);
        }
        default: {
            return 0.5 * (1.0 + erff(x * 0.7071067811865476)) +
                x * 0.3989422804014327 * exp(-0.5 * x * x);
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn bin_op(@builtin(workgroup_id) wg: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx >= tp.count) {
        return;
    }
    let x = ta[idx % tp.aCount];
    let y = tb[idx % tp.bCount];
    switch tp.mode {
        case 0u: { tout[idx] = x + y; }
        case 1u: { tout[idx] = x - y; }
        case 2u: { tout[idx] = x * y; }
        default: { tout[idx] = x / y; }
    }
}

@compute @workgroup_size(256, 1, 1)
fn act_fwd(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < tp.count) {
        tout[idx] = act_of(ta[idx], tp.mode);
    }
}

// act_bwd turns the output gradient in tb into the input gradient, given
// the pre-activation input in ta.
@compute @workgroup_size(256, 1, 1)
fn act_bwd(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < tp.count) {
        tout[idx] = tb[idx] * act_grad_of(ta[idx], tp.mode);
    }
}

// sum_cols reduces a (rows, cols) tensor to one row: the gradient of a
// value that was broadcast over the rows. One thread per column.
@compute @workgroup_size(256, 1, 1)
fn sum_cols(@builtin(workgroup_id) wg: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (col >= tp.count) {
        return;
    }
    var sum = 0.0;
    for (var r = 0u; r < tp.rows; r = r + 1u) {
        sum = sum + ta[r * tp.count + col];
    }
    tout[col] = sum;
}

// adam_step applies one Adam update to the weights in tout, with the
// gradient in ta and the moment estimates in tm and tv. decay is AdamW's
// decoupled weight decay; the bias corrections rc1 and rc2 come from the
// host, which counts the steps.
@compute @workgroup_size(256, 1, 1)
fn adam_step(@builtin(workgroup_id) wg: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx >= tp.count) {
        return;
    }
    let g = ta[idx];
    let m = tp.beta1 * tm[idx] + (1.0 - tp.beta1) * g;
    let v = tp.beta2 * tv[idx] + (1.0 - tp.beta2) * g * g;
    tm[idx] = m;
    tv[idx] = v;
    let w = tout[idx];
    tout[idx] = w - tp.lr * (m * tp.rc1 / (sqrt(v * tp.rc2) + tp.eps) + tp.decay * w);
}

@group(0) @binding(52) var<storage, read> tc: array<f32>;
@group(0) @binding(53) var<uniform> pmp: PermuteParams;

struct PermuteParams {
    count: u32, rank: u32, pad0: u32, pad1: u32,
    shape: vec4<u32>,  // output shape, in the first rank lanes
    stride: vec4<u32>, // source stride for each output axis
}

var<workgroup> lred: array<f32, 256>;
var<workgroup> lred2: array<f32, 256>;

// rowSum reduces one value per thread to a workgroup total in lred.
fn rowSum(lid: u32, v: f32) -> f32 {
    lred[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { lred[lid] = lred[lid] + lred[lid + s]; }
        workgroupBarrier();
    }
    let total = lred[0];
    workgroupBarrier();
    return total;
}

// ln_stats returns the mean and inverse standard deviation of one row.
fn ln_stats(base: u32, cols: u32, lid: u32) -> vec2<f32> {
    var sum = 0.0;
    for (var i = lid; i < cols; i = i + 256u) {
        sum = sum + ta[base + i];
    }
    let mean = rowSum(lid, sum) / f32(cols);
    var sq = 0.0;
    for (var i = lid; i < cols; i = i + 256u) {
        let d = ta[base + i] - mean;
        sq = sq + d * d;
    }
    let variance = rowSum(lid, sq) / f32(cols);
    return vec2<f32>(mean, 1.0 / sqrt(variance + tp.eps));
}

// ln_fwd normalizes each row and applies the per-feature gain and bias.
@compute @workgroup_size(256, 1, 1)
fn ln_fwd(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= tp.rows) {
        return;
    }
    let cols = tp.count;
    let base = row * cols;
    let st = ln_stats(base, cols, lid.x);
    for (var i = lid.x; i < cols; i = i + 256u) {
        let h = (ta[base + i] - st.x) * st.y;
        tout[base + i] = h * tb[i] + tc[i];
    }
}

// ln_xhat is the same normalization without the affine part: the backward
// pass needs it to build the gain's gradient.
@compute @workgroup_size(256, 1, 1)
fn ln_xhat(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= tp.rows) {
        return;
    }
    let cols = tp.count;
    let base = row * cols;
    let st = ln_stats(base, cols, lid.x);
    for (var i = lid.x; i < cols; i = i + 256u) {
        tout[base + i] = (ta[base + i] - st.x) * st.y;
    }
}

// ln_bwd turns the output gradient in tb into the input gradient, given the
// input in ta and the gain in tc. It recomputes the row statistics rather
// than carrying them over from the forward pass: one more pass over a row
// that is already in cache costs less than the buffer would.
@compute @workgroup_size(256, 1, 1)
fn ln_bwd(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= tp.rows) {
        return;
    }
    let cols = tp.count;
    let base = row * cols;
    let st = ln_stats(base, cols, lid.x);
    // sum(dxhat) and sum(dxhat * xhat) over the row.
    var s1 = 0.0;
    var s2 = 0.0;
    for (var i = lid.x; i < cols; i = i + 256u) {
        let xhat = (ta[base + i] - st.x) * st.y;
        let dxhat = tb[base + i] * tc[i];
        s1 = s1 + dxhat;
        s2 = s2 + dxhat * xhat;
    }
    let sumD = rowSum(lid.x, s1);
    let sumDX = rowSum(lid.x, s2);
    let n = f32(cols);
    for (var i = lid.x; i < cols; i = i + 256u) {
        let xhat = (ta[base + i] - st.x) * st.y;
        let dxhat = tb[base + i] * tc[i];
        tout[base + i] = st.y / n * (n * dxhat - sumD - xhat * sumDX);
    }
}

// softmax_bwd turns the output gradient in tb into the input gradient of a
// softmax whose output is in ta: dx = y * (grad - sum(grad * y)).
@compute @workgroup_size(256, 1, 1)
fn softmax_bwd(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= tp.rows) {
        return;
    }
    let cols = tp.count;
    let base = row * cols;
    var dot = 0.0;
    for (var i = lid.x; i < cols; i = i + 256u) {
        dot = dot + ta[base + i] * tb[base + i];
    }
    let total = rowSum(lid.x, dot);
    for (var i = lid.x; i < cols; i = i + 256u) {
        tout[base + i] = ta[base + i] * (tb[base + i] - total);
    }
}

// permute reorders the axes of a tensor of rank up to four: each output
// element walks back to its source through the per-axis strides the host
// worked out.
@compute @workgroup_size(256, 1, 1)
fn permute(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx >= pmp.count) {
        return;
    }
    var rem = idx;
    var src = 0u;
    for (var d = pmp.rank; d > 0u; d = d - 1u) {
        let axis = d - 1u;
        let size = pmp.shape[axis];
        let i = rem % size;
        rem = rem / size;
        src = src + i * pmp.stride[axis];
    }
    tout[idx] = ta[src];
}

@group(0) @binding(54) var<storage, read> tids: array<u32>;
@group(0) @binding(55) var<storage, read_write> tatom: array<atomic<u32>>;

// embed_gather copies one table row per index: the forward pass of an
// embedding lookup.
@compute @workgroup_size(256, 1, 1)
fn embed_gather(@builtin(workgroup_id) wg: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx >= tp.count) {
        return;
    }
    let d = tp.aCount;
    tout[idx] = ta[tids[idx / d] * d + (idx % d)];
}

// embed_scatter adds each gradient row back into the table row its index
// names. Two tokens in a batch can be the same word, so the adds have to
// be atomic -- and WGSL has no atomic add for f32, so this is the usual
// compare-and-swap on the bit pattern.
@compute @workgroup_size(256, 1, 1)
fn embed_scatter(@builtin(workgroup_id) wg: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx >= tp.count) {
        return;
    }
    let d = tp.aCount;
    let dst = tids[idx / d] * d + (idx % d);
    let add = ta[idx];
    var old = atomicLoad(&tatom[dst]);
    loop {
        let sum = bitcast<u32>(bitcast<f32>(old) + add);
        let res = atomicCompareExchangeWeak(&tatom[dst], old, sum);
        if (res.exchanged) {
            break;
        }
        old = res.old_value;
    }
}

// slice_cols lifts a column window out of every row. The fused QKV
// projection lands in one buffer and prefill wants q, k, and v as
// tensors of their own; a bind offset cannot express that, because the
// window repeats once per row rather than sitting at one place in the
// buffer.
struct SliceParams { rows: u32, cols: u32, stride: u32, off: u32 }
@group(0) @binding(36) var<uniform> slp: SliceParams;
@group(0) @binding(37) var<storage, read> slsrc: array<f32>;
@group(0) @binding(38) var<storage, read_write> sldst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn slice_cols(@builtin(workgroup_id) wg: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < slp.rows * slp.cols) {
        let r = idx / slp.cols;
        sldst[idx] = slsrc[r * slp.stride + slp.off + (idx - r * slp.cols)];
    }
}

// glu_split joins a fused gate|up matmul result in one dispatch:
// out[r][i] = act(src[r][i]) * src[r][inter+i], the activation silu or
// (gelu != 0) gelu_tanh — the same expressions as silu_mul_ip and
// gelu_mul_ip, so a fused projection matches the split one bit for bit.
struct GSParams { rows: u32, inter: u32, gelu: u32, pad: u32 }
@group(0) @binding(44) var<uniform> gsp: GSParams;

@compute @workgroup_size(256, 1, 1)
fn glu_split(@builtin(workgroup_id) wg: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wg.y * nwg.x + wg.x) * 256u + lid.x;
    if (idx < gsp.rows * gsp.inter) {
        let r = idx / gsp.inter;
        let i = idx - r * gsp.inter;
        let g = slsrc[r * 2u * gsp.inter + i];
        let u = slsrc[r * 2u * gsp.inter + gsp.inter + i];
        var a = g / (1.0 + exp(-g));
        if (gsp.gelu != 0u) {
            let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
            a = 0.5 * g * (1.0 + tanh(inner));
        }
        sldst[idx] = a * u;
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
struct Q4MParams { rows: u32, cols: u32, words: u32, m: u32, groups: u32, flags: u32, pad1: u32, pad2: u32 }
@group(0) @binding(29) var<uniform> q4p: Q4MParams;
@group(0) @binding(30) var<storage, read> q4w: array<u32>;
@group(0) @binding(31) var<storage, read> q4sc: array<f32>;
@group(0) @binding(32) var<storage, read> q4x: array<f32>;
@group(0) @binding(33) var<storage, read_write> q4out: array<f32>;
@group(0) @binding(35) var<storage, read> q4bias: array<f32>;

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
                var o = sum[l];
                if ((q4p.flags & 1u) != 0u) {
                    o = o + q4bias[j + l];
                }
                if ((q4p.flags & 2u) != 0u) {
                    o = o + q4out[r * q4p.cols + j + l];
                }
                q4out[r * q4p.cols + j + l] = o;
            }
        }
    }
}
`

// intDotWGSL is the integer-dot GEMM module, compiled separately from
// matmulWGSL so a naga without dot4I8Packed only disables this path.
// qacts_pack quantizes activation rows to symmetric int8 (one scale per
// row, four values per u32); qmatmul_i then multiplies packed
// activations against the resident col-packed weights, transposing each
// staged weight slice to K-packed form in shared memory, four int8
// multiply-adds per dot4I8Packed.
const intDotWGSL = `
struct QIParams { rows: u32, kw4: u32, cols: u32, words: u32, m: u32, flags: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> qip: QIParams;
@group(0) @binding(1) var<storage, read> qix: array<f32>;
@group(0) @binding(2) var<storage, read_write> qia: array<u32>;
@group(0) @binding(3) var<storage, read_write> qis: array<f32>;
@group(0) @binding(4) var<storage, read> qiw: array<u32>;
@group(0) @binding(5) var<storage, read> qisc: array<f32>;
@group(0) @binding(6) var<storage, read_write> qio: array<f32>;
@group(0) @binding(7) var<storage, read> qibias: array<f32>;

var<workgroup> qired: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn qacts_pack(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let t = lid.x;
    var mx = 0.0;
    for (var i = t; i < qip.rows; i = i + 256u) {
        mx = max(mx, abs(qix[r * qip.rows + i]));
    }
    qired[t] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            qired[t] = max(qired[t], qired[t + s]);
        }
        workgroupBarrier();
    }
    let scale = max(qired[0], 1e-20) / 127.0;
    if (t == 0u) {
        qis[r] = scale;
    }
    let inv = 1.0 / scale;
    for (var kw = t; kw < qip.kw4; kw = kw + 256u) {
        var word = 0u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            let i = kw * 4u + b;
            var q = 0;
            if (i < qip.rows) {
                q = i32(round(clamp(qix[r * qip.rows + i] * inv, -127.0, 127.0)));
            }
            word = word | ((u32(q) & 0xFFu) << (8u * b));
        }
        qia[r * qip.kw4 + kw] = word;
    }
}

const QIR = 64u; // output rows per workgroup
const QIC = 64u; // output columns per workgroup
const QIKW = 16u; // kwords staged per barrier period
// Staged slices live as [kword][quad] vec4s: one vec4 holds four
// consecutive rows (or columns), so the inner loop issues two vec4 loads
// per kword instead of eight scalar ones.
var<workgroup> qita: array<vec4<u32>, 256>; // [kw][rowQuad] 16x16
var<workgroup> qitw: array<vec4<u32>, 256>; // [kw][colQuad] 16x16, K-packed

@compute @workgroup_size(256, 1, 1)
fn qmatmul_i(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let csub = lid.x % 16u;
    let rsub = lid.x / 16u;
    let c0 = wid.x * QIC;
    let r0 = wid.y * QIR;
    var a00 = 0; var a01 = 0; var a02 = 0; var a03 = 0;
    var a10 = 0; var a11 = 0; var a12 = 0; var a13 = 0;
    var a20 = 0; var a21 = 0; var a22 = 0; var a23 = 0;
    var a30 = 0; var a31 = 0; var a32 = 0; var a33 = 0;
    let stgKW = lid.x / 16u;  // kword this thread stages
    let stgQ = lid.x % 16u;   // row/col quad this thread stages
    let kwTiles = (qip.kw4 + QIKW - 1u) / QIKW;
    for (var kt = 0u; kt < kwTiles; kt = kt + 1u) {
        let kwb = kt * QIKW;
        let kwOK = kwb + stgKW < qip.kw4;
        // Activations: four consecutive rows at this kword.
        var av = vec4<u32>(0u, 0u, 0u, 0u);
        if (kwOK) {
            for (var b = 0u; b < 4u; b = b + 1u) {
                let r = r0 + stgQ * 4u + b;
                if (r < qip.m) {
                    av[b] = qia[r * qip.kw4 + kwb + stgKW];
                }
            }
        }
        qita[stgKW * 16u + stgQ] = av;
        // Weights: the four columns of one quad share a packed word, so
        // four global loads (one per K row) transpose into four K-packed
        // lanes at once.
        var wv = vec4<u32>(0u, 0u, 0u, 0u);
        let wcol = (c0 + stgQ * 4u) / 4u;
        if (kwOK && wcol < qip.words) {
            let k = (kwb + stgKW) * 4u;
            for (var b = 0u; b < 4u; b = b + 1u) {
                if (k + b < qip.rows) {
                    let wd = qiw[(k + b) * qip.words + wcol];
                    wv.x = wv.x | (((wd >> 0u) & 0xFFu) << (8u * b));
                    wv.y = wv.y | (((wd >> 8u) & 0xFFu) << (8u * b));
                    wv.z = wv.z | (((wd >> 16u) & 0xFFu) << (8u * b));
                    wv.w = wv.w | (((wd >> 24u) & 0xFFu) << (8u * b));
                }
            }
        }
        qitw[stgKW * 16u + stgQ] = wv;
        workgroupBarrier();
        for (var kw = 0u; kw < QIKW; kw = kw + 1u) {
            let va = qita[kw * 16u + rsub];
            let wa = qitw[kw * 16u + csub];
            let v0 = va.x;
            let v1 = va.y;
            let v2 = va.z;
            let v3 = va.w;
            let w0 = wa.x;
            let w1 = wa.y;
            let w2 = wa.z;
            let w3 = wa.w;
            a00 = a00 + dot4I8Packed(v0, w0);
            a01 = a01 + dot4I8Packed(v0, w1);
            a02 = a02 + dot4I8Packed(v0, w2);
            a03 = a03 + dot4I8Packed(v0, w3);
            a10 = a10 + dot4I8Packed(v1, w0);
            a11 = a11 + dot4I8Packed(v1, w1);
            a12 = a12 + dot4I8Packed(v1, w2);
            a13 = a13 + dot4I8Packed(v1, w3);
            a20 = a20 + dot4I8Packed(v2, w0);
            a21 = a21 + dot4I8Packed(v2, w1);
            a22 = a22 + dot4I8Packed(v2, w2);
            a23 = a23 + dot4I8Packed(v2, w3);
            a30 = a30 + dot4I8Packed(v3, w0);
            a31 = a31 + dot4I8Packed(v3, w1);
            a32 = a32 + dot4I8Packed(v3, w2);
            a33 = a33 + dot4I8Packed(v3, w3);
        }
        workgroupBarrier();
    }
    let rr = r0 + rsub * 4u;
    let cc = c0 + csub * 4u;
    for (var r = 0u; r < 4u; r = r + 1u) {
        if (rr + r < qip.m) {
            var av = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (r == 0u) { av = vec4<f32>(f32(a00), f32(a01), f32(a02), f32(a03)); }
            if (r == 1u) { av = vec4<f32>(f32(a10), f32(a11), f32(a12), f32(a13)); }
            if (r == 2u) { av = vec4<f32>(f32(a20), f32(a21), f32(a22), f32(a23)); }
            if (r == 3u) { av = vec4<f32>(f32(a30), f32(a31), f32(a32), f32(a33)); }
            let sa = qis[rr + r];
            for (var c = 0u; c < 4u; c = c + 1u) {
                if (cc + c < qip.cols) {
                    var o = av[c] * sa * qisc[cc + c];
                    if ((qip.flags & 1u) != 0u) {
                        o = o + qibias[cc + c];
                    }
                    if ((qip.flags & 2u) != 0u) {
                        o = o + qio[(rr + r) * qip.cols + cc + c];
                    }
                    qio[(rr + r) * qip.cols + cc + c] = o;
                }
            }
        }
    }
}
`

// attnF16WGSL is the f16 module: the grouped attention kernel reading a
// half-precision KV cache, and the conversion kernel that appends f32
// rows into it. Compiled only when the device granted shader-f16;
// accumulation stays in f32, so only the cache storage narrows.
const attnF16WGSL = `
enable f16;

struct APParams { seqQ: u32, seqKV: u32, dh: u32, d: u32, rows: u32, off: u32, dkv: u32, window: u32 }
@group(0) @binding(3) var<storage, read> offs: array<vec4<u32>>;
@group(0) @binding(10) var<uniform> ap: APParams;
@group(0) @binding(11) var<storage, read> aq: array<f32>;
// K is bound as u32 words: the QK loop unpacks two f16 lanes per load,
// halving its load instructions against per-element f16 reads. V keeps
// the f16 view — its loop is lane-coalesced already.
@group(0) @binding(12) var<storage, read> akh: array<u32>;
@group(0) @binding(13) var<storage, read> avh: array<f16>;
@group(0) @binding(14) var<storage, read_write> aout: array<f32>;

struct CVParams { n: u32, off: u32, pad0: u32, pad1: u32 }
@group(0) @binding(20) var<uniform> cvp: CVParams;
@group(0) @binding(21) var<storage, read> cvsrc: array<f32>;
@group(0) @binding(22) var<storage, read_write> cvdst: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn rows_to_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < cvp.n) {
        cvdst[cvp.off + gid.x] = f16(cvsrc[gid.x]);
    }
}

const AT = 64u;
var<workgroup> qrow_h: array<f32, 2048>;
var<workgroup> sc_h: array<f32, 512>;
var<workgroup> red_h: array<f32, 512>;

@compute @workgroup_size(64, 1, 1)
fn attn_causal_gh(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(num_workgroups) nwg: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= ap.rows) {
        return;
    }
    let grp = ap.d / ap.dkv;
    let bh = row / ap.seqQ;
    let qi = row % ap.seqQ;
    let offQ = offs[bh].x;
    let offKV = offs[bh].y;
    let offO = offs[bh].z;
    let t = lid.x;
    for (var c = t; c < grp * ap.dh; c = c + 64u) {
        qrow_h[c] = aq[offQ + qi * ap.d + c];
    }
    workgroupBarrier();
    let limit = qi + ap.off + 1u;
    var start = 0u;
    if (ap.window > 0u && limit > ap.window) {
        start = limit - ap.window;
    }
    let scale = inverseSqrt(f32(ap.dh));
    // Per-head state lives in named scalars, never in indexed arrays:
    // function-scope arrays with dynamic indices land in scratch memory
    // and made this kernel about twice as slow. Heads past grp
    // accumulate zeros (workgroup memory is zero-initialized) and are
    // never written back, so the unrolled arithmetic matches the old
    // per-head loop bit for bit.
    var m0 = -3.40282e38; var m1 = -3.40282e38; var m2 = -3.40282e38; var m3 = -3.40282e38;
    var m4 = -3.40282e38; var m5 = -3.40282e38; var m6 = -3.40282e38; var m7 = -3.40282e38;
    var l0 = 0.0; var l1 = 0.0; var l2 = 0.0; var l3 = 0.0;
    var l4 = 0.0; var l5 = 0.0; var l6 = 0.0; var l7 = 0.0;
    var acc0 = vec4<f32>(); var acc1 = vec4<f32>(); var acc2 = vec4<f32>(); var acc3 = vec4<f32>();
    var acc4 = vec4<f32>(); var acc5 = vec4<f32>(); var acc6 = vec4<f32>(); var acc7 = vec4<f32>();
    let tiles = (limit + AT - 1u) / AT;
    for (var tt = start / AT; tt < tiles; tt = tt + 1u) {
        let j = tt * AT + t;
        let valid = j >= start && j < limit;
        var d0 = 0.0; var d1 = 0.0; var d2 = 0.0; var d3 = 0.0;
        var d4 = 0.0; var d5 = 0.0; var d6 = 0.0; var d7 = 0.0;
        if (valid) {
            let kbase = (offKV + j * ap.dkv) >> 1u;
            for (var c = 0u; c < ap.dh; c = c + 2u) {
                let kv = unpack2x16float(akh[kbase + (c >> 1u)]);
                d0 = d0 + qrow_h[c] * kv.x + qrow_h[c + 1u] * kv.y;
                d1 = d1 + qrow_h[ap.dh + c] * kv.x + qrow_h[ap.dh + c + 1u] * kv.y;
                d2 = d2 + qrow_h[2u * ap.dh + c] * kv.x + qrow_h[2u * ap.dh + c + 1u] * kv.y;
                d3 = d3 + qrow_h[3u * ap.dh + c] * kv.x + qrow_h[3u * ap.dh + c + 1u] * kv.y;
                d4 = d4 + qrow_h[4u * ap.dh + c] * kv.x + qrow_h[4u * ap.dh + c + 1u] * kv.y;
                d5 = d5 + qrow_h[5u * ap.dh + c] * kv.x + qrow_h[5u * ap.dh + c + 1u] * kv.y;
                d6 = d6 + qrow_h[6u * ap.dh + c] * kv.x + qrow_h[6u * ap.dh + c + 1u] * kv.y;
                d7 = d7 + qrow_h[7u * ap.dh + c] * kv.x + qrow_h[7u * ap.dh + c + 1u] * kv.y;
            }
            d0 = d0 * scale; d1 = d1 * scale; d2 = d2 * scale; d3 = d3 * scale;
            d4 = d4 * scale; d5 = d5 * scale; d6 = d6 * scale; d7 = d7 * scale;
        } else {
            d0 = -3.40282e38; d1 = -3.40282e38; d2 = -3.40282e38; d3 = -3.40282e38;
            d4 = -3.40282e38; d5 = -3.40282e38; d6 = -3.40282e38; d7 = -3.40282e38;
        }
        red_h[t] = d0;         red_h[64u + t] = d1;
        red_h[128u + t] = d2;  red_h[192u + t] = d3;
        red_h[256u + t] = d4;  red_h[320u + t] = d5;
        red_h[384u + t] = d6;  red_h[448u + t] = d7;
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) {
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        red_h[h * 64u + t] = max(red_h[h * 64u + t], red_h[h * 64u + t + r]);
                    }
                }
            }
            workgroupBarrier();
        }
        let n0 = max(m0, red_h[0]);   let n1 = max(m1, red_h[64]);
        let n2 = max(m2, red_h[128]); let n3 = max(m3, red_h[192]);
        let n4 = max(m4, red_h[256]); let n5 = max(m5, red_h[320]);
        let n6 = max(m6, red_h[384]); let n7 = max(m7, red_h[448]);
        var p0 = 0.0; var p1 = 0.0; var p2 = 0.0; var p3 = 0.0;
        var p4 = 0.0; var p5 = 0.0; var p6 = 0.0; var p7 = 0.0;
        if (valid) {
            p0 = exp(d0 - n0); p1 = exp(d1 - n1); p2 = exp(d2 - n2); p3 = exp(d3 - n3);
            p4 = exp(d4 - n4); p5 = exp(d5 - n5); p6 = exp(d6 - n6); p7 = exp(d7 - n7);
        }
        workgroupBarrier();
        sc_h[t] = p0;         sc_h[64u + t] = p1;
        sc_h[128u + t] = p2;  sc_h[192u + t] = p3;
        sc_h[256u + t] = p4;  sc_h[320u + t] = p5;
        sc_h[384u + t] = p6;  sc_h[448u + t] = p7;
        red_h[t] = p0;         red_h[64u + t] = p1;
        red_h[128u + t] = p2;  red_h[192u + t] = p3;
        red_h[256u + t] = p4;  red_h[320u + t] = p5;
        red_h[384u + t] = p6;  red_h[448u + t] = p7;
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) {
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        red_h[h * 64u + t] = red_h[h * 64u + t] + red_h[h * 64u + t + r];
                    }
                }
            }
            workgroupBarrier();
        }
        // exp underflows to zero on the first tile, where m is -inf-like.
        let r0 = exp(m0 - n0); let r1 = exp(m1 - n1); let r2 = exp(m2 - n2); let r3 = exp(m3 - n3);
        let r4 = exp(m4 - n4); let r5 = exp(m5 - n5); let r6 = exp(m6 - n6); let r7 = exp(m7 - n7);
        l0 = l0 * r0 + red_h[0];   l1 = l1 * r1 + red_h[64];
        l2 = l2 * r2 + red_h[128]; l3 = l3 * r3 + red_h[192];
        l4 = l4 * r4 + red_h[256]; l5 = l5 * r5 + red_h[320];
        l6 = l6 * r6 + red_h[384]; l7 = l7 * r7 + red_h[448];
        m0 = n0; m1 = n1; m2 = n2; m3 = n3;
        m4 = n4; m5 = n5; m6 = n6; m7 = n7;
        acc0 = acc0 * r0; acc1 = acc1 * r1; acc2 = acc2 * r2; acc3 = acc3 * r3;
        acc4 = acc4 * r4; acc5 = acc5 * r5; acc6 = acc6 * r6; acc7 = acc7 * r7;
        let jEnd = min(limit, tt * AT + AT);
        for (var jj = max(start, tt * AT); jj < jEnd; jj = jj + 1u) {
            var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (t < ap.dh) {
                vv.x = f32(avh[offKV + jj * ap.dkv + t]);
            }
            if (64u + t < ap.dh) {
                vv.y = f32(avh[offKV + jj * ap.dkv + 64u + t]);
            }
            if (128u + t < ap.dh) {
                vv.z = f32(avh[offKV + jj * ap.dkv + 128u + t]);
            }
            if (192u + t < ap.dh) {
                vv.w = f32(avh[offKV + jj * ap.dkv + 192u + t]);
            }
            let sj = jj - tt * AT;
            acc0 = acc0 + sc_h[sj] * vv;
            acc1 = acc1 + sc_h[64u + sj] * vv;
            acc2 = acc2 + sc_h[128u + sj] * vv;
            acc3 = acc3 + sc_h[192u + sj] * vv;
            acc4 = acc4 + sc_h[256u + sj] * vv;
            acc5 = acc5 + sc_h[320u + sj] * vv;
            acc6 = acc6 + sc_h[384u + sj] * vv;
            acc7 = acc7 + sc_h[448u + sj] * vv;
        }
        workgroupBarrier();
    }
    for (var h = 0u; h < 8u; h = h + 1u) {
        if (h < grp) {
            var av = vec4<f32>();
            var lh = 1.0;
            if (h == 0u) { av = acc0; lh = l0; }
            if (h == 1u) { av = acc1; lh = l1; }
            if (h == 2u) { av = acc2; lh = l2; }
            if (h == 3u) { av = acc3; lh = l3; }
            if (h == 4u) { av = acc4; lh = l4; }
            if (h == 5u) { av = acc5; lh = l5; }
            if (h == 6u) { av = acc6; lh = l6; }
            if (h == 7u) { av = acc7; lh = l7; }
            let o = offO + qi * ap.d + h * ap.dh;
            if (t < ap.dh) {
                aout[o + t] = av.x / lh;
            }
            if (64u + t < ap.dh) {
                aout[o + 64u + t] = av.y / lh;
            }
            if (128u + t < ap.dh) {
                aout[o + 128u + t] = av.z / lh;
            }
            if (192u + t < ap.dh) {
                aout[o + 192u + t] = av.w / lh;
            }
        }
    }
}

struct RCParams { d: u32, headSz: u32, qw: u32, kvDim: u32, pos: u32, dstOff: u32, theta: f32, pad: u32 }
@group(0) @binding(43) var<uniform> rcp: RCParams;
@group(0) @binding(41) var<storage, read_write> rcx: array<f32>;
@group(0) @binding(42) var<storage, read_write> rcv: array<f16>;

// rope_cache fuses decode's rotation with the cache append: one dispatch
// rotates the fused q|k row in place, writes the rotated k on into the
// f16 K cache, and copies v into the V cache — three dependent ~40us
// dispatches become one. The rotation matches rope_rows bit for bit and
// the narrowing matches rows_to_f16, so nothing downstream changes.
@compute @workgroup_size(64, 1, 1)
fn rope_cache(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = rcp.headSz / 2u;
    let pairs = (rcp.d / rcp.headSz) * half;
    if (gid.x < pairs) {
        let h = gid.x / half;
        let c = gid.x % half;
        let base = h * rcp.headSz + c;
        let freq = pow(rcp.theta, -2.0 * f32(c) / f32(rcp.headSz));
        let ang = f32(rcp.pos) * freq;
        let sn = sin(ang);
        let cs = cos(ang);
        let a = rcx[base];
        let b = rcx[base + half];
        let na = a * cs - b * sn;
        let nb = b * cs + a * sn;
        rcx[base] = na;
        rcx[base + half] = nb;
        // A k head sits entirely past the query region, so both pair
        // elements land in the cache together.
        if (base >= rcp.qw) {
            cvdst[rcp.dstOff + base - rcp.qw] = f16(na);
            cvdst[rcp.dstOff + base + half - rcp.qw] = f16(nb);
        }
        return;
    }
    let i = gid.x - pairs;
    if (i < rcp.kvDim) {
        rcv[rcp.dstOff + i] = f16(rcx[rcp.qw + rcp.kvDim + i]);
    }
}

// attn_split_gh is attn_causal_gh's decode split: a single query row at
// two KV heads leaves ten of twelve WGPs idle while one workgroup walks
// the whole context (~680us a layer at 400 positions), so each workgroup
// takes one slab of KV tiles instead and parks its unnormalized head
// accumulators with the running (m, l) softmax state in ascr, which
// attn_reduce_g folds. Fields are repurposed: seqQ carries the tiles per
// slab and rows the slab count. Query row 0 only, no sliding window.
@group(0) @binding(40) var<storage, read_write> ascr: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn attn_split_gh(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let slab = wid.x;
    let kvh = wid.y;
    let grp = ap.d / ap.dkv;
    let offQ = offs[kvh].x;
    let offKV = offs[kvh].y;
    let t = lid.x;
    for (var c = t; c < grp * ap.dh; c = c + 64u) {
        qrow_h[c] = aq[offQ + c];
    }
    workgroupBarrier();
    let limit = ap.off + 1u;
    let scale = inverseSqrt(f32(ap.dh));
    var m0 = -3.40282e38; var m1 = -3.40282e38; var m2 = -3.40282e38; var m3 = -3.40282e38;
    var m4 = -3.40282e38; var m5 = -3.40282e38; var m6 = -3.40282e38; var m7 = -3.40282e38;
    var l0 = 0.0; var l1 = 0.0; var l2 = 0.0; var l3 = 0.0;
    var l4 = 0.0; var l5 = 0.0; var l6 = 0.0; var l7 = 0.0;
    var acc0 = vec4<f32>(); var acc1 = vec4<f32>(); var acc2 = vec4<f32>(); var acc3 = vec4<f32>();
    var acc4 = vec4<f32>(); var acc5 = vec4<f32>(); var acc6 = vec4<f32>(); var acc7 = vec4<f32>();
    let tiles = (limit + AT - 1u) / AT;
    let tEnd = min(tiles, (slab + 1u) * ap.seqQ);
    for (var tt = slab * ap.seqQ; tt < tEnd; tt = tt + 1u) {
        let j = tt * AT + t;
        let valid = j < limit;
        var d0 = 0.0; var d1 = 0.0; var d2 = 0.0; var d3 = 0.0;
        var d4 = 0.0; var d5 = 0.0; var d6 = 0.0; var d7 = 0.0;
        if (valid) {
            let kbase = (offKV + j * ap.dkv) >> 1u;
            for (var c = 0u; c < ap.dh; c = c + 2u) {
                let kv = unpack2x16float(akh[kbase + (c >> 1u)]);
                d0 = d0 + qrow_h[c] * kv.x + qrow_h[c + 1u] * kv.y;
                d1 = d1 + qrow_h[ap.dh + c] * kv.x + qrow_h[ap.dh + c + 1u] * kv.y;
                d2 = d2 + qrow_h[2u * ap.dh + c] * kv.x + qrow_h[2u * ap.dh + c + 1u] * kv.y;
                d3 = d3 + qrow_h[3u * ap.dh + c] * kv.x + qrow_h[3u * ap.dh + c + 1u] * kv.y;
                d4 = d4 + qrow_h[4u * ap.dh + c] * kv.x + qrow_h[4u * ap.dh + c + 1u] * kv.y;
                d5 = d5 + qrow_h[5u * ap.dh + c] * kv.x + qrow_h[5u * ap.dh + c + 1u] * kv.y;
                d6 = d6 + qrow_h[6u * ap.dh + c] * kv.x + qrow_h[6u * ap.dh + c + 1u] * kv.y;
                d7 = d7 + qrow_h[7u * ap.dh + c] * kv.x + qrow_h[7u * ap.dh + c + 1u] * kv.y;
            }
            d0 = d0 * scale; d1 = d1 * scale; d2 = d2 * scale; d3 = d3 * scale;
            d4 = d4 * scale; d5 = d5 * scale; d6 = d6 * scale; d7 = d7 * scale;
        } else {
            d0 = -3.40282e38; d1 = -3.40282e38; d2 = -3.40282e38; d3 = -3.40282e38;
            d4 = -3.40282e38; d5 = -3.40282e38; d6 = -3.40282e38; d7 = -3.40282e38;
        }
        red_h[t] = d0;         red_h[64u + t] = d1;
        red_h[128u + t] = d2;  red_h[192u + t] = d3;
        red_h[256u + t] = d4;  red_h[320u + t] = d5;
        red_h[384u + t] = d6;  red_h[448u + t] = d7;
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) {
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        red_h[h * 64u + t] = max(red_h[h * 64u + t], red_h[h * 64u + t + r]);
                    }
                }
            }
            workgroupBarrier();
        }
        let n0 = max(m0, red_h[0]);   let n1 = max(m1, red_h[64]);
        let n2 = max(m2, red_h[128]); let n3 = max(m3, red_h[192]);
        let n4 = max(m4, red_h[256]); let n5 = max(m5, red_h[320]);
        let n6 = max(m6, red_h[384]); let n7 = max(m7, red_h[448]);
        var p0 = 0.0; var p1 = 0.0; var p2 = 0.0; var p3 = 0.0;
        var p4 = 0.0; var p5 = 0.0; var p6 = 0.0; var p7 = 0.0;
        if (valid) {
            p0 = exp(d0 - n0); p1 = exp(d1 - n1); p2 = exp(d2 - n2); p3 = exp(d3 - n3);
            p4 = exp(d4 - n4); p5 = exp(d5 - n5); p6 = exp(d6 - n6); p7 = exp(d7 - n7);
        }
        workgroupBarrier();
        sc_h[t] = p0;         sc_h[64u + t] = p1;
        sc_h[128u + t] = p2;  sc_h[192u + t] = p3;
        sc_h[256u + t] = p4;  sc_h[320u + t] = p5;
        sc_h[384u + t] = p6;  sc_h[448u + t] = p7;
        red_h[t] = p0;         red_h[64u + t] = p1;
        red_h[128u + t] = p2;  red_h[192u + t] = p3;
        red_h[256u + t] = p4;  red_h[320u + t] = p5;
        red_h[384u + t] = p6;  red_h[448u + t] = p7;
        workgroupBarrier();
        for (var r = 32u; r > 0u; r = r >> 1u) {
            if (t < r) {
                for (var h = 0u; h < 8u; h = h + 1u) {
                    if (h < grp) {
                        red_h[h * 64u + t] = red_h[h * 64u + t] + red_h[h * 64u + t + r];
                    }
                }
            }
            workgroupBarrier();
        }
        let r0 = exp(m0 - n0); let r1 = exp(m1 - n1); let r2 = exp(m2 - n2); let r3 = exp(m3 - n3);
        let r4 = exp(m4 - n4); let r5 = exp(m5 - n5); let r6 = exp(m6 - n6); let r7 = exp(m7 - n7);
        l0 = l0 * r0 + red_h[0];   l1 = l1 * r1 + red_h[64];
        l2 = l2 * r2 + red_h[128]; l3 = l3 * r3 + red_h[192];
        l4 = l4 * r4 + red_h[256]; l5 = l5 * r5 + red_h[320];
        l6 = l6 * r6 + red_h[384]; l7 = l7 * r7 + red_h[448];
        m0 = n0; m1 = n1; m2 = n2; m3 = n3;
        m4 = n4; m5 = n5; m6 = n6; m7 = n7;
        acc0 = acc0 * r0; acc1 = acc1 * r1; acc2 = acc2 * r2; acc3 = acc3 * r3;
        acc4 = acc4 * r4; acc5 = acc5 * r5; acc6 = acc6 * r6; acc7 = acc7 * r7;
        let jEnd = min(limit, tt * AT + AT);
        for (var jj = tt * AT; jj < jEnd; jj = jj + 1u) {
            var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (t < ap.dh) {
                vv.x = f32(avh[offKV + jj * ap.dkv + t]);
            }
            if (64u + t < ap.dh) {
                vv.y = f32(avh[offKV + jj * ap.dkv + 64u + t]);
            }
            if (128u + t < ap.dh) {
                vv.z = f32(avh[offKV + jj * ap.dkv + 128u + t]);
            }
            if (192u + t < ap.dh) {
                vv.w = f32(avh[offKV + jj * ap.dkv + 192u + t]);
            }
            let sj = jj - tt * AT;
            acc0 = acc0 + sc_h[sj] * vv;
            acc1 = acc1 + sc_h[64u + sj] * vv;
            acc2 = acc2 + sc_h[128u + sj] * vv;
            acc3 = acc3 + sc_h[192u + sj] * vv;
            acc4 = acc4 + sc_h[256u + sj] * vv;
            acc5 = acc5 + sc_h[320u + sj] * vv;
            acc6 = acc6 + sc_h[384u + sj] * vv;
            acc7 = acc7 + sc_h[448u + sj] * vv;
        }
        workgroupBarrier();
    }
    let stride = 8u * ap.dh + 16u;
    let sbase = (kvh * ap.rows + slab) * stride;
    for (var h = 0u; h < 8u; h = h + 1u) {
        if (h < grp) {
            var av = vec4<f32>();
            var lh = 0.0;
            var mh = -3.40282e38;
            if (h == 0u) { av = acc0; lh = l0; mh = m0; }
            if (h == 1u) { av = acc1; lh = l1; mh = m1; }
            if (h == 2u) { av = acc2; lh = l2; mh = m2; }
            if (h == 3u) { av = acc3; lh = l3; mh = m3; }
            if (h == 4u) { av = acc4; lh = l4; mh = m4; }
            if (h == 5u) { av = acc5; lh = l5; mh = m5; }
            if (h == 6u) { av = acc6; lh = l6; mh = m6; }
            if (h == 7u) { av = acc7; lh = l7; mh = m7; }
            let o = sbase + h * ap.dh;
            if (t < ap.dh) {
                ascr[o + t] = av.x;
            }
            if (64u + t < ap.dh) {
                ascr[o + 64u + t] = av.y;
            }
            if (128u + t < ap.dh) {
                ascr[o + 128u + t] = av.z;
            }
            if (192u + t < ap.dh) {
                ascr[o + 192u + t] = av.w;
            }
            if (t == 0u) {
                ascr[sbase + 8u * ap.dh + h] = mh;
                ascr[sbase + 8u * ap.dh + 8u + h] = lh;
            }
        }
    }
}

// attn_reduce_g folds attn_split_gh's slabs: rebase every slab's l and
// accumulators onto the global max and normalize once. One workgroup per
// KV head; a slab that saw only masked positions carries l = 0 and an
// m of -inf, and drops out through exp underflow.
@compute @workgroup_size(64, 1, 1)
fn attn_reduce_g(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let kvh = wid.x;
    let grp = ap.d / ap.dkv;
    let slabs = ap.rows;
    let stride = 8u * ap.dh + 16u;
    let base = kvh * slabs * stride;
    let offO = offs[kvh].z;
    let t = lid.x;
    for (var h = 0u; h < grp; h = h + 1u) {
        var mh = -3.40282e38;
        for (var s = 0u; s < slabs; s = s + 1u) {
            mh = max(mh, ascr[base + s * stride + 8u * ap.dh + h]);
        }
        var lh = 0.0;
        for (var s = 0u; s < slabs; s = s + 1u) {
            let ms = ascr[base + s * stride + 8u * ap.dh + h];
            lh = lh + ascr[base + s * stride + 8u * ap.dh + 8u + h] * exp(ms - mh);
        }
        for (var c = t; c < ap.dh; c = c + 64u) {
            var o = 0.0;
            for (var s = 0u; s < slabs; s = s + 1u) {
                let ms = ascr[base + s * stride + 8u * ap.dh + h];
                o = o + ascr[base + s * stride + h * ap.dh + c] * exp(ms - mh);
            }
            aout[offO + h * ap.dh + c] = o / lh;
        }
    }
}
`

// IntDot reports whether the optional integer-dot GEMM module compiled
// on this device; without it large batches use the f32 tiled kernel.
func (g *Device) IntDot() bool { return g.hasIntDot }

// gpuPipelines holds one compute pipeline (and its auto bind-group layout)
// per kernel entry point. It is embedded in each binding generation's Device
// struct.
type gpuPipelines struct {
	matmul, matmulT, matmulS, matmulTS             uintptr
	matmulTN, matmulTNS                            uintptr
	layMatmulTN, layMatmulTNS                      uintptr
	binOp, actFwd, actBwd, sumCols, adamStep       uintptr
	layBinOp, layActFwd, layActBwd                 uintptr
	laySumCols, layAdamStep                        uintptr
	matmulL, matmulLT, matmulLTN                   uintptr
	matmulV4, matmulV4T, matmulV4TN                uintptr
	layMatmulV4, layMatmulV4T, layMatmulV4TN       uintptr
	layMatmulL, layMatmulLT, layMatmulLTN          uintptr
	lnFwd, lnXhat, lnBwd, softmaxBwd, permute      uintptr
	embedGather, embedScatter                      uintptr
	layEmbedGather, layEmbedScatter                uintptr
	layLnFwd, layLnXhat, layLnBwd                  uintptr
	laySoftmaxBwd, layPermute                      uintptr
	scale, softmax, attn, qmatmul                  uintptr
	rmsnorm, rope, addIP, siluMulIP, q4matmul      uintptr
	geluMulIP, qmatmulB, attnG, qmatmulT           uintptr
	qacts, qmatmulI, attnF16, rowsToF16            uintptr
	attnSplit, attnReduce, ropeCache               uintptr
	layAttnSplit, layAttnReduce, layRopeCache      uintptr
	gluSplit, layGluSplit                          uintptr
	sliceCols, laySliceCols                        uintptr
	layMatmul, layMatmulT, layMatmulS, layMatmulTS uintptr
	layScale, laySoftmax, layAttn, layQmatmul      uintptr
	layRmsnorm, layRope, layAddIP, laySiluMulIP    uintptr
	layQ4matmul, layGeluMulIP, layQmatmulB         uintptr
	layAttnG, layQmatmulT, layQacts, layQmatmulI   uintptr
	layAttnF16, layRowsToF16                       uintptr
}

// initPipelines compiles every kernel from g.module; the caller holds
// wgpuMu.
func (g *Device) initPipelines() error {
	for _, x := range []struct {
		pipe, lay *uintptr
		entry     string
	}{
		{&g.pipes.matmul, &g.pipes.layMatmul, "main"},
		{&g.pipes.matmulT, &g.pipes.layMatmulT, "matmul_t"},
		{&g.pipes.matmulS, &g.pipes.layMatmulS, "matmul_s"},
		{&g.pipes.matmulTS, &g.pipes.layMatmulTS, "matmul_ts"},
		{&g.pipes.matmulTN, &g.pipes.layMatmulTN, "matmul_tn"},
		{&g.pipes.matmulTNS, &g.pipes.layMatmulTNS, "matmul_tns"},
		{&g.pipes.matmulL, &g.pipes.layMatmulL, "matmul_l"},
		{&g.pipes.matmulV4, &g.pipes.layMatmulV4, "matmul_v4"},
		{&g.pipes.matmulV4T, &g.pipes.layMatmulV4T, "matmul_v4t"},
		{&g.pipes.matmulV4TN, &g.pipes.layMatmulV4TN, "matmul_v4tn"},
		{&g.pipes.matmulLT, &g.pipes.layMatmulLT, "matmul_lt"},
		{&g.pipes.matmulLTN, &g.pipes.layMatmulLTN, "matmul_ltn"},
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
		{&g.pipes.qmatmulB, &g.pipes.layQmatmulB, "qmatmul_b"},
		{&g.pipes.attnG, &g.pipes.layAttnG, "attn_causal_g"},
		{&g.pipes.qmatmulT, &g.pipes.layQmatmulT, "qmatmul_t"},
		{&g.pipes.sliceCols, &g.pipes.laySliceCols, "slice_cols"},
		{&g.pipes.gluSplit, &g.pipes.layGluSplit, "glu_split"},
		{&g.pipes.binOp, &g.pipes.layBinOp, "bin_op"},
		{&g.pipes.actFwd, &g.pipes.layActFwd, "act_fwd"},
		{&g.pipes.actBwd, &g.pipes.layActBwd, "act_bwd"},
		{&g.pipes.sumCols, &g.pipes.laySumCols, "sum_cols"},
		{&g.pipes.adamStep, &g.pipes.layAdamStep, "adam_step"},
		{&g.pipes.lnFwd, &g.pipes.layLnFwd, "ln_fwd"},
		{&g.pipes.lnXhat, &g.pipes.layLnXhat, "ln_xhat"},
		{&g.pipes.lnBwd, &g.pipes.layLnBwd, "ln_bwd"},
		{&g.pipes.softmaxBwd, &g.pipes.laySoftmaxBwd, "softmax_bwd"},
		{&g.pipes.permute, &g.pipes.layPermute, "permute"},
		{&g.pipes.embedGather, &g.pipes.layEmbedGather, "embed_gather"},
		{&g.pipes.embedScatter, &g.pipes.layEmbedScatter, "embed_scatter"},
	} {
		*x.pipe = g.makePipeline(x.entry)
		if *x.pipe == 0 || uncapturedCB != "" {
			return fmt.Errorf("tensai: wgpu pipeline %q creation failed: %s", x.entry, uncapturedCB)
		}
		*x.lay = fnPipelineGetLayout(*x.pipe, 0)
	}
	// The integer-dot module is optional — dot4I8Packed needs a newer
	// naga — so a compile failure here just leaves the f32 tiled kernel.
	g.module2 = g.makeModuleFrom(intDotWGSL)
	if g.module2 != 0 && uncapturedCB == "" {
		ok := true
		for _, x := range []struct {
			pipe, lay *uintptr
			entry     string
		}{
			{&g.pipes.qacts, &g.pipes.layQacts, "qacts_pack"},
			{&g.pipes.qmatmulI, &g.pipes.layQmatmulI, "qmatmul_i"},
		} {
			*x.pipe = g.makePipelineIn(g.module2, x.entry)
			if *x.pipe == 0 || uncapturedCB != "" {
				ok = false
				break
			}
			*x.lay = fnPipelineGetLayout(*x.pipe, 0)
		}
		g.hasIntDot = ok
	}
	uncapturedCB = ""
	// The f16 module needs the shader-f16 device feature; failure to
	// compile just leaves the f32 cache path.
	if g.hasF16 {
		ok := false
		g.module3 = g.makeModuleFrom(attnF16WGSL)
		if g.module3 != 0 && uncapturedCB == "" {
			ok = true
			for _, x := range []struct {
				pipe, lay *uintptr
				entry     string
			}{
				{&g.pipes.attnF16, &g.pipes.layAttnF16, "attn_causal_gh"},
				{&g.pipes.rowsToF16, &g.pipes.layRowsToF16, "rows_to_f16"},
				{&g.pipes.attnSplit, &g.pipes.layAttnSplit, "attn_split_gh"},
				{&g.pipes.attnReduce, &g.pipes.layAttnReduce, "attn_reduce_g"},
				{&g.pipes.ropeCache, &g.pipes.layRopeCache, "rope_cache"},
			} {
				*x.pipe = g.makePipelineIn(g.module3, x.entry)
				if *x.pipe == 0 || uncapturedCB != "" {
					ok = false
					break
				}
				*x.lay = fnPipelineGetLayout(*x.pipe, 0)
			}
		}
		g.hasF16 = ok
		uncapturedCB = ""
	}
	return nil
}

// releasePipelines drops every pipeline and layout; the caller holds
// wgpuMu.
func (g *Device) releasePipelines() {
	for _, h := range []uintptr{
		g.pipes.layMatmul, g.pipes.layMatmulT, g.pipes.layMatmulS,
		g.pipes.layMatmulTS, g.pipes.layScale, g.pipes.laySoftmax, g.pipes.layAttn,
		g.pipes.layQmatmul, g.pipes.layRmsnorm, g.pipes.layRope,
		g.pipes.layAddIP, g.pipes.laySiluMulIP, g.pipes.layQ4matmul,
		g.pipes.layGeluMulIP, g.pipes.layQmatmulB, g.pipes.layAttnG,
		g.pipes.layQmatmulT, g.pipes.layQacts, g.pipes.layQmatmulI,
		g.pipes.layAttnF16, g.pipes.layRowsToF16,
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
		g.pipes.geluMulIP, g.pipes.qmatmulB, g.pipes.attnG,
		g.pipes.qmatmulT, g.pipes.sliceCols, g.pipes.qacts, g.pipes.qmatmulI,
		g.pipes.attnF16, g.pipes.rowsToF16,
	} {
		if h != 0 {
			fnPipelineRelease(h)
		}
	}
}

// dispatch runs one compute pass and pumps the error callback once; the
// caller holds wgpuMu. Inside a batch (see BeginBatch) the pass is only
// recorded — submission waits for Flush.
func (g *Device) dispatch(pipe, bindGroup uintptr, x, y, z uint32) error {
	encoder := g.batchEnc
	if encoder == 0 {
		encoder = fnDeviceCreateCmdEncoder(g.device, nil)
	}
	// Inside a batch every dispatch shares one open compute pass: the
	// spec orders dispatches within a pass (each is its own
	// synchronization scope), and per-pass setup is where drivers like
	// dozen burn milliseconds. Buffer copies end the pass (endBatchPass);
	// the next dispatch reopens it.
	var pass uintptr
	if g.batchEnc != 0 {
		if g.batchPass == 0 {
			g.batchPass = fnEncoderBeginComputePass(encoder, nil)
		}
		pass = g.batchPass
	} else {
		pass = fnEncoderBeginComputePass(encoder, nil)
	}
	fnPassSetPipeline(pass, pipe)
	fnPassSetBindGroup(pass, 0, bindGroup, 0, nil)
	fnPassDispatch(pass, x, y, z)
	if g.batchEnc != 0 {
		if uncapturedCB != "" {
			return fmt.Errorf("tensai: gpu dispatch failed: %s", uncapturedCB)
		}
		return nil
	}
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

// BeginBatch opens a command encoder that collects every subsequent
// operation on this Device — compute dispatches and buffer copies — into one
// submission, which Flush sends. One submission instead of hundreds is
// the difference between a usable and an unusable decode step on drivers
// with per-submit overhead. Operations inside a batch report validation
// errors at Flush; Download flushes an open batch implicitly.
func (g *Device) BeginBatch() error {
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
func (g *Device) Flush() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	return g.flushLocked()
}

// uarCap sizes the uniform staging arena: a decode chunk stages well
// under a thousand 256-byte slots.
const uarCap = 1 << 18

// stageUniform parks a small parameter block in the per-batch uniform
// arena instead of a pooled buffer of its own: the bytes accumulate on
// the CPU and reach the device in one queue write at flush, replacing a
// takeBuffer/putBuffer round trip and a queue write per dispatch. The
// arena rewinds after every flush — a queue write for the next chunk is
// ordered after the flushed submission, so the slots may be reused.
// Only meaningful inside a batch; callers fall back when it reports
// false. The caller holds wgpuMu.
func (g *Device) stageUniform(p unsafe.Pointer, n int) (uintptr, uint64, bool) {
	if g.batchEnc == 0 || n > 256 {
		return 0, 0, false
	}
	if g.uarBuf == 0 {
		g.uarBuf = g.newBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, uarCap)
		if g.uarBuf == 0 {
			return 0, 0, false
		}
		g.uarStage = make([]byte, uarCap)
	}
	off := g.uarOff
	if off+256 > uarCap {
		return 0, 0, false
	}
	copy(g.uarStage[off:], unsafe.Slice((*byte)(p), n))
	g.uarOff = off + 256 // the minimum uniform buffer offset alignment
	return g.uarBuf, off, true
}

// paramsBuffer places a dispatch's parameter block: in the batch arena
// when one is open, else in a pooled buffer written immediately. The
// returned func releases the fallback buffer and is a no-op for the
// arena. The caller holds wgpuMu.
func (g *Device) paramsBuffer(p unsafe.Pointer, n uint64) (uintptr, uint64, func()) {
	if b, off, ok := g.stageUniform(p, int(n)); ok {
		return b, off, func() {}
	}
	usage := uint64(wgpuBufferUsageUniform | wgpuBufferUsageCopyDst)
	b := g.takeBuffer(usage, n)
	fnQueueWriteBuffer(g.queue, b, 0, p, uintptr(n))
	return b, 0, func() { g.putBuffer(usage, n, b) }
}

// offsBuffer returns a resident storage buffer holding these attention
// offsets. They depend only on the head geometry, so every step asks
// for the same bytes; the buffer uploads once and lives with the
// device. The caller holds wgpuMu.
func (g *Device) offsBuffer(offs []uint32) uintptr {
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&offs[0])), len(offs)*4)
	key := string(raw)
	if b, ok := g.offsCache[key]; ok {
		return b
	}
	b := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	if b == 0 {
		return 0
	}
	fnQueueWriteBuffer(g.queue, b, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)
	if g.offsCache == nil {
		g.offsCache = make(map[string]uintptr)
	}
	g.offsCache[key] = b
	return b
}

// endBatchPass closes the batch's open compute pass, if any, so the
// encoder can record buffer copies or finish. The caller holds wgpuMu.
func (g *Device) endBatchPass() {
	if g.batchPass != 0 {
		fnPassEnd(g.batchPass)
		fnPassRelease(g.batchPass)
		g.batchPass = 0
	}
}

// flushLocked submits and closes an open batch encoder; the caller holds
// g.mu and wgpuMu.
func (g *Device) flushLocked() error {
	if g.batchEnc == 0 {
		return nil
	}
	g.endBatchPass()
	if g.uarOff > 0 {
		// One queue write covers every uniform the chunk staged; it is
		// enqueued before the submit, so the dispatches read fresh bytes.
		fnQueueWriteBuffer(g.queue, g.uarBuf, 0, unsafe.Pointer(&g.uarStage[0]), uintptr(g.uarOff))
		g.uarOff = 0
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
	free map[[2]uint64][]uintptr
	// batch holds tensor buffers freed while a batch is open. The encoded
	// passes execute in order with implicit barriers between them, so a
	// later dispatch may write such a buffer — but only a dispatch: queue
	// writes jump ahead of the unsubmitted encoder, so takeBuffer (which
	// feeds the queue-write sites) never draws from here; takeOutBuffer
	// (dispatch outputs) does. Flush folds batch into free.
	batch   map[[2]uint64][]uintptr
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
func (g *Device) takeBuffer(usage, size uint64) uintptr {
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
func (g *Device) putBuffer(usage, size uint64, buf uintptr) {
	if g.closed || g.pool.bytes+size > gpuPoolMaxBytes {
		g.dropBuffer(buf)
		return
	}
	if g.batchEnc != 0 {
		if usage == gpuTensorUsage {
			if g.pool.batch == nil {
				g.pool.batch = make(map[[2]uint64][]uintptr)
			}
			key := [2]uint64{usage, size}
			g.pool.batch[key] = append(g.pool.batch[key], buf)
			g.pool.bytes += size
			return
		}
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

// takeOutBuffer returns a tensor buffer a dispatch will write: unlike
// takeBuffer it may reuse one freed earlier in the still-open batch,
// which keeps a long prefill from allocating a fresh buffer (and, past
// the pool cap, wiping the bind-group cache) for every intermediate.
// The caller holds g.mu and wgpuMu.
func (g *Device) takeOutBuffer(size uint64) uintptr {
	key := [2]uint64{gpuTensorUsage, size}
	if l := g.pool.batch[key]; len(l) > 0 {
		buf := l[len(l)-1]
		g.pool.batch[key] = l[:len(l)-1]
		g.pool.bytes -= size
		return buf
	}
	return g.takeBuffer(gpuTensorUsage, size)
}

// drainPending moves batch-held buffers into the free lists once their
// batch has been submitted. The caller holds wgpuMu.
func (g *Device) drainPending() {
	if len(g.pool.batch) > 0 && g.pool.free == nil {
		g.pool.free = make(map[[2]uint64][]uintptr)
	}
	for key, l := range g.pool.batch {
		g.pool.free[key] = append(g.pool.free[key], l...)
		delete(g.pool.batch, key)
	}
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
func (g *Device) releasePool() {
	for _, l := range g.pool.free {
		for _, buf := range l {
			fnBufferRelease(buf)
		}
	}
	for _, l := range g.pool.batch {
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
	e      [8]struct {
		binding uint32
		buf     uintptr
		off     uint64
		size    uint64
	}
}

// bgCacheMax caps the cache; overflow drops everything (simple, and far
// beyond what a decode loop creates).
const bgCacheMax = 8192

// cachedBindGroup returns a bind group for the entries, creating and
// retaining it on first use. Cached groups are owned by the cache — the
// caller must not release them. The caller holds g.mu and wgpuMu.
func (g *Device) cachedBindGroup(layout uintptr, entries []wgpuBindGroupEntry) uintptr {
	var key bgKey
	key.layout = layout
	key.n = len(entries)
	for i, e := range entries {
		key.e[i].binding = e.binding
		key.e[i].buf = e.buffer
		key.e[i].off = e.offset
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
func (g *Device) invalidateBindGroups() {
	for _, bg := range g.bgCache {
		fnBindGroupRelease(bg)
	}
	g.bgCache = nil
}

// dropBuffer releases a buffer for real (not into the pool) and
// invalidates the bind-group cache. The caller holds wgpuMu.
func (g *Device) dropBuffer(buf uintptr) {
	fnBufferRelease(buf)
	g.invalidateBindGroups()
}

// Tensor is a tensor whose data lives in Device memory. Create one with
// Device.Upload or as the result of a Tensor operation, read it back with
// Download, and release it with Free — Device memory is not garbage
// collected.
type Tensor struct {
	g     *Device
	buf   uintptr
	shape []int
	freed bool
	f16   bool // half-precision storage (KV caches); most kernels want f32
	// off is a byte offset into buf. Non-zero only for a View, which
	// borrows a window of another tensor's buffer rather than owning one;
	// every kernel reaches its operand through bind/bindN, so the offset
	// applies uniformly and no call site can forget it.
	off  uint64
	view bool
}

// bind builds a bind-group entry covering the whole of t, and bindN one
// covering the first n bytes. Every kernel binds its tensor operands
// through these two so a View's offset is applied in one place instead of
// at two dozen call sites.
func bind(binding uint32, t *Tensor) wgpuBindGroupEntry {
	return bindN(binding, t, t.byteLen())
}

func bindN(binding uint32, t *Tensor, bytes uint64) wgpuBindGroupEntry {
	return wgpuBindGroupEntry{binding: binding, buffer: t.buf, offset: t.off, size: bytes}
}

// View borrows a window of t's buffer as a tensor of its own: the fused
// QKV projection lands in one buffer and q, k, and v are read back out of
// it without a copy. The offset must be 256-byte aligned, which every
// WebGPU device accepts as a storage-buffer binding offset. A view does
// not own the buffer -- Free on it is a no-op, and it must not outlive
// the tensor it borrows from.
func (t *Tensor) View(off int, shape ...int) (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if t.f16 {
		return nil, errors.New("tensai: cannot view a half-precision tensor")
	}
	n := dims.Prod(shape)
	if off < 0 || n <= 0 || off+n > t.Size() {
		return nil, fmt.Errorf("tensai: view of %d elements at %d overflows %d", n, off, t.Size())
	}
	if (uint64(off)*4)%256 != 0 {
		return nil, fmt.Errorf("tensai: view offset %d is not 256-byte aligned", off)
	}
	return &Tensor{
		g:     t.g,
		buf:   t.buf,
		shape: append([]int(nil), shape...),
		off:   t.off + uint64(off)*4,
		view:  true,
	}, nil
}

// SliceCols lifts a column window out of every row into a tensor of its
// own: t is [rows, stride] and the result [rows, cols] holding columns
// [off, off+cols). View covers the case where the window is one
// contiguous run of the buffer; this is the case where it repeats per
// row, which no binding offset can describe, so it costs a copy.
func (t *Tensor) SliceCols(off, cols int) (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	n := len(t.shape)
	if n < 2 {
		return nil, errors.New("tensai: gpu slice needs at least 2 axes")
	}
	stride := t.shape[n-1]
	if off < 0 || cols <= 0 || off+cols > stride {
		return nil, fmt.Errorf("tensai: slice of %d columns at %d overflows %d", cols, off, stride)
	}
	if t.f16 {
		return nil, errors.New("tensai: cannot slice a half-precision tensor")
	}
	rows := t.Size() / stride
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(rows*cols) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	params := [4]uint32{uint32(rows), uint32(cols), uint32(stride), uint32(off)}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)
	bufOut := g.takeOutBuffer(outBytes)

	entries := [3]wgpuBindGroupEntry{
		{binding: 36, buffer: bufParams, size: 16},
		bind(37, t),
		{binding: 38, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.laySliceCols, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((rows*cols + 255) / 256)
	if err := g.dispatch(g.pipes.sliceCols, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	shape := append(append([]int(nil), t.shape[:n-1]...), cols)
	return &Tensor{g: g, buf: bufOut, shape: shape}, nil
}

// byteLen is the tensor's buffer size, which the pool keys on.
func (t *Tensor) byteLen() uint64 {
	if t.f16 {
		return (uint64(t.Size())*2 + 3) &^ 3
	}
	return uint64(t.Size()) * 4
}

// HasF16 reports whether the device granted shader-f16 and the f16
// kernels compiled: NewF16Tensor and the half-precision KV cache path
// are usable only then.
func (g *Device) HasF16() bool { return g.hasF16 }

// NewF16Tensor allocates a zeroed half-precision Device tensor. Only the
// attention cache kernels read and write f16 tensors: fill it with
// CopyRowsInto, feed it to GroupedCausalAttention.
// NewZeroTensor allocates a float32 tensor on the device without
// sending anything: a buffer starts zeroed, so a KV cache -- which is
// gigabytes of it, and every row written before it is read -- has no
// reason to be uploaded from the host at all.
func (g *Device) NewZeroTensor(shape ...int) (*Tensor, error) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if n <= 0 {
		return nil, errors.New("tensai: empty tensor")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	t := &Tensor{g: g, shape: append([]int(nil), shape...)}
	t.buf = g.newBuffer(gpuTensorUsage, t.byteLen())
	if t.buf == 0 {
		return nil, errors.New("tensai: gpu buffer allocation failed")
	}
	return t, nil
}

func (g *Device) NewF16Tensor(shape ...int) (*Tensor, error) {
	if !g.hasF16 {
		return nil, errors.New("tensai: device has no shader-f16 support")
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	if n <= 0 {
		return nil, errors.New("tensai: empty f16 tensor")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	t := &Tensor{g: g, shape: append([]int(nil), shape...), f16: true}
	t.buf = g.newBuffer(gpuTensorUsage, t.byteLen())
	if t.buf == 0 {
		return nil, errors.New("tensai: gpu buffer allocation failed")
	}
	return t, nil
}

// Every Tensor buffer is storage-usable (kernel input and output),
// copyable in (Upload) and out (Download).
const gpuTensorUsage = wgpuBufferUsageStorage | wgpuBufferUsageCopySrc | wgpuBufferUsageCopyDst

// gpuReadbackBuffer is the reusable MapRead staging buffer. Downloads are
// serialized by Device.mu and wgpuMu, and mapRead waits for the copy to finish,
// so the buffer is idle again before it is returned to this slot.
type gpuReadbackBuffer struct {
	buf  uintptr
	size uint64
}

// takeReadback returns an unmapped staging buffer at least bytes large. The
// single retained buffer grows to the largest download seen by this Device.
// The caller holds Device.mu and wgpuMu.
func (g *Device) takeReadback(bytes uint64) (uintptr, uint64) {
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
// download. The caller holds Device.mu and wgpuMu.
func (g *Device) putReadback(buf uintptr, size uint64) {
	if g.readback.buf != 0 {
		g.dropBuffer(g.readback.buf)
	}
	g.readback = gpuReadbackBuffer{buf: buf, size: size}
}

// releaseReadback drops the retained staging buffer during Device.Close. The
// caller holds wgpuMu.
func (g *Device) releaseReadback() {
	if g.readback.buf != 0 {
		fnBufferRelease(g.readback.buf)
	}
	g.readback = gpuReadbackBuffer{}
}

// StorageLimit reports how many bytes a single Device buffer may hold under
// the device limits negotiated at Open time (0 when unknown). tensai.Tensor
// operations return an error instead of touching the driver when a buffer
// would exceed it.
func (g *Device) StorageLimit() uint64 { return g.maxStorage }

func (g *Device) checkSize(bytes uint64) error {
	if g.maxStorage != 0 && bytes > g.maxStorage {
		return fmt.Errorf("tensai: gpu buffer of %d bytes exceeds the device storage limit of %d", bytes, g.maxStorage)
	}
	return nil
}

// Shape returns a copy of the tensor's shape.
func (t *Tensor) Shape() []int { return append([]int(nil), t.shape...) }

// Size returns the total number of elements.
func (t *Tensor) Size() int { return dims.Prod(t.shape) }

// Upload copies a tensor into Device memory. The returned Tensor is
// independent of t.
func (g *Device) Upload(t *tensai.Tensor) (*Tensor, error) {
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
	return &Tensor{g: g, buf: buf, shape: append([]int(nil), t.Shape...)}, nil
}

// Download copies the tensor back into host memory.
func (t *Tensor) Download() (*tensai.Tensor, error) {
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
	out := tensai.NewTensor(t.shape...)
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
	fnEncoderCopyBuffer(encoder, t.buf, t.off, staging, 0, bytes)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)

	src, err := g.mapRead(staging, bytes)
	if err != nil {
		return nil, err
	}
	copy(out.Data, unsafe.Slice((*tensai.Float)(src), len(out.Data)))
	fnBufferUnmap(staging)
	reuse = true
	return out, nil
}

// DownloadRange copies elements [off, off+n) back into host memory as a
// flat tensor — reading freshly appended rows out of a resident cache
// without moving the whole buffer. Like Download it flushes an open
// batch first.
func (t *Tensor) DownloadRange(off, n int) (*tensai.Tensor, error) {
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
	out := tensai.NewTensor(n)
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
	fnEncoderCopyBuffer(encoder, t.buf, t.off+uint64(off)*4, staging, 0, bytes)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)
	src, err := g.mapRead(staging, bytes)
	if err != nil {
		return nil, err
	}
	copy(out.Data, unsafe.Slice((*tensai.Float)(src), n))
	fnBufferUnmap(staging)
	reuse = true
	return out, nil
}

// Free releases the Device buffer (into the transient pool, from which the
// next same-sized tensor will reuse it). The tensor must not be used
// afterwards; calling Free again is a no-op.
func (t *Tensor) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if t.freed {
		return
	}
	t.freed = true
	if t.view { // borrowed, not owned: the owner still holds the buffer
		return
	}
	t.g.putBuffer(gpuTensorUsage, t.byteLen(), t.buf)
}

// MatMul multiplies two Device-resident tensors with the same shape and
// broadcasting semantics as the package-level MatMul, returning a new
// Device-resident tensor without any host transfer. Chain calls freely; only
// Download moves data back.
func (t *Tensor) MatMul(o *Tensor) (*Tensor, error) {
	return t.matmul(o, gemmNN)
}

// MatMulT multiplies t by o with o's last two axes read transposed: a
// (batch..., m, k) tensor times a (batch..., n, k) tensor yields
// (batch..., m, n), without materializing the transpose. This is the
// attention pattern q @ k^T.
func (t *Tensor) MatMulT(o *Tensor) (*Tensor, error) {
	return t.matmul(o, gemmNT)
}

// MatMulTN multiplies t by o with t's last two axes read transposed: a
// (batch..., k, m) tensor times a (batch..., k, n) tensor yields
// (batch..., m, n). This is the weight gradient of a matmul -- the input
// activation transposed times the output gradient -- again without
// materializing a transpose.
func (t *Tensor) MatMulTN(o *Tensor) (*Tensor, error) {
	return t.matmul(o, gemmTN)
}

// gemmMode selects which operand of a product is read transposed, matching
// the three MatMul entry points in the root package.
type gemmMode int

const (
	gemmNN gemmMode = iota // a * b
	gemmTN                 // a^T * b
	gemmNT                 // a * b^T
)

func (t *Tensor) matmul(o *Tensor, mode gemmMode) (*Tensor, error) {
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
	// The stored geometry of t; the logical product is (m, k) * (k, n).
	aRows, aCols := t.shape[na-2], t.shape[na-1]
	m, k := aRows, aCols
	if mode == gemmTN {
		k, m = aRows, aCols
	}
	var n int
	switch mode {
	case gemmNT:
		if o.shape[nb-1] != k {
			return nil, fmt.Errorf("tensai: matmul-t shape mismatch: %v * %v^T", t.shape, o.shape)
		}
		n = o.shape[nb-2]
	case gemmTN:
		if o.shape[nb-2] != k {
			return nil, fmt.Errorf("tensai: matmul-tn shape mismatch: (%v)^T * %v", t.shape, o.shape)
		}
		n = o.shape[nb-1]
	default:
		if o.shape[nb-2] != k {
			return nil, fmt.Errorf("tensai: matmul shape mismatch: %v * %v", t.shape, o.shape)
		}
		n = o.shape[nb-1]
	}
	batch, err := dims.Broadcast(t.shape[:na-2], o.shape[:nb-2])
	if err != nil {
		return nil, err
	}
	batches := dims.Prod(batch)
	outShape := append(append([]int(nil), batch...), m, n)

	// Element offsets of each batch's (contiguous) matrix in t, o, out.
	as := dims.BroadcastStrides(t.shape[:na-2], batch)
	bs := dims.BroadcastStrides(o.shape[:nb-2], batch)
	offs := make([]uint32, 4*batches)
	for bi := 0; bi < batches; bi++ {
		offA, offB := 0, 0
		for d, rem := len(batch)-1, bi; d >= 0; d-- {
			i := rem % batch[d]
			rem /= batch[d]
			offA += i * as[d]
			offB += i * bs[d]
		}
		offs[4*bi] = uint32(offA * aRows * aCols)
		offs[4*bi+1] = uint32(offB * o.shape[nb-2] * o.shape[nb-1])
		offs[4*bi+2] = uint32(bi * m * n)
	}
	// Row strides of the operands as they are stored.
	lda, ldb := aCols, n
	if mode == gemmNT {
		ldb = k
	}
	return t.g.stridedMatMul(t, o, outShape, mode, m, k, n, batches, lda, ldb, n, offs)
}

// stridedMatMul runs `batches` independent (m x k) x (k x n) products —
// x (n x k) read transposed when transB — where offs holds per-batch
// element offsets (offA, offB, offC, 0) into a, b, and the freshly
// allocated output, and lda/ldb/ldc are the row strides. Explicit strides
// let callers carve sub-matrices out of a wider layout, which is how
// multi-head attention splits heads without materializing a permute.
func (g *Device) stridedMatMul(a, b *Tensor, outShape []int, mode gemmMode, m, k, n, batches, lda, ldb, ldc int, offs []uint32) (*Tensor, error) {
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

	outBytes := uint64(dims.Prod(outShape)) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOffs := g.takeBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.takeOutBuffer(outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	defer g.putBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4, bufOffs)

	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	// The 64x64 register-tiled kernels win on big products. Skinny ones
	// (m or n under half a block) would leave most of each workgroup
	// idle, and products with only a handful of 64x64 blocks cannot fill
	// the Device at all — both take the plain 16x16 variants, whose grid has
	// sixteen times the workgroups.
	block := gpuBlock
	pipe, lay := g.pipes.matmulL, g.pipes.layMatmulL
	switch mode {
	case gemmNT:
		pipe, lay = g.pipes.matmulLT, g.pipes.layMatmulLT
	case gemmTN:
		pipe, lay = g.pipes.matmulLTN, g.pipes.layMatmulLTN
	}
	// The vec4 loads need every address they widen to be quad-aligned, and
	// which dimension each mode widens along differs.
	// A vec4 binding also has to cover a whole number of quads, which a
	// view carved out of a packed layout need not.
	wide := !noWideGemm && lda%4 == 0 && ldb%4 == 0 &&
		a.Size()%4 == 0 && b.Size()%4 == 0 && quadAligned(offs)
	switch {
	case !wide:
	case mode == gemmNN && k%4 == 0 && n%4 == 0:
		pipe, lay = g.pipes.matmulV4, g.pipes.layMatmulV4
	case mode == gemmNT && k%4 == 0:
		pipe, lay = g.pipes.matmulV4T, g.pipes.layMatmulV4T
	case mode == gemmTN && m%4 == 0 && n%4 == 0:
		pipe, lay = g.pipes.matmulV4TN, g.pipes.layMatmulV4TN
	default:
		wide = false
	}
	blocks := ((m + gpuBlock - 1) / gpuBlock) * ((n + gpuBlock - 1) / gpuBlock) * batches
	if m < 32 || n < 32 || blocks < 16 {
		// The small kernels read the operands through the scalar bindings,
		// so the wide ones must not be bound with them.
		block = 16
		wide = false
		pipe, lay = g.pipes.matmulS, g.pipes.layMatmulS
		switch mode {
		case gemmNT:
			pipe, lay = g.pipes.matmulTS, g.pipes.layMatmulTS
		case gemmTN:
			pipe, lay = g.pipes.matmulTNS, g.pipes.layMatmulTNS
		}
	}
	// A bind group has to match its pipeline's layout exactly, and the
	// vec4 kernel reads the operands only through the wide bindings.
	entries := make([]wgpuBindGroupEntry, 0, 5)
	entries = append(entries, wgpuBindGroupEntry{binding: 0, buffer: bufParams, size: 32})
	if wide {
		entries = append(entries, bind(56, a), bind(57, b))
	} else {
		entries = append(entries, bind(1, a), bind(2, b))
	}
	entries = append(entries,
		wgpuBindGroupEntry{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		wgpuBindGroupEntry{binding: 4, buffer: bufOut, size: outBytes})
	bindGroup := g.cachedBindGroup(lay, entries)
	runtime.KeepAlive(&entries)

	err := g.dispatch(pipe, bindGroup,
		uint32((n+block-1)/block), uint32((m+block-1)/block), uint32(batches))
	if err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: outShape}, nil
}

// noWideGemm turns the vec4-load kernel off, so a benchmark can compare it
// against the scalar-load one in the same process.
var noWideGemm bool

// quadAligned reports whether every batch offset a kernel will divide by
// four actually is a multiple of four.
func quadAligned(offs []uint32) bool {
	for i := 0; i+1 < len(offs); i += 4 {
		if offs[i]%4 != 0 || offs[i+1]%4 != 0 {
			return false
		}
	}
	return true
}

// split2D spreads n workgroups over a 2-D dispatch grid, since a single
// axis is capped at 65535.
func split2D(n int) (x, y uint32) {
	if n <= 65535 {
		return uint32(n), 1
	}
	return 65535, uint32((n + 65534) / 65535)
}

// Scale multiplies every element by s, in place — the Device counterpart of
// tensai.Tensor.Scale.
func (t *Tensor) Scale(s tensai.Float) error {
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
		bindN(6, t, uint64(count)*4),
	}
	bindGroup := g.cachedBindGroup(g.pipes.layScale, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((count + gpuKernelWG - 1) / gpuKernelWG)
	return g.dispatch(g.pipes.scale, bindGroup, x, y, 1)
}

// Softmax applies a numerically stable softmax over the last axis,
// returning a new Device-resident tensor.
func (t *Tensor) Softmax() (*Tensor, error) {
	return t.softmax(0, 0)
}

// softmax optionally applies a causal mask: with qmod > 0, row r is query
// index r%qmod and attends to the first r%qmod+off+1 columns; masked
// columns come out as exactly zero.
func (t *Tensor) softmax(qmod, off int) (*Tensor, error) {
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
	bufOut := g.takeOutBuffer(bytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [3]wgpuBindGroupEntry{
		{binding: 7, buffer: bufParams, size: 16},
		bindN(8, t, bytes),
		{binding: 9, buffer: bufOut, size: bytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.laySoftmax, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.softmax, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, nil
}

// Attention computes scaled dot-product attention softmax(q*k^T/sqrt(d))*v
// entirely on the Device — the resident counterpart of the autograd Attention.
// q, k, v are (batch..., seqLen, d) tensors; nothing touches host memory.
func (q *Tensor) Attention(k, v *Tensor) (*Tensor, error) {
	return q.attention(k, v, false)
}

// CausalAttention is Attention with a causal mask: query i attends only to
// key positions 0..i+(seqKV-seqQ), so k and v may hold seqKV >= seqQ
// positions with the queries aligned to their end — the prompt-prefill
// pattern of autoregressive models.
func (q *Tensor) CausalAttention(k, v *Tensor) (*Tensor, error) {
	return q.attention(k, v, true)
}

func (q *Tensor) attention(k, v *Tensor, causal bool) (*Tensor, error) {
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
	if err := scores.Scale(1 / kernels.SqrtF(tensai.Float(q.shape[nq-1]))); err != nil {
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
// entirely on the Device. q is (batch..., seqQ, d) and k, v are
// (batch..., seqKV, d) in the usual packed head layout, d = heads * dh;
// the result is (batch..., seqQ, d). Heads are carved out of the packed
// layout with strided kernels, so no permute is ever materialized.
func (q *Tensor) MultiHeadAttention(k, v *Tensor, heads int) (*Tensor, error) {
	return q.multiHeadAttention(k, v, heads, false)
}

// CausalMultiHeadAttention is MultiHeadAttention with a causal mask: query
// i attends only to key positions 0..i+(seqKV-seqQ), so k and v may hold
// seqKV >= seqQ positions with the queries aligned to their end — the
// prompt-prefill pattern of autoregressive models.
func (q *Tensor) CausalMultiHeadAttention(k, v *Tensor, heads int) (*Tensor, error) {
	return q.multiHeadAttention(k, v, heads, true)
}

func (q *Tensor) multiHeadAttention(k, v *Tensor, heads int, causal bool) (*Tensor, error) {
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
	if !dims.Same(k.shape, v.shape) {
		return nil, fmt.Errorf("tensai: attention k and v shapes differ: %v vs %v", k.shape, v.shape)
	}
	if len(k.shape) != nq || k.shape[nq-1] != d || !dims.Same(k.shape[:nq-2], q.shape[:nq-2]) {
		return nil, fmt.Errorf("tensai: attention shape mismatch: q %v, k %v", q.shape, k.shape)
	}
	seq := q.shape[nq-2]
	seqKV := k.shape[nq-2]
	if causal && seqKV < seq {
		return nil, fmt.Errorf("tensai: causal attention needs seqKV >= seqQ, got %d < %d", seqKV, seq)
	}
	batch := dims.Prod(q.shape[:nq-2])
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
	scores, err := q.g.stridedMatMul(q, k, []int{bh, seq, seqKV}, gemmNT,
		seq, dh, seqKV, bh, d, d, seqKV, offs)
	if err != nil {
		return nil, err
	}
	defer scores.Free()
	if err := scores.Scale(1 / kernels.SqrtF(tensai.Float(dh))); err != nil {
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
	return q.g.stridedMatMul(weights, v, outShape, gemmNN,
		seq, seqKV, dh, bh, seqKV, d, d, offs2)
}

// fusedCausalMHA runs causal multi-head attention as one flash-attention
// style dispatch: the scores matrix is never materialized, so memory use
// is just q, k, v, and the output, independent of sequence length.
func (q *Tensor) fusedCausalMHA(k, v *Tensor, heads, batch, seq, seqKV, d, dh, bh int) (*Tensor, error) {
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

	outBytes := uint64(dims.Prod(outShape)) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufOffs := g.takeBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.takeOutBuffer(outBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	defer g.putBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4, bufOffs)

	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	entries := [6]wgpuBindGroupEntry{
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 10, buffer: bufParams, size: 32},
		bind(11, q),
		bind(12, k),
		bind(13, v),
		{binding: 14, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layAttn, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.attn, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: outShape}, nil
}

// MatMul is the Device version of the package-level MatMul: identical shape
// and broadcasting semantics, executed as one compute dispatch. Inputs and
// the result live in host memory; each call uploads the operands and reads
// the product back, so it only pays off for large products — keep operands
// resident with Upload / Tensor.MatMul to amortize the transfers.
// BinOp names an element-wise binary operation the device can run.
type BinOp uint32

const (
	OpAdd BinOp = iota
	OpSub
	OpMul
	OpDiv
)

// Act names an activation the device can run, forward or backward.
type Act uint32

const (
	ActReLU Act = iota
	ActTanh
	ActSigmoid
	ActGELU
)

// trainParams is the uniform every training kernel reads.
type trainParams struct {
	count, aCount, bCount, mode uint32
	rows                        uint32
	lr, beta1, beta2            float32
	rc1, rc2, eps, decay        float32
}

const trainParamBytes = 48

// trainOp runs one element-wise training kernel over count elements,
// allocating the (outShape) result it writes. a binds to 47, b -- which may
// be nil -- to 48, and the result to 49.
func (g *Device) trainOp(pipe, lay uintptr, p trainParams, count int, outShape []int, a, b *Tensor) (*Tensor, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(dims.Prod(outShape)) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&p), trainParamBytes)
	bufOut := g.takeOutBuffer(outBytes)

	entries := make([]wgpuBindGroupEntry, 0, 4)
	entries = append(entries,
		wgpuBindGroupEntry{binding: 46, buffer: bufParams, size: trainParamBytes},
		bind(47, a))
	if b != nil {
		entries = append(entries, bind(48, b))
	}
	entries = append(entries, wgpuBindGroupEntry{binding: 49, buffer: bufOut, size: outBytes})
	bindGroup := g.cachedBindGroup(lay, entries)
	runtime.KeepAlive(&entries)

	x, y := split2D((count + 255) / 256)
	if err := g.dispatch(pipe, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), outShape...)}, nil
}

// broadcastCount checks that n elements repeat cyclically into count, which
// is the broadcast the kernels can express.
func broadcastCount(count, n int) (uint32, error) {
	if n == 0 || count%n != 0 {
		return 0, fmt.Errorf("tensai: gpu cannot broadcast %d elements into %d", n, count)
	}
	return uint32(n), nil
}

// Binary returns t op o element-wise. o may be shorter than t as long as it
// divides it, in which case it repeats -- the trailing-axis broadcast of a
// bias or a per-feature scale.
func (t *Tensor) Binary(op BinOp, o *Tensor) (*Tensor, error) {
	if t.freed || o.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if t.g != o.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	count := t.Size()
	bCount, err := broadcastCount(count, o.Size())
	if err != nil {
		return nil, err
	}
	p := trainParams{count: uint32(count), aCount: uint32(count), bCount: bCount, mode: uint32(op)}
	return t.g.trainOp(t.g.pipes.binOp, t.g.pipes.layBinOp, p, count, t.shape, t, o)
}

// Activate applies an activation element-wise.
func (t *Tensor) Activate(a Act) (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	count := t.Size()
	p := trainParams{count: uint32(count), aCount: uint32(count), bCount: uint32(count), mode: uint32(a)}
	return t.g.trainOp(t.g.pipes.actFwd, t.g.pipes.layActFwd, p, count, t.shape, t, nil)
}

// ActivateGrad turns an output gradient into the input gradient of an
// activation: t holds the pre-activation input the forward pass saw.
func (t *Tensor) ActivateGrad(a Act, grad *Tensor) (*Tensor, error) {
	if t.freed || grad.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if t.g != grad.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	count := t.Size()
	if grad.Size() != count {
		return nil, fmt.Errorf("tensai: activation gradient has %d elements, want %d", grad.Size(), count)
	}
	p := trainParams{count: uint32(count), aCount: uint32(count), bCount: uint32(count), mode: uint32(a)}
	return t.g.trainOp(t.g.pipes.actBwd, t.g.pipes.layActBwd, p, count, t.shape, t, grad)
}

// LayerNorm normalizes the last axis of a (rows, cols) tensor and applies
// a per-feature gain and bias, both of which must hold cols elements.
func (t *Tensor) LayerNorm(gain, bias *Tensor, eps tensai.Float) (*Tensor, error) {
	rows, cols, err := t.rowsCols()
	if err != nil {
		return nil, err
	}
	if gain.Size() != cols || bias.Size() != cols {
		return nil, fmt.Errorf("tensai: layernorm gain and bias must hold %d elements", cols)
	}
	p := trainParams{count: uint32(cols), rows: uint32(rows), eps: float32(eps)}
	return t.g.trainRows(t.g.pipes.lnFwd, t.g.pipes.layLnFwd, p, rows, t.shape, t, gain, bias)
}

// LayerNormXhat is the normalization without the affine part -- the
// backward pass builds the gain's gradient from it.
func (t *Tensor) LayerNormXhat(eps tensai.Float) (*Tensor, error) {
	rows, cols, err := t.rowsCols()
	if err != nil {
		return nil, err
	}
	p := trainParams{count: uint32(cols), rows: uint32(rows), eps: float32(eps)}
	return t.g.trainRows(t.g.pipes.lnXhat, t.g.pipes.layLnXhat, p, rows, t.shape, t, nil, nil)
}

// LayerNormGrad turns the output gradient into the input gradient of a
// LayerNorm whose input was t and whose gain was gain.
func (t *Tensor) LayerNormGrad(grad, gain *Tensor, eps tensai.Float) (*Tensor, error) {
	rows, cols, err := t.rowsCols()
	if err != nil {
		return nil, err
	}
	if grad.Size() != t.Size() {
		return nil, fmt.Errorf("tensai: layernorm gradient has %d elements, want %d", grad.Size(), t.Size())
	}
	if gain.Size() != cols {
		return nil, fmt.Errorf("tensai: layernorm gain must hold %d elements", cols)
	}
	p := trainParams{count: uint32(cols), rows: uint32(rows), eps: float32(eps)}
	return t.g.trainRows(t.g.pipes.lnBwd, t.g.pipes.layLnBwd, p, rows, t.shape, t, grad, gain)
}

// SoftmaxGrad turns the output gradient into the input gradient of a
// softmax whose output is t.
func (t *Tensor) SoftmaxGrad(grad *Tensor) (*Tensor, error) {
	rows, cols, err := t.rowsCols()
	if err != nil {
		return nil, err
	}
	if grad.Size() != t.Size() {
		return nil, fmt.Errorf("tensai: softmax gradient has %d elements, want %d", grad.Size(), t.Size())
	}
	p := trainParams{count: uint32(cols), rows: uint32(rows)}
	return t.g.trainRows(t.g.pipes.softmaxBwd, t.g.pipes.laySoftmaxBwd, p, rows, t.shape, t, grad, nil)
}

// rowsCols splits a tensor into rows of its last axis.
func (t *Tensor) rowsCols() (int, int, error) {
	if t.freed {
		return 0, 0, errors.New("tensai: gpu tensor already freed")
	}
	n := len(t.shape)
	if n < 1 {
		return 0, 0, fmt.Errorf("tensai: row op needs at least one axis, got %v", t.shape)
	}
	cols := t.shape[n-1]
	if cols == 0 {
		return 0, 0, fmt.Errorf("tensai: row op on an empty tensor %v", t.shape)
	}
	return t.Size() / cols, cols, nil
}

// trainRows runs a kernel that takes one workgroup per row. b and c may be
// nil; they bind to 48 and 52 when they are not.
func (g *Device) trainRows(pipe, lay uintptr, p trainParams, rows int, outShape []int, a, b, c *Tensor) (*Tensor, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(dims.Prod(outShape)) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&p), trainParamBytes)
	bufOut := g.takeOutBuffer(outBytes)

	entries := make([]wgpuBindGroupEntry, 0, 5)
	entries = append(entries,
		wgpuBindGroupEntry{binding: 46, buffer: bufParams, size: trainParamBytes},
		bind(47, a))
	if b != nil {
		entries = append(entries, bind(48, b))
	}
	entries = append(entries, wgpuBindGroupEntry{binding: 49, buffer: bufOut, size: outBytes})
	if c != nil {
		entries = append(entries, bind(52, c))
	}
	bindGroup := g.cachedBindGroup(lay, entries)
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(pipe, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), outShape...)}, nil
}

// UploadIndices puts token indices on the device for Embed and EmbedGrad.
// They are u32 rather than f32; the tensor is a handle for those two calls
// and not something to compute with.
func (g *Device) UploadIndices(ids []int) (*Tensor, error) {
	if len(ids) == 0 {
		return nil, errors.New("tensai: cannot upload an empty index list")
	}
	raw := make([]uint32, len(ids))
	for i, id := range ids {
		if id < 0 {
			return nil, fmt.Errorf("tensai: negative index %d", id)
		}
		raw[i] = uint32(id)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	bytes := uint64(len(raw)) * 4
	if err := g.checkSize(bytes); err != nil {
		return nil, err
	}
	buf := g.takeBuffer(gpuTensorUsage, bytes)
	if buf == 0 {
		return nil, errors.New("tensai: gpu buffer allocation failed")
	}
	fnQueueWriteBuffer(g.queue, buf, 0, unsafe.Pointer(&raw[0]), uintptr(bytes))
	runtime.KeepAlive(raw)
	return &Tensor{g: g, buf: buf, shape: []int{len(raw)}}, nil
}

// Embed looks every index up in t, a (vocab, dim) table, and returns the
// rows stacked in order.
func (t *Tensor) Embed(ids *Tensor) (*Tensor, error) {
	if t.freed || ids.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("tensai: embed needs a (vocab, dim) table, got %v", t.shape)
	}
	dim := t.shape[1]
	rows := ids.Size()
	count := rows * dim
	p := trainParams{count: uint32(count), aCount: uint32(dim), bCount: uint32(rows)}
	return t.g.embedOp(t.g.pipes.embedGather, t.g.pipes.layEmbedGather, p, count,
		[]int{rows, dim}, t, ids, nil)
}

// EmbedGrad adds each row of grad into the row of t its index names, which
// is the backward pass of Embed. t is the table's gradient, and repeated
// indices accumulate.
func (t *Tensor) EmbedGrad(grad, ids *Tensor) error {
	if t.freed || grad.freed || ids.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	if len(t.shape) != 2 {
		return fmt.Errorf("tensai: embed gradient needs a (vocab, dim) table, got %v", t.shape)
	}
	dim := t.shape[1]
	rows := ids.Size()
	if grad.Size() != rows*dim {
		return fmt.Errorf("tensai: embed gradient has %d elements, want %d", grad.Size(), rows*dim)
	}
	p := trainParams{count: uint32(rows * dim), aCount: uint32(dim), bCount: uint32(rows)}
	_, err := t.g.embedOp(t.g.pipes.embedScatter, t.g.pipes.layEmbedScatter, p, rows*dim,
		nil, grad, ids, t)
	return err
}

// embedOp dispatches one of the two embedding kernels. outShape is nil for
// the scatter, which accumulates into the existing buffer bound as atom.
func (g *Device) embedOp(pipe, lay uintptr, p trainParams, count int, outShape []int, a, ids, atom *Tensor) (*Tensor, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&p), trainParamBytes)

	entries := make([]wgpuBindGroupEntry, 0, 4)
	entries = append(entries,
		wgpuBindGroupEntry{binding: 46, buffer: bufParams, size: trainParamBytes},
		bind(47, a),
		bind(54, ids))
	var bufOut uintptr
	var out *Tensor
	if outShape != nil {
		outBytes := uint64(dims.Prod(outShape)) * 4
		if err := g.checkSize(outBytes); err != nil {
			return nil, err
		}
		bufOut = g.takeOutBuffer(outBytes)
		entries = append(entries, wgpuBindGroupEntry{binding: 49, buffer: bufOut, size: outBytes})
		out = &Tensor{g: g, buf: bufOut, shape: append([]int(nil), outShape...)}
	} else {
		entries = append(entries, bind(55, atom))
	}
	bindGroup := g.cachedBindGroup(lay, entries)
	runtime.KeepAlive(&entries)

	x, y := split2D((count + 255) / 256)
	if err := g.dispatch(pipe, bindGroup, x, y, 1); err != nil {
		if bufOut != 0 {
			g.dropBuffer(bufOut)
		}
		return nil, err
	}
	return out, nil
}

// Permute reorders the axes of a tensor of rank up to four: perm[i] names
// the axis of t that becomes axis i of the result, the way
// tensai.Tensor.Transpose reads it.
func (t *Tensor) Permute(perm ...int) (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	n := len(t.shape)
	if n < 2 || n > 4 || len(perm) != n {
		return nil, fmt.Errorf("tensai: gpu permute %v of shape %v is not supported", perm, t.shape)
	}
	seen := make([]bool, n)
	for _, p := range perm {
		if p < 0 || p >= n || seen[p] {
			return nil, fmt.Errorf("tensai: invalid permutation %v", perm)
		}
		seen[p] = true
	}
	// Contiguous strides of the source, then the stride to walk for each
	// axis of the output.
	src := make([]int, n)
	stride := 1
	for i := n - 1; i >= 0; i-- {
		src[i] = stride
		stride *= t.shape[i]
	}
	outShape := make([]int, n)
	var params struct {
		count, rank, pad0, pad1 uint32
		shape                   [4]uint32
		stride                  [4]uint32
	}
	for i, p := range perm {
		outShape[i] = t.shape[p]
		params.shape[i] = uint32(t.shape[p])
		params.stride[i] = uint32(src[p])
	}
	params.count = uint32(t.Size())
	params.rank = uint32(n)

	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""
	const permuteParamBytes = 48
	outBytes := uint64(t.Size()) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, permuteParamBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, permuteParamBytes, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params), permuteParamBytes)
	bufOut := g.takeOutBuffer(outBytes)

	entries := [3]wgpuBindGroupEntry{
		{binding: 53, buffer: bufParams, size: permuteParamBytes},
		bind(47, t),
		{binding: 49, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layPermute, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((t.Size() + 255) / 256)
	if err := g.dispatch(g.pipes.permute, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: outShape}, nil
}

// SumCols reduces a (rows, cols) tensor to a (1, cols) row -- the gradient
// a value broadcast over the rows collects.
func (t *Tensor) SumCols() (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("tensai: sumcols needs a 2-D tensor, got %v", t.shape)
	}
	rows, cols := t.shape[0], t.shape[1]
	p := trainParams{count: uint32(cols), aCount: uint32(rows * cols), bCount: uint32(cols), rows: uint32(rows)}
	return t.g.trainOp(t.g.pipes.sumCols, t.g.pipes.laySumCols, p, cols, []int{1, cols}, t, nil)
}

// AdamStep applies one Adam update to t in place, with the gradient in
// grad and the moment estimates in m and v (both the same size as t, and
// both updated). rc1 and rc2 are the bias corrections for the step count,
// and decay is AdamW's decoupled weight decay. The arithmetic matches the
// CPU kernel, so a model can move between the two mid-training.
func (t *Tensor) AdamStep(grad, m, v *Tensor, lr, beta1, beta2, rc1, rc2, eps, decay tensai.Float) error {
	if t.freed || grad.freed || m.freed || v.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	count := t.Size()
	if grad.Size() != count || m.Size() != count || v.Size() != count {
		return fmt.Errorf("tensai: adam operands must all hold %d elements", count)
	}
	p := trainParams{
		count: uint32(count), aCount: uint32(count), bCount: uint32(count),
		lr: float32(lr), beta1: float32(beta1), beta2: float32(beta2),
		rc1: float32(rc1), rc2: float32(rc2), eps: float32(eps), decay: float32(decay),
	}

	t.g.mu.Lock()
	defer t.g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if t.g.closed {
		return errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""
	g := t.g
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, trainParamBytes, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&p), trainParamBytes)

	entries := [5]wgpuBindGroupEntry{
		{binding: 46, buffer: bufParams, size: trainParamBytes},
		bind(47, grad),
		bind(49, t),
		bind(50, m),
		bind(51, v),
	}
	bindGroup := g.cachedBindGroup(g.pipes.layAdamStep, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((count + 255) / 256)
	return g.dispatch(g.pipes.adamStep, bindGroup, x, y, 1)
}

func (g *Device) MatMul(a, b *tensai.Tensor) (*tensai.Tensor, error) {
	return g.roundTrip(a, b, (*Tensor).MatMul)
}

// MatMulTN and MatMulNT are the transposed products a training step needs,
// with the same upload-compute-download round trip. Together with MatMul
// they satisfy tensai.Accelerator, so passing a Device to
// tensai.UseAccelerator moves every large product -- including both halves
// of a backward pass -- onto the GPU.
func (g *Device) MatMulTN(a, b *tensai.Tensor) (*tensai.Tensor, error) {
	return g.roundTrip(a, b, (*Tensor).MatMulTN)
}

func (g *Device) MatMulNT(a, b *tensai.Tensor) (*tensai.Tensor, error) {
	return g.roundTrip(a, b, (*Tensor).MatMulT)
}

// roundTrip uploads both operands, runs one product, and downloads the
// result. Callers that chain several products should keep their tensors
// resident with Upload instead.
func (g *Device) roundTrip(a, b *tensai.Tensor, op func(a, b *Tensor) (*Tensor, error)) (*tensai.Tensor, error) {
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
	gc, err := op(ga, gb)
	if err != nil {
		return nil, err
	}
	defer gc.Free()
	return gc.Download()
}

// QMatrix is an int8-quantized weight matrix resident in Device memory:
// the packed weights of a quant.QMatrix, four per u32 in plain row-major order,
// plus the per-column scales. Upload one with UploadQ8 and multiply
// activations into it with MatMul; the float32 weights never reach the
// device.
type QMatrix struct {
	g           *Device
	buf, scales uintptr
	rows, cols  int
	words       int // u32 words per weight row: ceil(cols/4)
	freed       bool
}

// UploadQ8 packs a quantized matrix into Device memory. The quant.QMatrix's
// interleaved row-quad layout (an AVX2 artifact) flattens back to
// row-major on the way in.
func (g *Device) UploadQ8(q *quant.QMatrix) (*QMatrix, error) {
	if q.Rows == 0 || q.Cols == 0 {
		return nil, errors.New("tensai: cannot upload an empty matrix")
	}
	words := (q.Cols + 3) / 4
	// One row per stretch of packed words, so the walk splits across
	// workers the way the int4 one does.
	packed := make([]uint32, q.Rows*words)
	workpool.Run(q.Rows, 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			for j := 0; j < q.Cols; j++ {
				b := uint32(uint8(q.Q[q.Index(i, j)]))
				packed[i*words+j/4] |= b << (8 * (j % 4))
			}
		}
	})
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
	return &QMatrix{g: g, buf: buf, scales: sbuf, rows: q.Rows, cols: q.Cols, words: words}, nil
}

// Shape returns (rows, cols).
func (q *QMatrix) Shape() (int, int) { return q.rows, q.cols }

// Free releases the Device buffers. Calling Free again is a no-op.
func (q *QMatrix) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if q.freed {
		return
	}
	q.freed = true
	q.g.dropBuffer(q.buf)
	q.g.dropBuffer(q.scales)
}

// MatMul computes x @ Q on the Device: x's last axis must equal the weight
// rows, and every leading axis is a batch of activation rows. Weights are
// dequantized in registers, so only a quarter of the f32 weight bytes
// cross the memory bus.
func (q *QMatrix) MatMul(x *Tensor) (*Tensor, error) {
	return q.MatMulOpts(x, nil, nil)
}

// MatMulOpts is MatMul with a fused epilogue: a per-column bias added
// when bias is non-nil, and the product accumulated into dst — which is
// then the returned tensor — when dst is non-nil. One dispatch instead
// of a matmul plus one or two element-wise passes.
func (q *QMatrix) MatMulOpts(x, bias, dst *Tensor) (*Tensor, error) {
	return q.matmulF32(x, bias, dst, nil, 0, nil, 0, 0, 0)
}

// MatMulRMSNorm is MatMulOpts with out = rmsnorm(x, norm, eps) @ Q: the
// single-row matvec runs the normalization as a kernel prologue — one
// dependent dispatch fewer per decode norm — while a batch, whose
// dispatch count is amortized anyway, norms in its own dispatch first.
func (q *QMatrix) MatMulRMSNorm(x, norm *Tensor, eps float64, bias, dst *Tensor) (*Tensor, error) {
	// 2048 is the QXS shared-memory stage in the kernel.
	if x.Size() == q.rows && q.rows <= 2048 {
		return q.matmulF32(x, bias, dst, norm, float32(eps), nil, 0, 0, 0)
	}
	nx, err := x.RMSNorm(norm, eps)
	if err != nil {
		return nil, err
	}
	defer nx.Free()
	return q.matmulF32(nx, bias, dst, nil, 0, nil, 0, 0, 0)
}

// MatMulAttnCombine is MatMulOpts for the output projection fed straight
// by attn_split_gh's scratch: the softmax combine that attn_reduce_g runs
// as a dependent dispatch of its own folds into the matvec prologue,
// staging the combined attention row in shared memory in the reduce
// kernel's exact order. Single-row decode only; the row must fit the
// kernel's shared stage.
func (q *QMatrix) MatMulAttnCombine(scr *Tensor, slabs, dh, group int, bias, dst *Tensor) (*Tensor, error) {
	if q.rows > 2048 { // QXS, the kernel's shared-memory stage
		return nil, fmt.Errorf("tensai: gpu attn combine over %d rows exceeds the staged 2048", q.rows)
	}
	if dh <= 0 || group <= 0 || group > 8 || slabs <= 0 || q.rows%(dh*group) != 0 {
		return nil, fmt.Errorf("tensai: gpu attn combine geometry dh=%d group=%d slabs=%d for %d rows", dh, group, slabs, q.rows)
	}
	return q.matmulF32(scr, bias, dst, nil, 0, scr, slabs, dh, group)
}

func (q *QMatrix) matmulF32(x, bias, dst, norm *Tensor, eps float32, scr *Tensor, slabs, dh, grp int) (*Tensor, error) {
	if q.freed || x.freed || (bias != nil && bias.freed) || (dst != nil && dst.freed) || (norm != nil && norm.freed) {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if norm != nil && norm.Size() != q.rows {
		return nil, fmt.Errorf("tensai: gpu qmatmul norm length %d != %d rows", norm.Size(), q.rows)
	}
	if q.g != x.g {
		return nil, errors.New("tensai: gpu tensors belong to different GPUs")
	}
	m := 1
	if scr == nil {
		// The combine prologue stages the activation row from scr, so x
		// (= scr) skips the row-shape check and the batch stays one row.
		n := len(x.shape)
		if n == 0 || x.shape[n-1] != q.rows {
			return nil, fmt.Errorf("tensai: gpu qmatmul shape mismatch: %v @ %dx%d", x.shape, q.rows, q.cols)
		}
		m = x.Size() / q.rows
	}
	if m > 65535 {
		return nil, fmt.Errorf("tensai: gpu qmatmul batch of %d rows exceeds 65535", m)
	}
	if bias != nil && bias.Size() != q.cols {
		return nil, fmt.Errorf("tensai: gpu qmatmul bias length %d != %d columns", bias.Size(), q.cols)
	}
	if dst != nil && dst.Size() != m*q.cols {
		return nil, fmt.Errorf("tensai: gpu qmatmul dst of %d elements, want %d", dst.Size(), m*q.cols)
	}
	outShape := []int{1, q.cols}
	if scr == nil {
		n := len(x.shape)
		outShape = append(append([]int(nil), x.shape[:n-1]...), q.cols)
	}
	if m >= 32 && q.g.hasIntDot {
		return q.matmulIntDot(x, bias, dst, m, outShape)
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

	outBytes := uint64(m*q.cols) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	var flags uint32
	biasBuf, biasSize := q.scales, uint64(q.words*4)*4 // dummy; flags gate the read
	if bias != nil {
		flags |= 1
		biasBuf, biasSize = bias.buf, uint64(bias.Size())*4
	}
	bufOut := uintptr(0)
	if dst != nil {
		flags |= 2
		bufOut = dst.buf
	} else {
		bufOut = g.takeOutBuffer(outBytes)
	}
	normBuf, normSize := q.scales, uint64(q.words*4)*4 // dummy; flags gate the read
	if norm != nil {
		flags |= 4
		normBuf, normSize = norm.buf, uint64(norm.Size())*4
	}
	scrBuf, scrSize := q.scales, uint64(q.words*4)*4 // dummy; flags gate the read
	params := [8]uint32{uint32(q.rows), uint32(q.cols), uint32(q.words), uint32(m), flags, math.Float32bits(eps)}
	if scr != nil {
		flags |= 8
		params[4] = flags
		params[6] = uint32(slabs)
		params[7] = uint32(dh) | uint32(grp)<<16
		scrBuf, scrSize = scr.buf, uint64(scr.Size())*4
	}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 32)
	defer release()

	entries := [8]wgpuBindGroupEntry{
		{binding: 15, buffer: bufParams, offset: offParams, size: 32},
		{binding: 16, buffer: q.buf, size: uint64(q.rows*q.words) * 4},
		{binding: 17, buffer: q.scales, size: uint64(q.words*4) * 4},
		bind(18, x),
		{binding: 19, buffer: bufOut, size: outBytes},
		{binding: 34, buffer: biasBuf, size: biasSize},
		{binding: 39, buffer: normBuf, size: normSize},
		{binding: 45, buffer: scrBuf, size: scrSize},
	}
	// A large batch takes the tiled GEMM, a small one the row-blocked
	// kernel; a single row keeps the matvec shape. Only the matvec's
	// layout knows the norm binding, so the batched kernels drop it.
	pipe, lay, bound := g.pipes.qmatmul, g.pipes.layQmatmul, entries[:]
	gy := uint32(m)
	if m >= 32 {
		pipe, lay, gy = g.pipes.qmatmulT, g.pipes.layQmatmulT, uint32((m+63)/64)
		bound = entries[:6]
	} else if m > 1 {
		pipe, lay, gy = g.pipes.qmatmulB, g.pipes.layQmatmulB, uint32((m+7)/8)
		bound = entries[:6]
	}
	bindGroup := g.cachedBindGroup(lay, bound)
	runtime.KeepAlive(&entries)

	err := g.dispatch(pipe, bindGroup,
		uint32((q.words+15)/16), gy, 1)
	if err != nil {
		if dst == nil {
			g.dropBuffer(bufOut)
		}
		return nil, err
	}
	if dst != nil {
		return dst, nil
	}
	return &Tensor{g: g, buf: bufOut, shape: outShape}, nil
}

// matmulIntDot is MatMul's integer path: qacts_pack quantizes the
// activation rows to symmetric int8 once, then qmatmul_i runs the
// GEMM on dot4I8Packed. Activations lose their f32 precision to one
// int8 scale per row — the same shape of rounding the CPU decode path
// applies — so outputs differ from the f32 kernels within quantization
// tolerance.
func (q *QMatrix) matmulIntDot(x, bias, dst *Tensor, m int, outShape []int) (*Tensor, error) {
	g := q.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	kw4 := (q.rows + 3) / 4
	outBytes := uint64(m*q.cols) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	var flags uint32
	biasBuf, biasSize := q.scales, uint64(q.words*4)*4 // dummy; flags gate the read
	if bias != nil {
		flags |= 1
		biasBuf, biasSize = bias.buf, uint64(bias.Size())*4
	}
	params := [8]uint32{uint32(q.rows), uint32(kw4), uint32(q.cols), uint32(q.words), uint32(m), flags}
	paBytes := uint64(m*kw4) * 4
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	bufPA := g.takeOutBuffer(paBytes)
	bufAS := g.takeOutBuffer(uint64(m) * 4)
	bufOut := uintptr(0)
	if dst != nil {
		flags |= 2
		params[5] = flags
		bufOut = dst.buf
	} else {
		bufOut = g.takeOutBuffer(outBytes)
	}
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	defer g.putBuffer(gpuTensorUsage, paBytes, bufPA)
	defer g.putBuffer(gpuTensorUsage, uint64(m)*4, bufAS)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)

	qe := [4]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, size: 32},
		bind(1, x),
		{binding: 2, buffer: bufPA, size: paBytes},
		{binding: 3, buffer: bufAS, size: uint64(m) * 4},
	}
	bgA := g.cachedBindGroup(g.pipes.layQacts, qe[:])
	runtime.KeepAlive(&qe)
	if err := g.dispatch(g.pipes.qacts, bgA, uint32(m), 1, 1); err != nil {
		if dst == nil {
			g.dropBuffer(bufOut)
		}
		return nil, err
	}
	me := [7]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, size: 32},
		{binding: 2, buffer: bufPA, size: paBytes},
		{binding: 3, buffer: bufAS, size: uint64(m) * 4},
		{binding: 4, buffer: q.buf, size: uint64(q.rows*q.words) * 4},
		{binding: 5, buffer: q.scales, size: uint64(q.words*4) * 4},
		{binding: 6, buffer: bufOut, size: outBytes},
		{binding: 7, buffer: biasBuf, size: biasSize},
	}
	bgM := g.cachedBindGroup(g.pipes.layQmatmulI, me[:])
	runtime.KeepAlive(&me)
	if err := g.dispatch(g.pipes.qmatmulI, bgM, uint32((q.cols+63)/64), uint32((m+63)/64), 1); err != nil {
		if dst == nil {
			g.dropBuffer(bufOut)
		}
		return nil, err
	}
	if dst != nil {
		return dst, nil
	}
	return &Tensor{g: g, buf: bufOut, shape: outShape}, nil
}

// RMSNorm normalizes each row of the last axis by its root mean square and
// multiplies by the weight vector w (length = last axis), the pre-norm of
// Llama-family transformer blocks. Returns a new Device-resident tensor.
func (t *Tensor) RMSNorm(w *Tensor, eps float64) (*Tensor, error) {
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
	bufOut := g.takeOutBuffer(uint64(t.Size()) * 4)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [4]wgpuBindGroupEntry{
		{binding: 20, buffer: bufParams, size: 16},
		bind(21, t),
		bind(22, w),
		{binding: 23, buffer: bufOut, size: uint64(t.Size()) * 4},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRmsnorm, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.rmsnorm, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, nil
}

// RMSNormEach is RMSNorm over consecutive groups of len(w) elements
// instead of whole rows of the last axis — Qwen3's per-head QK-norm,
// where every head of a packed projection normalizes against the same
// per-channel weights.
func (t *Tensor) RMSNormEach(w *Tensor, eps float64) (*Tensor, error) {
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
	bufOut := g.takeOutBuffer(uint64(t.Size()) * 4)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)

	entries := [4]wgpuBindGroupEntry{
		{binding: 20, buffer: bufParams, size: 16},
		bind(21, t),
		bindN(22, w, uint64(n)*4),
		{binding: 23, buffer: bufOut, size: uint64(t.Size()) * 4},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRmsnorm, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(g.pipes.rmsnorm, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, nil
}

// RoPE applies rotary position embeddings in place, half-split style: the
// last axis divides into heads of headSz, and element pair (c, c+headSz/2)
// of each head in row r rotates by (pos0+r) * theta^(-2c/headSz).
func (t *Tensor) RoPE(headSz, pos0 int, theta float64) error {
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
		bind(25, t),
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRope, entries[:])
	runtime.KeepAlive(&entries)

	pairs := (d / headSz) * (headSz / 2)
	return g.dispatch(g.pipes.rope, bindGroup, uint32((pairs+63)/64), uint32(rows), 1)
}

// eltwiseIP dispatches one of the two in-place elementwise kernels with o
// as the second operand, repeating o when it is shorter (a row bias
// against a batch of rows).
func (t *Tensor) eltwiseIP(pipe, lay uintptr, o *Tensor) error {
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
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 16)
	defer release()

	entries := [3]wgpuBindGroupEntry{
		{binding: 26, buffer: bufParams, offset: offParams, size: 16},
		bind(27, t),
		bind(28, o),
	}
	bindGroup := g.cachedBindGroup(lay, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((t.Size() + gpuKernelWG - 1) / gpuKernelWG)
	return g.dispatch(pipe, bindGroup, x, y, 1)
}

// Add computes t += o elementwise in place. o may be shorter as long as
// its size divides t's: it repeats, which applies a row bias to every row.
func (t *Tensor) Add(o *Tensor) error {
	return t.eltwiseIP(t.g.pipes.addIP, t.g.pipes.layAddIP, o)
}

// SiluMul computes t = silu(t) * o elementwise in place — the SwiGLU
// joint, with t the gate projection and o the up projection.
func (t *Tensor) SiluMul(o *Tensor) error {
	return t.eltwiseIP(t.g.pipes.siluMulIP, t.g.pipes.laySiluMulIP, o)
}

// GLUSplit joins a fused gate|up projection: t holds rows of 2*inter
// with gate in the first half and up in the second, and the result is
// act(gate) * up per row — silu, or gelu_tanh when gelu is set. One
// dispatch in place of the slice-out-and-multiply chain.
func (t *Tensor) GLUSplit(inter int, gelu bool) (*Tensor, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if inter <= 0 || t.Size()%(2*inter) != 0 {
		return nil, fmt.Errorf("tensai: glu split of %d elements into pairs of %d", t.Size(), inter)
	}
	rows := t.Size() / (2 * inter)
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(rows*inter) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufOut := g.takeOutBuffer(outBytes)
	var gu uint32
	if gelu {
		gu = 1
	}
	params := [4]uint32{uint32(rows), uint32(inter), gu, 0}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 16)
	defer release()

	entries := [3]wgpuBindGroupEntry{
		{binding: 44, buffer: bufParams, offset: offParams, size: 16},
		bind(37, t),
		{binding: 38, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layGluSplit, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D((rows*inter + gpuKernelWG - 1) / gpuKernelWG)
	if err := g.dispatch(g.pipes.gluSplit, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: []int{rows, inter}}, nil
}

// GeluMul computes t = gelu_tanh(t) * o elementwise in place — Gemma's
// gate.
func (t *Tensor) GeluMul(o *Tensor) error {
	return t.eltwiseIP(t.g.pipes.geluMulIP, t.g.pipes.layGeluMulIP, o)
}

// CopyRowsInto copies t's whole buffer into dst starting at element offset
// off — appending freshly projected k/v rows to a preallocated cache
// without leaving the device.
// IsF16 reports whether the tensor stores half-precision elements.
func (t *Tensor) IsF16() bool { return t.f16 }

// RopeCacheF16 rotates the leading qw+kvDim elements of a fused single
// qkv row in place — RoPE at position pos over headSz-wide heads — and
// appends the rotated k and the trailing v into the half-precision kc
// and vc caches at element offset dstOff, all in one dispatch. Decode
// used to spend three dependent dispatches on this.
func (t *Tensor) RopeCacheF16(kc, vc *Tensor, headSz, qw, kvDim, pos int, theta float64, dstOff int) error {
	if t.freed || kc.freed || vc.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	if t.g != kc.g || t.g != vc.g {
		return errors.New("tensai: gpu tensors belong to different GPUs")
	}
	g := t.g
	if g.pipes.ropeCache == 0 {
		return errors.New("tensai: device has no rope_cache kernel")
	}
	if !kc.f16 || !vc.f16 {
		return errors.New("tensai: rope_cache needs f16 caches")
	}
	d := qw + kvDim
	if headSz <= 0 || d%headSz != 0 || qw%headSz != 0 || t.Size() < d+kvDim {
		return fmt.Errorf("tensai: rope_cache shape mismatch: qw=%d kvDim=%d headSz=%d in %d", qw, kvDim, headSz, t.Size())
	}
	if dstOff < 0 || dstOff+kvDim > kc.Size() || dstOff+kvDim > vc.Size() {
		return fmt.Errorf("tensai: rope_cache append of %d at %d overflows caches", kvDim, dstOff)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	pairs := (d / headSz) * (headSz / 2)
	total := pairs + kvDim
	params := [8]uint32{
		uint32(d), uint32(headSz), uint32(qw), uint32(kvDim),
		uint32(pos), uint32(dstOff), math.Float32bits(float32(theta)),
	}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 32)
	defer release()
	entries := [4]wgpuBindGroupEntry{
		{binding: 43, buffer: bufParams, offset: offParams, size: 32},
		bind(41, t),
		bind(22, kc),
		bind(42, vc),
	}
	bindGroup := g.cachedBindGroup(g.pipes.layRopeCache, entries[:])
	runtime.KeepAlive(&entries)
	if err := g.dispatch(g.pipes.ropeCache, bindGroup, uint32((total+63)/64), 1, 1); err != nil {
		return err
	}
	if uncapturedCB != "" {
		return fmt.Errorf("tensai: gpu rope_cache failed: %s", uncapturedCB)
	}
	return nil
}

func (t *Tensor) CopyRowsInto(dst *Tensor, off int) error {
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
	if dst.f16 {
		// A half-precision destination converts through a kernel; the
		// pass batches it like any other dispatch, no copy involved.
		n := t.Size()
		params := [4]uint32{uint32(n), uint32(off)}
		bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 16)
		defer release()
		entries := [3]wgpuBindGroupEntry{
			{binding: 20, buffer: bufParams, offset: offParams, size: 16},
			bindN(21, t, uint64(n)*4),
			bind(22, dst),
		}
		bindGroup := g.cachedBindGroup(g.pipes.layRowsToF16, entries[:])
		runtime.KeepAlive(&entries)
		if err := g.dispatch(g.pipes.rowsToF16, bindGroup, uint32((n+255)/256), 1, 1); err != nil {
			return err
		}
		if uncapturedCB != "" {
			return fmt.Errorf("tensai: gpu f16 copy failed: %s", uncapturedCB)
		}
		return nil
	}
	if g.batchEnc != 0 {
		g.endBatchPass()
		fnEncoderCopyBuffer(g.batchEnc, t.buf, t.off, dst.buf, dst.off+uint64(off)*4, uint64(t.Size())*4)
		return nil
	}
	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	fnEncoderCopyBuffer(encoder, t.buf, t.off, dst.buf, dst.off+uint64(off)*4, uint64(t.Size())*4)
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
func (q *Tensor) GroupedCausalAttention(k, v *Tensor, heads, kvHeads, seqKV, window int) (*Tensor, error) {
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
	if dh > 512 {
		return nil, fmt.Errorf("tensai: grouped attention head dimension %d exceeds 512", dh)
	}
	seq := q.Size() / d
	kvDim := kvHeads * dh
	if seqKV < seq {
		return nil, fmt.Errorf("tensai: grouped attention needs seqKV >= seq, got %d < %d", seqKV, seq)
	}
	if k.Size() < seqKV*kvDim || v.Size() < seqKV*kvDim {
		return nil, fmt.Errorf("tensai: kv cache of %d/%d elements is smaller than %d x %d", k.Size(), v.Size(), seqKV, kvDim)
	}
	// Groups of up to eight query heads sharing a KV head run in one
	// workgroup (attn_causal_g), streaming each K and V row once per
	// group; other shapes keep the per-head kernel.
	group := heads / kvHeads
	grouped := group > 1 && group <= 8 && dh <= 256
	if k.f16 != v.f16 {
		return nil, errors.New("tensai: k and v caches must share a precision")
	}
	if k.f16 && !grouped {
		return nil, fmt.Errorf("tensai: the f16 cache path needs a query-head group of 2..8, got %d", group)
	}
	nwg := heads
	if grouped {
		nwg = kvHeads
	}
	offs := make([]uint32, 4*nwg)
	for h := 0; h < nwg; h++ {
		if grouped {
			offs[4*h] = uint32(h * group * dh)
			offs[4*h+1] = uint32(h * dh)
			offs[4*h+2] = uint32(h * group * dh)
		} else {
			offs[4*h] = uint32(h * dh)
			offs[4*h+1] = uint32((h / group) * dh)
			offs[4*h+2] = uint32(h * dh)
		}
	}
	// A one-row grouped decode leaves kvHeads workgroups walking the
	// whole context serially — two workgroups on a twelve-WGP device —
	// so a long context splits into KV slabs folded by a second, tiny
	// dispatch. Short contexts stay serial: the fold costs a dispatch.
	if grouped && k.f16 && seq == 1 && window == 0 && seqKV >= 256 && q.g.pipes.attnSplit != 0 {
		return q.groupedDecodeSplit(k, v, kvHeads, dh, d, kvDim, seqKV, offs)
	}
	rows := nwg * seq
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
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 32)
	defer release()
	bufOffs := g.offsBuffer(offs)
	if bufOffs == 0 {
		return nil, errors.New("tensai: gpu offset buffer allocation failed")
	}
	bufOut := g.takeOutBuffer(outBytes)

	entries := [6]wgpuBindGroupEntry{
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 10, buffer: bufParams, offset: offParams, size: 32},
		bind(11, q),
		bind(12, k),
		bind(13, v),
		{binding: 14, buffer: bufOut, size: outBytes},
	}
	pipe, lay := g.pipes.attn, g.pipes.layAttn
	if grouped {
		pipe, lay = g.pipes.attnG, g.pipes.layAttnG
		if k.f16 {
			pipe, lay = g.pipes.attnF16, g.pipes.layAttnF16
		}
	}
	bindGroup := g.cachedBindGroup(lay, entries[:])
	runtime.KeepAlive(&entries)

	x, y := split2D(rows)
	if err := g.dispatch(pipe, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), q.shape...)}, nil
}

// GroupedCausalAttentionParts runs only the split half of the decode
// attention, handing back the per-slab softmax scratch and the slab
// count so the caller can fold the combine into the projection that
// follows (QMatrix.MatMulAttnCombine). A shape that would not take the
// split path returns a nil tensor and no error; the caller then runs
// GroupedCausalAttention as usual.
func (q *Tensor) GroupedCausalAttentionParts(k, v *Tensor, heads, kvHeads, seqKV, window int) (*Tensor, int, error) {
	if q.freed || k.freed || v.freed {
		return nil, 0, errors.New("tensai: gpu tensor already freed")
	}
	nd := len(q.shape)
	if nd == 0 || heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, 0, nil
	}
	d := q.shape[nd-1]
	if d%heads != 0 {
		return nil, 0, nil
	}
	dh := d / heads
	group := heads / kvHeads
	grouped := group > 1 && group <= 8 && dh <= 256
	if !grouped || !k.f16 || q.Size() != d || window != 0 || seqKV < 256 || q.g.pipes.attnSplit == 0 {
		return nil, 0, nil
	}
	kvDim := kvHeads * dh
	if k.Size() < seqKV*kvDim || v.Size() < seqKV*kvDim {
		return nil, 0, fmt.Errorf("tensai: kv cache of %d/%d elements is smaller than %d x %d", k.Size(), v.Size(), seqKV, kvDim)
	}
	offs := make([]uint32, 4*kvHeads)
	for h := 0; h < kvHeads; h++ {
		offs[4*h] = uint32(h * group * dh)
		offs[4*h+1] = uint32(h * dh)
		offs[4*h+2] = uint32(h * group * dh)
	}
	return q.attnSplitPass(k, v, kvHeads, dh, d, kvDim, seqKV, offs)
}

// attnSlabs sizes the decode split: tiles per slab and the slab count
// for a context of seqKV positions.
func attnSlabs(seqKV int) (per, slabs int) {
	tiles := (seqKV + 63) / 64
	per = (tiles + 31) / 32 // tiles per slab: at most 32 slabs
	slabs = (tiles + per - 1) / per
	return per, slabs
}

// attnSplitPass runs attn_split_gh over (slabs x kvHeads) workgroups and
// returns the scratch of per-slab softmax states for a reduce — the
// standalone kernel's or one fused into the projection that follows.
func (q *Tensor) attnSplitPass(k, v *Tensor, kvHeads, dh, d, kvDim, seqKV int, offs []uint32) (*Tensor, int, error) {
	g := q.g
	per, slabs := attnSlabs(seqKV)
	stride := 8*dh + 16
	scrBytes := uint64(kvHeads*slabs*stride) * 4

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, 0, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	// seqQ carries the tiles per slab and rows the slab count; the split
	// kernels read nothing else differently.
	params := [8]uint32{
		uint32(per), uint32(seqKV), uint32(dh), uint32(d),
		uint32(slabs), uint32(seqKV - 1), uint32(kvDim), 0,
	}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 32)
	defer release()
	bufOffs := g.offsBuffer(offs)
	if bufOffs == 0 {
		return nil, 0, errors.New("tensai: gpu offset buffer allocation failed")
	}
	scr := g.takeOutBuffer(scrBytes)

	split := [6]wgpuBindGroupEntry{
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 10, buffer: bufParams, offset: offParams, size: 32},
		bind(11, q),
		bind(12, k),
		bind(13, v),
		{binding: 40, buffer: scr, size: scrBytes},
	}
	bg := g.cachedBindGroup(g.pipes.layAttnSplit, split[:])
	runtime.KeepAlive(&split)
	if err := g.dispatch(g.pipes.attnSplit, bg, uint32(slabs), uint32(kvHeads), 1); err != nil {
		g.dropBuffer(scr)
		return nil, 0, err
	}
	return &Tensor{g: g, buf: scr, shape: []int{kvHeads * slabs, stride}}, slabs, nil
}

// groupedDecodeSplit runs the split decode attention: attn_split_gh over
// (slabs x kvHeads) workgroups into a scratch of per-slab softmax states,
// then attn_reduce_g folds them into the output row. Both dispatches ride
// whatever batch is open, like the single-kernel path.
func (q *Tensor) groupedDecodeSplit(k, v *Tensor, kvHeads, dh, d, kvDim, seqKV int, offs []uint32) (*Tensor, error) {
	scr, slabs, err := q.attnSplitPass(k, v, kvHeads, dh, d, kvDim, seqKV, offs)
	if err != nil {
		return nil, err
	}
	defer scr.Free()
	g := q.g
	per, _ := attnSlabs(seqKV)

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(q.Size()) * 4
	params := [8]uint32{
		uint32(per), uint32(seqKV), uint32(dh), uint32(d),
		uint32(slabs), uint32(seqKV - 1), uint32(kvDim), 0,
	}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&params[0]), 32)
	defer release()
	bufOffs := g.offsBuffer(offs)
	if bufOffs == 0 {
		return nil, errors.New("tensai: gpu offset buffer allocation failed")
	}
	bufOut := g.takeOutBuffer(outBytes)
	reduce := [4]wgpuBindGroupEntry{
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 10, buffer: bufParams, offset: offParams, size: 32},
		{binding: 14, buffer: bufOut, size: outBytes},
		bind(40, scr),
	}
	bg := g.cachedBindGroup(g.pipes.layAttnReduce, reduce[:])
	runtime.KeepAlive(&reduce)
	if err := g.dispatch(g.pipes.attnReduce, bg, uint32(kvHeads), 1, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), q.shape...)}, nil
}

// Q4Matrix is an int4-quantized weight matrix resident in Device memory:
// the nibbles of a quant.Q4Matrix packed four row-pair bytes per u32 with the
// per-(group, column) scales alongside. The kernel subtracts the nibble
// offset in registers and folds group scales at group boundaries, so a
// matvec streams an eighth of the f32 weight bytes.
type Q4Matrix struct {
	g           *Device
	buf, scales uintptr
	rows, cols  int
	words       int // u32 words per pair row: ceil(cols/4)
	freed       bool
}

// UploadQ4 packs a 4-bit quantized matrix into Device memory. The kernel
// folds scales on 64-row groups, so other group lengths are rejected.
func (g *Device) UploadQ4(q *quant.Q4Matrix) (*Q4Matrix, error) {
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
	// A row pair owns its stretch of the packed words, so the walk splits
	// across workers cleanly. It is a billion iterations on a small model
	// and was most of what -gpu spent getting started.
	packed := make([]uint32, pairs*words)
	workpool.Run(pairs, 1, func(lo, hi int) {
		for i2 := lo; i2 < hi; i2++ {
			for j := 0; j < q.Cols; j++ {
				b := nib(2*i2, j) | nib(2*i2+1, j)<<4
				packed[i2*words+j/4] |= b << (8 * (j % 4))
			}
		}
	})
	groups := (q.Rows + 63) / 64 // q4Group
	scales := make([]float32, groups*words*4)
	workpool.Run(groups, 1, func(lo, hi int) {
		for gi := lo; gi < hi; gi++ {
			for j := 0; j < q.Cols; j++ {
				scales[gi*words*4+j] = q.Scale[q.TableIndex(gi, j)]
			}
		}
	})

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
	return &Q4Matrix{g: g, buf: buf, scales: sbuf, rows: q.Rows, cols: q.Cols, words: words}, nil
}

// Shape returns (rows, cols).
func (q *Q4Matrix) Shape() (int, int) { return q.rows, q.cols }

// Free releases the Device buffers. Calling Free again is a no-op.
func (q *Q4Matrix) Free() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if q.freed {
		return
	}
	q.freed = true
	q.g.dropBuffer(q.buf)
	q.g.dropBuffer(q.scales)
}

// MatMul computes x @ Q on the Device: x's last axis must equal the weight
// rows, and every leading axis is a batch of activation rows.
func (q *Q4Matrix) MatMul(x *Tensor) (*Tensor, error) {
	return q.MatMulOpts(x, nil, nil)
}

// MatMulOpts is MatMul with a fused epilogue: a per-column bias added
// when bias is non-nil, and the product accumulated into dst — which is
// then the returned tensor — when dst is non-nil.
// MatMulRMSNorm matches QMatrix.MatMulRMSNorm; the 4-bit kernel has no
// fused prologue, so the normalization always runs as its own dispatch.
func (q *Q4Matrix) MatMulRMSNorm(x, norm *Tensor, eps float64, bias, dst *Tensor) (*Tensor, error) {
	nx, err := x.RMSNorm(norm, eps)
	if err != nil {
		return nil, err
	}
	defer nx.Free()
	return q.MatMulOpts(nx, bias, dst)
}

func (q *Q4Matrix) MatMulOpts(x, bias, dst *Tensor) (*Tensor, error) {
	if q.freed || x.freed || (bias != nil && bias.freed) || (dst != nil && dst.freed) {
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
	if bias != nil && bias.Size() != q.cols {
		return nil, fmt.Errorf("tensai: gpu q4matmul bias length %d != %d columns", bias.Size(), q.cols)
	}
	if dst != nil && dst.Size() != m*q.cols {
		return nil, fmt.Errorf("tensai: gpu q4matmul dst of %d elements, want %d", dst.Size(), m*q.cols)
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
	var flags uint32
	biasBuf, biasSize := q.scales, uint64(groups*q.words*4)*4 // dummy; flags gate the read
	if bias != nil {
		flags |= 1
		biasBuf, biasSize = bias.buf, uint64(bias.Size())*4
	}
	bufOut := uintptr(0)
	if dst != nil {
		flags |= 2
		bufOut = dst.buf
	} else {
		bufOut = g.takeOutBuffer(outBytes)
	}
	params := [8]uint32{
		uint32(q.rows), uint32(q.cols), uint32(q.words), uint32(m),
		uint32(groups), flags, 0, 0,
	}
	bufParams := g.takeBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32)
	defer g.putBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 32, bufParams)
	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 32)

	pairs := (q.rows + 1) / 2
	entries := [6]wgpuBindGroupEntry{
		{binding: 29, buffer: bufParams, size: 32},
		{binding: 30, buffer: q.buf, size: uint64(pairs*q.words) * 4},
		{binding: 31, buffer: q.scales, size: uint64(groups*q.words*4) * 4},
		bind(32, x),
		{binding: 33, buffer: bufOut, size: outBytes},
		{binding: 35, buffer: biasBuf, size: biasSize},
	}
	bindGroup := g.cachedBindGroup(g.pipes.layQ4matmul, entries[:])
	runtime.KeepAlive(&entries)

	err := g.dispatch(g.pipes.q4matmul, bindGroup,
		uint32((q.words+15)/16), uint32(m), 1)
	if err != nil {
		if dst == nil {
			g.dropBuffer(bufOut)
		}
		return nil, err
	}
	if dst != nil {
		return dst, nil
	}
	return &Tensor{g: g, buf: bufOut, shape: outShape}, nil
}

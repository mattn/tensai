# Platforms

Nothing here is required: tensai builds and runs everywhere Go does, with the
portable kernels and no GPU. The table says what each platform adds on top,
and what it is still missing.

## By operating system

The CPU kernels do not depend on the operating system at all: what decides
them is the architecture and the Go version, in the next table. What the OS
decides is the GPU backend and how the weight cache is read.

| OS | GPU | Weight cache | Freeing mapped pages |
|---|---|---|---|
| Linux | Vulkan, `dlopen` | Memory-mapped | `madvise(MADV_DONTNEED)` |
| macOS | Metal, `dlopen` | Memory-mapped | `madvise`, issued as a raw syscall |
| Windows | D3D12 or Vulkan, `LoadLibrary` | Memory-mapped, through a file mapping | Not available, pages stay until unmap |
| Other (BSD, illumos, ...) | Not built | Read through `ReadAt` instead | Not available |

Released binaries cover linux, macOS and Windows on both amd64 and arm64.
Anywhere else, `go build` still produces a working tensai: it loses the
mapping (a load reads the cache file instead, which costs time and memory
but changes nothing else) and the GPU tags.

## CPU kernels

The vector kernels are Go, written against the experimental `simd/archsimd`
package, and they turn on with `GOEXPERIMENT=simd` at build time. A build
without it, or on a combination not listed below, uses the portable bodies
and computes the same results, an order of magnitude slower.

| Arch | Go | Kernels | Notes |
|---|---|---|---|
| amd64 | 1.26, 1.27 | AVX2 | Checked at runtime; a CPU without AVX2 and FMA falls back |
| arm64 | 1.27 | NEON | Go 1.26's `simd/archsimd` has no arm64 half, so 1.26 falls back |
| others | any | portable | |

What the NEON build vectorizes today is narrower than the AVX2 one:

| Kernel | amd64 | arm64 |
|---|---|---|
| int8 matvec (`-q8`, requantized weights) | AVX2 | NEON |
| Attention dot products, value accumulation, softmax `exp` | AVX2 | NEON |
| Element-wise rows (add, scale, SwiGLU gate) | AVX2 | NEON |
| 4-bit matvec (`-q4`) | AVX2 | portable |
| Grouped int8 matvec (gguf blocks repacked) | AVX2 | portable |
| Batched prefill folds | AVX2 | portable |
| MXFP4 (gpt-oss) | AVX2 | portable |
| Activation quantizer | AVX2 | portable |
| Dense float matmul (training) | AVX2 | portable |

`tensai bench` prints which family it ran, so a build that quietly fell back
is visible in the first lines of its output.

## GPU

The GPU backend is a build tag away and loads
[wgpu-native](https://github.com/gfx-rs/wgpu-native) at runtime, so there is
no cgo and no GPU SDK to install. See [GPU (WebGPU)](gpu.md) for the library
versions and the environment variable that points at them.

| OS | Loader | Backend wgpu-native picks |
|---|---|---|
| Linux | `dlopen` | Vulkan |
| macOS | `dlopen` | Metal |
| Windows | `LoadLibrary` | D3D12, or Vulkan |
| others | none | Builds, reports the GPU as unavailable |

Both `amd64` and `arm64` are cross-built for all three, and the two tags pick
the binding generation: `-tags wgpu24` for the v29 C API, `-tags wgpu` for
v22.1.0.5. Inside WSL2 the v29 one is the one that reaches the real GPU
through dozen; a plain `wgpu` build lands on a software rasterizer there.

One model-side limit rather than a platform one: a Gemma 4 checkpoint whose
layers take their values from their keys runs its decode on the CPU, on every
platform, because the device kernels have no copy for that shape yet. The
runtime says so and continues rather than failing.

## What none of this changes

A model is fetched, repacked and cached the same way everywhere, and the
numbers a run produces are the same: the vector kernels are written to match
the portable ones, and a quantized matvec gives bit-identical results under
AVX2, under NEON, and under neither. A platform can make tensai slower. It
does not make it answer differently.

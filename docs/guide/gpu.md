# GPU (WebGPU)

Building with `-tags wgpu` (linux, macOS, Windows) enables an experimental GPU backend with the same shape and broadcasting semantics as the CPU `MatMul`. There is no cgo involved: the bindings load the [wgpu-native](https://github.com/gfx-rs/wgpu-native) shared library at runtime via `ebitengine/purego` (`dlopen` on linux/macOS, `LoadLibrary` on Windows). wgpu-native picks Vulkan on Linux, Vulkan or D3D12 on Windows, and Metal on macOS, so AMD, Intel, Apple, and NVIDIA GPUs all work — as do CPU Vulkan implementations like lavapipe, which is how the tests run on machines without a GPU.

## Setup

Download a **v22.1.0.5** release binary (the C API these bindings target), then either install it where the loader finds it or point `TENSAI_WGPU_LIB` at it:

```bash
curl -sLO https://github.com/gfx-rs/wgpu-native/releases/download/v22.1.0.5/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip -d wgpu
TENSAI_WGPU_LIB=$PWD/wgpu/lib/libwgpu_native.so go test -tags wgpu ./...
```

On Windows, take `wgpu-windows-x86_64-msvc-release.zip` from the same release and point the variable at the `wgpu_native.dll` inside it (any `wgpu_native.dll` on `PATH` or next to the executable is found without the variable):

```powershell
$env:TENSAI_WGPU_LIB="$PWD\wgpu\lib\wgpu_native.dll"
go run -tags wgpu ./_example/wgpu
```

Without the build tag, `gpu.Open` returns an error and nothing else changes.

## Basic usage

```go
dev, err := gpu.Open() // fails cleanly when no GPU / library is present
if err != nil { /* fall back to tensai.MatMul */ }
defer dev.Close()
fmt.Println(dev.Name()) // e.g. "AMD Radeon 780M (integrated)"
out, err := dev.MatMul(a, b)
```

On machines with both an integrated and a discrete GPU, pass a preference: `gpu.Open(gpu.LowPower)` steers to the iGPU, `gpu.HighPerformance` to the dGPU (it is a hint — with a single adapter you always get that one).

## Resident buffers

`dev.MatMul(a, b)` is shorthand for Upload → MatMul → Download → Free. Buffers can instead stay resident on the GPU, so a weight rides the bus once instead of on every call and intermediates never leave the device:

```go
gw, _ := dev.Upload(w)              // weight uploaded once
defer gw.Free()                     // GPU memory is not garbage collected
gx, _ := dev.Upload(x)
h, _ := gx.MatMul(gw)               // chain freely; nothing touches the host
out, _ := h.MatMul(gw2)
result, _ := out.Download()         // one readback at the end
```

Residency matters most on discrete GPUs, where every transfer crosses PCIe; on shared-memory iGPUs the win is smaller and comes mainly from skipping intermediate readbacks.

## Attention on the device

Beyond MatMul, resident tensors support `MatMulT`, an in-place `Scale`, and a row-parallel `Softmax` over the last axis — enough to run single-head attention entirely on the GPU:

```go
out, _ := gq.Attention(gk, gv)                 // softmax(q@k^T/sqrt(d))@v, no host round-trips
out, _ = gq.MultiHeadAttention(gk, gv, heads)  // packed (batch, seq, heads*dh) layout
```

Multi-head attention carves each head out of the packed layout with strided kernels, so no permute is ever materialized. The causal variants (`CausalAttention`, `CausalMultiHeadAttention`) mask future positions inside the kernel, with k and v allowed to hold more positions than q — the prompt-prefill and chunked-decode patterns of autoregressive models — so no mask tensor is ever built either. `CausalMultiHeadAttention` runs as one fused flash-attention-style dispatch (an online softmax over kv tiles, for head dimensions up to 128): the scores matrix never exists, so memory stays at q+k+v+output regardless of sequence length.

## Quantized weights on the device

`UploadQ8` packs a `QMatrix` four int8 weights per u32, and `gpu.QMatrix.MatMul` dequantizes them in registers, so a decode matvec moves a quarter of the f32 bytes. `UploadQ4` does the same for the int4 twin, so `-q4 -gpu` runs models whose int8 weights would not fit.

The rest of a transformer decode step is there as well — `RMSNorm`, in-place `RoPE`, `Add`, `SiluMul`, `GroupedCausalAttention` (a KV cache packing fewer heads than the queries), and `CopyRowsInto` to append fresh k/v rows to a resident cache — so `tensai run -q8 -gpu` runs every block on the device and only the hidden state comes back per token. `BeginBatch`/`Flush` record a whole token's dispatches into one submission, and freed intermediates recycle through a buffer pool.

## Measuring the crossover

`_example/wgpu -sweep` walks a ladder of sizes and marks where the GPU overtakes the CPU kernel. Because the CPU side is the same `dotRows` kernel the rest of the package uses, building the example twice compares portable Go, AVX2, and both GPU usage patterns:

```bash
GOEXPERIMENT=nosimd go build -tags wgpu -o wgpu-nosimd ./_example/wgpu
GOEXPERIMENT=simd   go build -tags wgpu -o wgpu-simd   ./_example/wgpu
./wgpu-nosimd -sweep && ./wgpu-simd -sweep
```

On a Ryzen iGPU (AMD Radeon 780M, native Windows, AVX2 CPU kernel) the register-tiled kernels put every rung of the ladder on the GPU side:

```
             shape                   MFLOP   gpu+xfer   resident        cpu   res/cpu
mnist dense  1x100x784@784x128        20.1     1.51ms      597µs      652µs     1.09x
mnist conv2  1x19600x72@72x16         45.2    1.388ms      763µs    2.354ms     3.09x
tiny         1x128x128@128x128         4.2      432µs      302µs      410µs     1.36x
small        1x512x512@512x512       268.4    1.331ms    1.216ms    6.865ms     5.65x
medium       8x512x512@512x512      2147.5    8.053ms    6.297ms    71.56ms    11.36x
large        32x512x512@512x512     8589.9    86.71ms   28.856ms  266.277ms     9.23x
huge         64x512x512@512x512    17179.9  116.374ms   62.128ms  566.726ms     9.12x
```

The crossover moves with the CPU kernel, GPU driver, and transfer pattern — through a translation layer like dozen inside WSL2 the ratios shrink to roughly parity-to-3x, and on CPU Vulkan implementations the GPU path loses outright; measure on the driver you will ship on.

## `-tags wgpu24`: the new wgpu-native API, and the real GPU inside WSL2

`-tags wgpu24` builds the same `gpu.Open` API against the reworked wgpu-native C API — pair it with a **v29-series** release binary. The new API's payoff is `WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter`, which un-hides non-conformant Vulkan drivers. Concretely: Mesa's dozen (Vulkan-on-D3D12) exposes the real host GPU inside WSL2, but the v22 API hides it as non-conformant and falls back to lavapipe; the wgpu24 build reaches it:

```bash
VK_DRIVER_FILES=/path/to/dzn_icd.json \
TENSAI_WGPU_LIB=$PWD/wgpu29/lib/libwgpu_native.so \
    go run -tags wgpu24 ./_example/wgpu   # adapter: Microsoft Direct3D12 (AMD Radeon(TM) Graphics)
```

When both tags are set, `wgpu24` wins. Note that new does not mean faster: on a Radeon 780M at `32x512x512@512x512` the v22 library runs the same shader in 85ms and the v29 one in 165ms. Use `wgpu24` for the adapters it reaches, not for speed.

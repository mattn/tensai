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

## Training: the accelerator hook

The products above are all inference shapes. Training adds two more per matmul -- the input gradient `grad * w^T` and the weight gradient `x^T * grad` -- which resident tensors now have as `MatMulT` and `MatMulTN`.

A `Device` satisfies `tensai.Accelerator`, so a single call routes every product the CPU would otherwise run, in any package, onto the GPU:

```go
dev, err := gpu.Open(gpu.HighPerformance)
if err == nil {
	defer dev.Close()
	tensai.UseAccelerator(dev) // and tensai.UseAccelerator(nil) to stop
}
```

Only products at or above `tensai.DefaultAcceleratorThreshold` (4e8 multiply-accumulates) go to the device; below it the AVX2 kernels win, and the round trip would only cost time. `tensai.UseAcceleratorThreshold` sets a different bound. If the backend returns an error the product runs on the CPU instead, so acceleration never changes an answer -- only how long it takes to get it.

Nothing else in a training step moves: activations, gradients, and the optimizer stay on the host, so every accelerated product uploads its operands and downloads its result. That caps the win, and it is why the threshold is where it is. On an AMD 780M through `-tags wgpu24`, one autograd step of a two-projection block (`x @ w1 -> GELU -> @ w2 -> MSE`, so six products) measures:

| model width | CPU (AVX2) | with the device | |
|---|---|---|---|
| 512 | 31.1ms | 30.8ms | below the threshold: unchanged |
| 1024 | 173ms | 93ms | 1.85x |
| 2048 | 1382ms | 442ms | 3.13x |

Keeping the whole graph resident instead removes the transfers as well: the same step runs 193ms at width 2048, 7x the CPU. See [Automatic Differentiation](autograd.md) for how a tape does that.

Those are the products. The rest of a backward pass is there too, as resident kernels the inference path never needed on their own:

```go
h, _ := gx.MatMul(gw)               // forward
a, _ := h.Activate(gpu.ActGELU)     // and h.ActivateGrad(gpu.ActGELU, grad) on the way back
s, _ := a.Binary(gpu.OpMul, gscale) // add, sub, mul, div; a shorter operand repeats
db, _ := gdelta.SumCols()           // the gradient of a row broadcast over a batch
gw.AdamStep(ggrad, gm, gv, lr, b1, b2, rc1, rc2, eps, 0)
```

`LayerNorm`, `LayerNormGrad` and `LayerNormXhat` normalize the last axis and take it apart again, `SoftmaxGrad` applies the softmax Jacobian in one pass, and `Permute` reorders up to four axes -- the reshape-and-transpose attention splits its heads with. `Embed` gathers table rows for a list of indices uploaded with `UploadIndices`, and `EmbedGrad` scatters the gradient back: WGSL has no atomic add for f32, so it is a compare-and-swap on the bit pattern, which lets a token that repeats in a batch accumulate correctly. `Activate` and `ActivateGrad` cover ReLU, tanh, sigmoid and GELU, and follow the CPU kernels exactly -- the GELU here is the error function, not the tanh approximation the inference path fuses into its FFN -- so a model can move between device and host mid-training. `AdamStep` matches the `optim` kernel the same way, moments included.

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

`-tags wgpu24` builds the same `gpu.Open` API against the reworked wgpu-native C API — pair it with a **v29-series** release binary. Releases carry both, so a driver that disagrees with one of them
does not also require a Go toolchain: `tensai_*` is the `wgpu24` build and
`tensai-wgpu22_*` the other, and the binary inside is called `tensai` either
way. The new API's payoff is `WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter`, which un-hides non-conformant Vulkan drivers. Concretely: Mesa's dozen (Vulkan-on-D3D12) exposes the real host GPU inside WSL2, but the v22 API hides it as non-conformant and falls back to lavapipe; the wgpu24 build reaches it:

```bash
VK_DRIVER_FILES=/path/to/dzn_icd.json \
TENSAI_WGPU_LIB=$PWD/wgpu29/lib/libwgpu_native.so \
    go run -tags wgpu24 ./_example/wgpu   # adapter: Microsoft Direct3D12 (AMD Radeon(TM) Graphics)
```

When both tags are set, `wgpu24` wins, and on a machine like this one that choice decides everything, because it decides which adapter you get. Measured on the Radeon 780M through dozen inside WSL2, where the v22 build falls back to llvmpipe:

| | `-tags wgpu` (v22) | `-tags wgpu24` (v29) |
|---|---|---|
| adapter reached | llvmpipe (software) | Microsoft Direct3D12 (Radeon 780M) |
| f32 `32x512x512@512x512`, resident inputs | 461.7ms | 69.5ms |
| int8 prefill / decode | 27.0 / 6.0 t/s | 1801 / 27.1 t/s |

This is why the `Makefile` builds with `wgpu24`. The gap is which hardware each build reaches, not a comparison of the two libraries on the same adapter — where a conformant driver is visible to both, pick either.

Which binding it builds is a variable, so a driver that disagrees with one of
them can be answered without editing anything:

```bash
make build              # wgpu24, the default
make build WGPU=wgpu    # the v22 binding
```

`install` and `cross` take it too. Pair the binary with the matching library:
a `wgpu24` build wants a v29-series `libwgpu_native`, a `wgpu` build wants v22,
and `TENSAI_WGPU_LIB` names the file to load when it is not the one beside the
binary.

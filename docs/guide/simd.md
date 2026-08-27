# SIMD Acceleration

tensai's fast kernels are AVX2, written with Go's experimental `simd/archsimd` package — still pure Go, no cgo, no assembly files.

```bash
GOEXPERIMENT=simd go build ./...
GOEXPERIMENT=simd go test -bench=Dot .
```

Requirements: amd64 and Go 1.26 or 1.27 (both `simd` API generations are supported via build tags). Every other build — other architectures, older Go, or `GOEXPERIMENT` unset — uses the portable fallbacks automatically, with identical results.

## What is vectorized

Where the AVX2 kernels apply today, and where they still could:

- [x] Matmul (`Dot`/`DotInto`) — used by `Dense`, `Conv2D` (im2col product), `KNN` distances, and autograd `MatMul`
- [x] ReLU / LeakyReLU forward & backward
- [x] Sigmoid / Tanh forward & backward (vectorized polynomial `exp`)
- [x] GELU forward & backward (vectorized `erf`)
- [x] LayerNorm forward & backward (vector row reductions)
- [x] Softmax / SoftmaxCrossEntropy exponentials and scaling
- [x] Adam / AdamW parameter update
- [x] SGD update (momentum form, same fused multiply-add loop as Adam)
- [x] Slice add & scale primitives (bias add, `Embedding` gradient scatter-add)
- [x] Transpose-free gradient matmul (`DotTAInto`) — `Dense`/`Conv2D` weight gradients no longer materialize `input^T` / `im2col^T`
- [x] Remaining transposes (`T`/`TInto`) — cache-blocked 32x32 tiles
- [x] Softmax backward row dot products (autograd) — fused AVX2 dot and Jacobian-vector accumulation
- [ ] MSE / BinaryCrossEntropy losses (BCE needs a vectorized `log`)
- [ ] Autograd element-wise backward passes (gradients accumulate with `+=`, so they need dedicated fused kernels)
- [ ] BatchNorm statistics (column-strided access needs a restructure)
- [ ] MaxPool2D window scan
- [ ] im2col / col2im gather-scatter (contiguous runs could use bulk copies)

The unchecked items are ordered roughly by expected impact; none of them show up prominently in training profiles today.

The int8/int4 quantized matmuls have their own AVX2 paths built on the 256-bit u8 x s8 pairwise multiply-add — see [Quantization](quantization.md).

## A live benchmark

`_example/plasma` animates a demoscene-style plasma in the terminal where the plasma function is a randomly weighted network (a CPPN) evaluated for every pixel of every frame as one batch. The status line shows the per-frame network time: 120x90 pixels runs at ~32 fps on the portable build and ~100 fps with `GOEXPERIMENT=simd` on the same machine.

```bash
go run ./_example/plasma
GOEXPERIMENT=simd go run ./_example/plasma
```

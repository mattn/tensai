# Quantization

Decoding a language model streams every weight once per token, so decode speed is memory bandwidth. Weight-only quantization pulls fewer bytes for the same matmul: int8 moves a quarter of the float32 bytes, int4 an eighth. tensai builds quantized *twins* of a `Matrix` — the float32 original stays untouched, and the quantized copy answers `MatVec`/`MatMul` with float32 inputs and outputs.

## int8: `quant.Quantize`

```go
q := quant.Quantize(w)     // *QMatrix, per-column scales
q.MatVec(x, out)                  // decode: one vector in, one out
q.MatMul(x, out)                  // prefill: batched rows
```

The int8 path is a full integer pipeline: weights are stored in interleaved row quads, activations are dynamically quantized to 7 bits per call, and the whole dot product runs on the 256-bit u8 x s8 pairwise multiply-add plus a widening pair-add — two instructions per column, four rows deep. That reaches memory bandwidth (~31GB/s of weights on 16 cores), which is the practical ceiling for a decode matvec.

## int4: `quant.Quantize4`

```go
q4, err := quant.Quantize4(w)  // *Q4Matrix, group-wise scales
```

int4 quantizes group-wise (a scale — or scale/min pair — per group of rows within a column) and accumulates in float32 at group boundaries. It halves the weights again relative to int8 — the difference between a 7B model fitting in RAM or not.

## MXFP4

`mxfp4.go` implements the MXFP4 block format gpt-oss ships its experts in: 32-element blocks of 4-bit floats sharing a power-of-two scale. Blocks are expanded through a one-shuffle table-lookup kernel, so gpt-oss-20b's experts stay in their native format end to end.

## Where quantization is used

- `_example/gpt2 -q8` quantizes the decode-path weights and doubles generation speed (23 → 46 tok/s), because decode streams the whole checkpoint per token
- The `tensai` command accepts `-q8` and `-q4`: each weight quantizes as it loads and its float32 copy dies immediately, so the full-precision model never has to fit in memory
- Quantized GGUF checkpoints (Q8_0, Q4_0, Q5_0, the Q4_K/Q5_K/Q6_K family, MXFP4) repack **directly** into these layouts from the memory-mapped file, skipping the float32 detour entirely — llama.cpp's own quantization stays intact
- On the GPU, `UploadQ8`/`UploadQ4` keep the weights quantized on the device and dequantize in registers — see [GPU (WebGPU)](gpu.md)

!!! note
    Quantized decoding stays coherent but greedy outputs no longer reproduce the float32 reference tokens exactly. Use the float32 path when you need bit-exact reference checks.

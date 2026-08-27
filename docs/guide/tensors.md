# Tensors and Matrices

tensai has two data types: the 2-D `Matrix` that all layers and the training loop operate on, and the rank-N `Tensor` that generalizes it with broadcasting and batched matrix multiplication. Both hold `tensai.Float` values, an alias for `float32`.

## Matrix

`Matrix` is a dense, row-major `MxN` matrix. By convention `M` is the batch size and `N` is the feature dimension — every operation in tensai is batched.

```go
m := tensai.NewMatrix(4, 2)                          // zero-filled
m, err := tensai.NewMatrixFromSlice(4, 2, []float32{ // from flat data
	0, 0,
	0, 1,
	1, 0,
	1, 1,
})
r := tensai.RandomMatrix(8, 8, rng)                  // uniform random

v := m.At(1, 0)      // read
m.Set(1, 0, 0.5)     // write
row := m.Row(1)      // a view of one row, no copy
t := m.T()           // transpose (materialized, cache-blocked)
```

The core products:

```go
c, err := tensai.Dot(a, b)      // matrix product, allocates the result
err = tensai.DotInto(out, a, b) // into a preallocated result
err = tensai.DotTAInto(out, a, b) // a^T @ b without materializing a^T
```

`Dot` is the kernel everything else rides on: `Dense` forward passes, `Conv2D`'s im2col product, `KNN` distances, autograd `MatMul`, and even the LLM examples' attention all end up here. Build with `GOEXPERIMENT=simd` and it becomes an AVX2 register-tiled kernel — see [SIMD Acceleration](simd.md).

## N-d Tensor

`Tensor` generalizes `Matrix` to any rank. Element-wise ops broadcast NumPy-style, and `MatMul` multiplies whole stacks of matrices at once — the leading batch axes broadcast too, so a shared 2-D weight applies to every sequence in a batch in one call:

```go
x := tensai.NewTensor(4, 6, 3)                    // (batch, position, channel)
mean, _ := tensai.NewTensorFromSlice([]float32{0.5, -1, 2}, 3)
centered, _ := x.Sub(mean)                        // (4,6,3) - (3)   -> (4,6,3)
h, _ := tensai.MatMul(centered, w)                // (4,6,3) @ (3,8) -> (4,6,8)

kt, _ := k.Transpose()                            // swap the last two axes
scores, _ := tensai.MatMul(q, kt)                 // (4,6,8) @ (4,8,6) -> (4,6,6)
scores.Scale(1 / float32(math.Sqrt(8)))
out, _ := tensai.MatMul(scores, v)                // attention for the whole batch
```

The per-matrix products inside a batched `MatMul` run on the same kernel as `Dot`, parallelized across the batch.

### Broadcasting rules

Shapes are compared from the trailing axis backwards, exactly as in NumPy: two axes are compatible when they are equal or one of them is 1, and missing leading axes count as 1. The result takes the larger extent on every axis.

### Views, reshape, transpose

Tensors are contiguous and row-major.

- `Reshape` returns a zero-copy view; one axis may be `-1` and is inferred from the rest
- `Matrix.Tensor()` and `Tensor.Matrix()` convert between the two types as zero-copy views
- `Transpose(perm...)` accepts an arbitrary axis permutation and **materializes** the result; with no arguments it swaps the last two axes

```go
flat, _ := x.Reshape(4, -1)   // (4,6,3) -> (4,18), no copy
m, _ := flat.Matrix()         // *Matrix view of the same data
back := m.Tensor()            // and back
```

See `_example/tensor` for a runnable tour: broadcasting, batched `MatMul`, and a full attention computation on tensors.

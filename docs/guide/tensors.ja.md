# テンソルと行列

tensai のデータ型は 2 つです。すべてのレイヤーと学習ループが扱う 2 次元の `Matrix` と、それをブロードキャストとバッチ行列積つきで一般化した任意ランクの `Tensor`。どちらも `tensai.Float` (= `float32`) を保持します。

## Matrix

`Matrix` は密な行優先の `MxN` 行列です。慣例として `M` がバッチサイズ、`N` が特徴次元 — tensai のすべての演算はバッチ化されています。

```go
m := tensai.NewMatrix(4, 2)                          // ゼロ埋め
m, err := tensai.NewMatrixFromSlice(4, 2, []float32{ // フラットなデータから
	0, 0,
	0, 1,
	1, 0,
	1, 1,
})
r := tensai.RandomMatrix(8, 8, rng)                  // 一様乱数

v := m.At(1, 0)      // 読み取り
m.Set(1, 0, 0.5)     // 書き込み
row := m.Row(1)      // 1 行のビュー、コピーなし
t := m.T()           // 転置 (実体化、キャッシュブロッキング済み)
```

中心となる積:

```go
c, err := tensai.Dot(a, b)        // 行列積、結果をアロケート
err = tensai.DotInto(out, a, b)   // 事前確保した結果へ
err = tensai.DotTAInto(out, a, b) // a^T を実体化せずに a^T @ b
```

`Dot` はすべてが乗るカーネルです。`Dense` の順伝播、`Conv2D` の im2col 積、`knn.Classifier` の距離計算、自動微分の `MatMul`、さらに LLM サンプルの attention まで、最後はここに行き着きます。`GOEXPERIMENT=simd` でビルドすると AVX2 のレジスタタイリングカーネルになります — [SIMD アクセラレーション](simd.md)参照。

## N 次元 Tensor

`Tensor` は `Matrix` を任意ランクに一般化します。要素ごとの演算は NumPy 流にブロードキャストし、`MatMul` は行列のスタックをまとめて掛けます — 先頭のバッチ軸もブロードキャストするので、共有の 2 次元重みが 1 回の呼び出しでバッチ内の全シーケンスに適用されます:

```go
x := tensai.NewTensor(4, 6, 3)                    // (batch, position, channel)
mean, _ := tensai.NewTensorFromSlice([]float32{0.5, -1, 2}, 3)
centered, _ := x.Sub(mean)                        // (4,6,3) - (3)   -> (4,6,3)
h, _ := tensai.MatMul(centered, w)                // (4,6,3) @ (3,8) -> (4,6,8)

kt, _ := k.Transpose()                            // 末尾 2 軸の入れ替え
scores, _ := tensai.MatMul(q, kt)                 // (4,6,8) @ (4,8,6) -> (4,6,6)
scores.Scale(1 / float32(math.Sqrt(8)))
out, _ := tensai.MatMul(scores, v)                // バッチ全体の attention
```

バッチ `MatMul` 内部の行列ごとの積は `Dot` と同じカーネルで走り、バッチ方向に並列化されます。

### ブロードキャスト規則

形状は末尾の軸から前へ向かって比較されます。NumPy とまったく同じで、2 つの軸は等しいか一方が 1 のとき互換で、足りない先頭の軸は 1 とみなされます。結果は各軸の大きい方の長さになります。

### ビュー、reshape、transpose

Tensor は連続した行優先メモリです。

- `Reshape` はゼロコピーのビューを返します。1 つの軸を `-1` にすると残りから推論されます
- `Matrix.Tensor()` と `Tensor.Matrix()` は 2 つの型をゼロコピービューで相互変換します
- `Transpose(perm...)` は任意の軸の置換を受け付け、結果を**実体化**します。引数なしなら末尾 2 軸の入れ替えです

```go
flat, _ := x.Reshape(4, -1)   // (4,6,3) -> (4,18)、コピーなし
m, _ := flat.Matrix()         // 同じデータの *Matrix ビュー
back := m.Tensor()            // そして戻す
```

ブロードキャスト、バッチ `MatMul`、attention の一連の計算を実際に動かせる `_example/tensor` も見てください。

# レイヤー・損失・最適化

## Layer インターフェース

すべてのレイヤーは同じインターフェースを実装します:

```go
type Layer interface {
	Init(inputCols int, rng *rand.Rand) (outputCols int, err error)
	Forward(input *Matrix) (*Matrix, error)
	Backward(gradOutput *Matrix) (*Matrix, error)
	Grads() (*Matrix, []Float)
	Params() (*Matrix, []Float)
	SetParams(weights *Matrix, bias []Float) error
}
```

`Init` は `Sequential.Compile` が呼び、列数をスタックに通していきます — `NewDense(8)` が出力幅しか指定しないのはこのためです。レイヤーは順伝播/逆伝播のスクラッチバッファを学習ステップをまたいで再利用するので、GC は学習ループに入り込みません (MLP の 1 ステップは約 29 アロケーション)。`Predict` は常に新しくアロケートした結果を返します。

## レイヤー一覧

| レイヤー | コンストラクタ | 備考 |
|---|---|---|
| Dense | `layer.NewDense(outCols)` | 全結合。Glorot/He スタイルの初期化 |
| Embedding | `layer.NewEmbedding(vocabSize, dim)` | 入力行は `Float` に格納された整数トークン ID。引いたベクトルは行方向に連結 |
| Conv2D | `layer.NewConv2D(inH, inW, inC, outC, kernel, stride, pad)` | im2col + `Dot` カーネル |
| MaxPool2D | `layer.NewMaxPool2D(inH, inW, channels, size)` | |
| BatchNorm | `layer.NewBatchNorm()` | 移動統計量はモデルと一緒に保存されます |
| LayerNorm | `layer.NewLayerNorm()` | 行ごとの正規化 |
| Dropout | `layer.NewDropout(rate)` | 学習時のみ有効 |

`Conv2D` と `MaxPool2D` は各行をチャンネル優先の画像として扱います: `index = (channel*height + y)*width + x`。`Dropout` と `BatchNorm` は `Fit`/`FitStep` 内では学習時の挙動、`Predict` 内では推論時の挙動へ自動的に切り替わります。

`Embedding` は行列のみの API を保っています。各入力行はトークン ID の列で、レイヤーは引いた埋め込みベクトルを列方向に連結します。たとえば `Compile(4, ...)` と `NewEmbedding(vocab, 8)` の組み合わせは、`Mx4` のトークン ID 行列を `LayerNorm`, `GELU`, `Dense` に食わせられる `Mx32` の密な特徴行列に変えます。

## 活性化関数

活性化関数もレイヤーです — 他と同じように `Add` します:

| 活性化 | 使い方 |
|---|---|
| ReLU | `&layer.ReLU{}` |
| LeakyReLU | `layer.NewLeakyReLU(0.01)` |
| GELU | `&layer.GELU{}` (SIMD ビルドではベクトル化 `erf`) |
| Sigmoid | `&layer.Sigmoid{}` |
| Tanh | `&layer.Tanh{}` |
| Softmax | `&layer.Softmax{}` (通常は `SoftmaxCrossEntropy` の方を使います) |

## 損失関数

| 損失 | 用途 | ターゲット |
|---|---|---|
| `loss.MeanSquaredError{}` | 回帰 | 予測と同じ形状 |
| `loss.SoftmaxCrossEntropy{}` | 多クラス分類 | クラス番号の `Mx1` 行列 |
| `loss.BinaryCrossEntropy{}` | 二値ターゲット | 予測と同じ形状 |

softmax は `SoftmaxCrossEntropy` の*内部*で適用されます (数値安定性のため行の最大値を引いてから)。モデルの最後は素の `Dense` で終わり、`Predict` は生のロジットを返すので、クラスは argmax で取ってください。

## オプティマイザ

| オプティマイザ | コンストラクタ |
|---|---|
| モーメンタム SGD | `optim.NewSGD(lr, momentum)` |
| Adam | `optim.NewAdam(lr)` |
| AdamW | `optim.NewAdamW(lr, weightDecay)` — decoupled weight decay |

Adam/AdamW のパラメータ更新は SIMD ビルドで AVX2 ベクトル化されるカーネルの 1 つです。

## k-NN ベースライン

`knn.New(k)` は学習不要のベースライン分類器です — `Fit` はデータを保持するだけで、`Predict` が距離行列を同じ SIMD matmul カーネルで組み立てます。MNIST の 5000 サンプルサブセットで約 91% (MLP は約 92%、CNN は約 95%) — ネットワークの隣に置く健全性チェックの床として便利です。

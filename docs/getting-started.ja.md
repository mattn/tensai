# はじめに

## インストール

```bash
go get github.com/mattn/tensai
```

tensai は小さなパッケージ群に分かれています。ルートの `tensai` パッケージがコア型 (`Matrix`, `Tensor`, `Float`, `Dot`) を持ち、残りはその 1 階層下にあります。

| import | 内容 |
|---|---|
| `github.com/mattn/tensai` | `Matrix`、N 次元 `Tensor`、`Dot`、SIMD matmul カーネル |
| `.../layer` | `Dense`, `Conv2D`, `BatchNorm`, `Dropout`, 活性化関数 |
| `.../loss` | `MeanSquaredError`, `SoftmaxCrossEntropy`, `BinaryCrossEntropy` |
| `.../optim` | `SGD`, `Adam`, `AdamW` |
| `.../model` | `Sequential`: `Compile` → `Fit` → `Predict`、JSON 保存/読込 |
| `.../autograd` | `Node`, `Param`, `Input`, `Trainer` |
| `.../rnn` | `Cell`, `LSTMCell`, `SelfAttention` |
| `.../dataset`, `.../knn` | データユーティリティと k-NN ベースライン |
| `.../quant` | int8 / int4 / MXFP4 の weight-only 量子化 |
| `.../gpu` | WebGPU バックエンド (`gpu.Open`) |

デフォルトビルドに外部依存はありません。Go 1.26 以降を推奨します。amd64 では `GOEXPERIMENT=simd` で AVX2 SIMD カーネルが有効になります (Go 1.26 と 1.27 の `simd` API 両方にビルドタグで対応)。それ以外のプラットフォームやビルドでは自動的にポータブル実装が使われます — 結果は同じで、速度が違うだけです。

```bash
# ポータブルビルド (全プラットフォーム)
go build ./...

# AVX2 アクセラレーション付きビルド (amd64, Go 1.26+)
GOEXPERIMENT=simd go build ./...
```

## 最初のモデル: XOR

XOR の学習はニューラルネットワークの「hello world」です — 線形モデルでは解けない最小の問題だからです。入力は `MxN` 行列で `M` がバッチサイズなので、XOR の 4 ケースは 1 つの `4x2` 行列になります:

```go
package main

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

func main() {
	net := model.NewSequential()
	net.Add(layer.NewDense(8))
	net.Add(&layer.Tanh{})
	net.Add(layer.NewDense(1))
	net.Add(&layer.Sigmoid{})

	// Compile(入力サイズ, 損失関数, オプティマイザ)
	if err := net.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05)); err != nil {
		panic(err)
	}

	// XOR の真理値表: 4 サンプル、各 2 特徴
	inputs, _ := tensai.NewMatrixFromSlice(4, 2, []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	targets, _ := tensai.NewMatrixFromSlice(4, 1, []float32{0, 1, 1, 0})

	if err := net.Fit(inputs, targets, 5000); err != nil {
		panic(err)
	}

	pred, _ := net.Predict(inputs)
	for r := 0; r < inputs.Rows; r++ {
		fmt.Printf("[%g %g] -> %.4f\n", inputs.At(r, 0), inputs.At(r, 1), pred.At(r, 0))
	}
}
```

## 分類

多クラス分類では、最後の `Dense` の幅をクラス数にして、損失を `SoftmaxCrossEntropy` に切り替えます:

```go
net := model.NewSequential()
net.Add(layer.NewDense(8))
net.Add(&layer.ReLU{})
net.Add(layer.NewDense(2)) // 出力幅 = クラス数

net.Compile(2, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.05))
```

`SoftmaxCrossEntropy` はターゲットとしてクラス番号の `Mx1` 行列を期待します。softmax は損失の内部で適用されるので、`Predict` は生のロジットを返します — 分類には argmax を使ってください。

## サンプルを動かす

リポジトリには、2 行の自動微分 hello-world から本物の LLM 推論まで、15 個の実行可能サンプルが入っています:

```bash
go run ./_example/xor
go run ./_example/mnist -model cnn
GOEXPERIMENT=simd go run ./_example/gpt2      # 初回に GPT-2 (~550MB) をダウンロード
GOEXPERIMENT=simd go run ./_example/qwen -q8  # 初回に Qwen2.5-0.5B (~1GB) をダウンロード
```

全リストは[サンプル](examples.md)を参照してください。

## ビルドフラグ早見表

| フラグ | 効果 |
|---|---|
| *(なし)* | ポータブルな純 Go ビルド、依存ゼロ |
| `GOEXPERIMENT=simd` | amd64 で AVX2 カーネル (Go 1.26/1.27) |
| `-tags wgpu` | wgpu-native **v22.1.0.5** 経由の WebGPU バックエンド (cgo 不要の依存 `ebitengine/purego` が 1 つ増えます) |
| `-tags wgpu24` | 刷新された wgpu-native C API (**v29** 系) に対する同じ API。WSL2 内の Mesa dozen のような非準拠アダプタに届きます |

詳細は [SIMD アクセラレーション](guide/simd.md)と [GPU (WebGPU)](guide/gpu.md) を参照してください。

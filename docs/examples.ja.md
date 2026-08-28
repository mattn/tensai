# サンプル

すべてのサンプルはリポジトリルートから `go run` で実行できます:

| サンプル | コマンド | 見せるもの |
|---|---|---|
| helloworld | `go run ./_example/helloworld` | 最小のプログラム: グラフ上で 2 つの値を加算 |
| dataset | `go run ./_example/dataset` | Dataset のワークフロー: shuffle, split, standardize, batches |
| xor | `go run ./_example/xor` | XOR の学習 (MSE + softmax の健全性チェック) |
| fizzbuzz | `go run ./_example/fizzbuzz` | 分類問題としての FizzBuzz |
| spiral | `go run ./_example/spiral` | 3 クラスのスパイラル分類 |
| iris | `go run ./_example/iris` | Iris の分類 |
| mnist | `go run ./_example/mnist` | MNIST 分類器 (`-model dense`, `cnn`, `knn`) と保存/読込 |
| charrnn | `go run ./_example/charrnn` | 自動微分エンジン上の文字レベル LSTM テキスト生成 |
| plasma | `go run ./_example/plasma` | ニューラルネットワークが描くターミナルプラズマ — 生きた SIMD ベンチマーク |
| dot | `go run ./_example/dot` | z = x + y グラフの Graphviz DOT エクスポート |
| tensor | `go run ./_example/tensor` | N 次元 Tensor ツアー: ブロードキャスト、バッチ MatMul、attention |
| wgpu | `go run -tags wgpu ./_example/wgpu` | WebGPU MatMul: アダプタ情報、CPU との照合、GPU vs CPU スイープ |
| gpt2 | `GOEXPERIMENT=simd go run ./_example/gpt2` | 公開 GPT-2 (124M) チェックポイントが純 Go でテキスト生成 |

gpt2 サンプルは初回に GPT-2 チェックポイント (~550MB) をダウンロードします。instruction-tuned モデル (Qwen2.5-0.5B から 7B まで 9 ファミリー) は `tensai` コマンドを使ってください: [LLM 推論](llm.md)を参照。

## MNIST

MNIST サンプルは標準の IDX gzip ファイルがなければ `_example/mnist/data` にダウンロードします (`MNIST_DIR` で別のキャッシュディレクトリを指定可能。生の IDX と `.gz` の両方を受け付けます):

```bash
go run ./_example/mnist                                  # dense MLP
go run ./_example/mnist -model cnn                       # Conv2D/MaxPool2D/Dropout + AdamW
go run ./_example/mnist -model knn                       # 学習不要の k-NN ベースライン
go run ./_example/mnist -model cnn -export mnist.tflite  # TFLite へエクスポート
```

5000 サンプルのサブセットで、k-NN ベースラインは約 91%、MLP は約 92%、CNN は約 95% です。学習する 2 つのバリアントは最後にモデルを保存し、再読込後に再スコアします。`-export` は LiteRT インタプリタ上で同一スコアになる TFLite flatbuffer を書き出します — [モデルフォーマット](formats.md)参照。

## charrnn

埋め込みのパブリックドメインテキストで文字レベル LSTM を学習し、`SaveParamsFile` でパラメータを保存し、新しいモデルに復元して、再読込したパラメータからサンプルを生成します。

## plasma

デモシーン風のプラズマをターミナルにアニメーションします。プラズマ関数はランダムに重み付けされたネットワーク (CPPN) で、毎フレーム全ピクセルを 1 バッチとして評価します。ステータス行にフレームあたりのネットワーク時間が出ます: ポータブルで約 32 fps、`GOEXPERIMENT=simd` で約 100 fps。`-seed` を変えると違う模様になります。

## wgpu

アダプタを表示し、GPU の結果を CPU カーネルと照合し、`-sweep` で行列サイズの階段を上りながら GPU が CPU を追い越す地点に印を付けます — [GPU (WebGPU)](guide/gpu.md) 参照。

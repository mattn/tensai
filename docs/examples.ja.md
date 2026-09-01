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
| tinygpt | `GOEXPERIMENT=simd go run ./_example/tinygpt` | n 次元自動微分エンジンでゼロから学習する文字レベル transformer (`-gpu` でデバイス学習) |
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

## tinygpt

charrnn と同じ埋め込みテキストで、小さな文字レベル transformer (トークン埋め込みと位置埋め込み、4 ヘッドの因果 attention と GELU の feed-forward を持つ pre-norm ブロック 2 段、最終 norm と出力射影) を学習し、そこからサンプリングします。パラメータは約 106k、`GOEXPERIMENT=simd` で 1 分ほど学習すれば、コーパスの文をそのまま再現するようになります。モデル全体が n 次元自動微分エンジンで書かれています: 活性は `(batch, sequence, model)` のテンソル、ヘッド分割は `Reshape` と `Transpose`、各ステップのバッファは `Tape` が再利用します。フラグは `-iters`, `-lr`, `-temp`, `-n`, `-seed`、それに形を変える `-model`, `-heads`, `-blocks`, `-batch`, `-seq`。

`-gpu` (wgpu ビルド時) はブロック全体をデバイスで学習します。値も勾配も Adam の更新もデバイスに留まり、毎ステップ帰ってくるのは損失だけです。速くなるかは形次第で、既定のサイズではテンソルが小さすぎて GPU が埋まらず AVX2 カーネルが勝ち、モデルを広げるとクロスオーバーします。AMD 780M では既定サイズで 23ms/step に対し `-gpu` が 62ms/step、`-model 256 -heads 8 -batch 16 -seq 64` では 282ms に対し 173ms でした。損失はどちらでも桁まで一致します。

## plasma

デモシーン風のプラズマをターミナルにアニメーションします。プラズマ関数はランダムに重み付けされたネットワーク (CPPN) で、毎フレーム全ピクセルを 1 バッチとして評価します。ステータス行にフレームあたりのネットワーク時間が出ます: ポータブルで約 32 fps、`GOEXPERIMENT=simd` で約 100 fps。`-seed` を変えると違う模様になります。

## wgpu

アダプタを表示し、GPU の結果を CPU カーネルと照合し、`-sweep` で行列サイズの階段を上りながら GPU が CPU を追い越す地点に印を付けます — [GPU (WebGPU)](guide/gpu.md) 参照。

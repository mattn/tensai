# tensai

<p align="center"><img src="../assets/logo.svg" width="420" alt="tensai"></p>

**tensai** は学習と実験のための小さな機械学習フレームワークです。純粋な Go で書かれており、順伝播・誤差逆伝播・最適化をすべて自前で実装しています。デフォルトビルドは外部依存ゼロ — cgo なし、アセンブリなし、C コンパイラ不要です。

```go
net := model.NewSequential()
net.Add(layer.NewDense(8))
net.Add(&layer.Tanh{})
net.Add(layer.NewDense(1))
net.Add(&layer.Sigmoid{})

net.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05))
net.Fit(inputs, targets, 5000)

pred, _ := net.Predict(inputs)
```

小さなフレームワークですが、届く範囲は意外と遠くまで伸びています。XOR を学習するのと同じカーネルが、公開されている GPT-2 チェックポイントを動かし、Qwen2.5 や Gemma 3 とチャットし、llama.cpp の GGUF 量子化をブロック単位で正確にデコードし、OpenAI 互換 API まで提供します — すべて純 Go です。

## ハイライト

- **行列と N 次元テンソル** — float32 の `Matrix` と任意ランクの `Tensor`。NumPy 流のブロードキャスト、バッチ `MatMul`、ゼロコピーの `Reshape` とビュー
- **レイヤー** — `Embedding`, `Dense`, `Conv2D`, `MaxPool2D`, `BatchNorm`, `LayerNorm`, `Dropout` と、`ReLU`, `LeakyReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`
- **学習** — `Sequential` モデルの `Compile` → `Fit` / `FitStep` → `Predict`、3 種類の損失関数、モーメンタム `SGD` / `Adam` / `AdamW`、データセットユーティリティ。MLP の 1 ステップは約 29 アロケーションで回ります
- **自動微分** — micrograd スタイルの行列上のリバースモードエンジン。その上に `rnn.Cell`, `rnn.LSTMCell`, `rnn.SelfAttention` が乗り、BPTT はただの Go のループです
- **SIMD アクセラレーション** — Go の実験的な `simd/archsimd` パッケージで書かれた AVX2 カーネル。`GOEXPERIMENT=simd` でビルドし、それ以外のビルドは自動的にポータブル実装へフォールバックします
- **WebGPU バックエンド** — `-tags wgpu` でバッチ `MatMul`、attention、量子化されたトランスフォーマーのデコード 1 ステップ丸ごとを wgpu-native の届く GPU 上で実行。`purego` 経由なので cgo 不要
- **int8 / int4 量子化** — メモリ帯域に到達する weight-only 量子化 matmul、gpt-oss 用の MXFP4 も
- **モデルフォーマット** — TFLite / ONNX エクスポート、safetensors の読み書き、K-quants まで揃った GGUF リーダー — エンコーダはすべてツリー内実装で、依存はゼロのまま
- **トークナイザ** — Hugging Face `tokenizer.json` のバイトレベル BPE (GPT-2, cl100k, o200k 系) と SentencePiece。リファレンス実装と完全一致することを検証済み
- **LLM 推論** — `_example/gpt2`、`_example/qwen` (9 モデルファミリー)、そして `run` / `chat` / `serve` サブコマンドを備えた `tensai` コマンド

## 次に読むページ

- [はじめに](getting-started.md) — インストールと最初のモデルの学習
- [ガイド](guide/tensors.md) — テンソル、レイヤー、学習、自動微分、量子化、SIMD、GPU
- [モデルフォーマット](formats.md) — TFLite, ONNX, safetensors, GGUF
- [LLM 推論](llm.md) — 純 Go で本物の言語モデルを動かす
- [サンプル](examples.md) — hello-world から 7B チャットモデルまで 15 個の実行可能サンプル

## 設計メモ

- すべての演算はバッチ化されています。入力は `MxN` 行列で、`M` がバッチサイズ、`N` が特徴次元です
- `Layer` インターフェースが `Forward` / `Backward` / `Params` / `Grads` を標準化しているので、新しいレイヤーの追加は素直に書けます
- `Dense` の重みは Glorot/He スタイルで初期化され、学習初期が安定します
- `SoftmaxCrossEntropy` は数値安定性のため softmax の前に行の最大値を引きます

## ライセンス

MIT — [Yasuhiro Matsumoto (a.k.a. mattn)](https://github.com/mattn)

# モデルフォーマット

tensai は 4 つのモデルフォーマットを話します。エンコーダ・デコーダはすべてツリー内実装です — FlatBuffers ライタ、protobuf エンコーダ、両チェックポイントリーダーがリポジトリ内に実装されているので、デフォルトビルドは依存ゼロのままです。

| パッケージ | 方向 | フォーマット |
|---|---|---|
| `encoding/safetensors` | 読み + 書き | 公開モデルの重みが通常配布されるチェックポイント形式 |
| `encoding/gguf` | 読み | llama.cpp のモデルコンテナ。量子化の階段も含む |
| `encoding/tflite` | 書き | TFLite/LiteRT ランタイム用の `.tflite` flatbuffers |
| `encoding/onnx` | 書き | ONNX (opset 13, FP32) |

## safetensors

`Open` はヘッダだけをパースし、各 `Tensor` 呼び出しはそのテンソルのバイト列だけを読みます。数ギガバイトのチェックポイントから残りをロードせずに単一テンソルを取り出せます。F32 はそのまま、F16/BF16/F64 は tensai の float32 に変換されます:

```go
import "github.com/mattn/tensai/encoding/safetensors"

f, err := safetensors.Open("model.safetensors")
defer f.Close()
w, err := f.Tensor("model.layers.0.attention.wq.weight") // *tensai.Tensor
```

`Save`/`SaveFile` はリファレンス実装がビット単位で読み戻せる F32 チェックポイントを書きます。相互運用性は双方向で検証済みです。

## GGUF

`encoding/gguf` は llama.cpp のコンテナを同じ遅延方式で読みます。`Open` は型付きメタデータ (`String`/`Int`/`Float`/`KV`) とテンソルディレクトリをパースし、各 `Tensor` 呼び出しがそのテンソルだけを読んで逆量子化します。対応型: F32/F16/BF16 に加え、ブロック量子化の Q8_0, Q4_0/Q4_1, Q5_0/Q5_1、K-quants の Q2_K〜Q6_K、非線形の IQ4_NL、gpt-oss の MXFP4 — llama.cpp 向けに通常公開されるチェックポイントの階段が丸ごと直接開きます。実際の llama.cpp 変換に対してブロック単位で正確なことを検証済みです。

`Names`, `Info`, `Metadata` はロードせずにチェックポイントを調べられます。次元は他のリーダーと同じく行優先で返ります。

!!! warning "RoPE の行順"
    形式から受け継ぐ注意点が 1 つ: llama.cpp のコンバータは attention の q/k 射影行を独自のインターリーブ RoPE 順に並べ替えます。GGUF の重みを half-split RoPE と組み合わせる場合はこれを元に戻す必要があります。

## TFLite エクスポート

```go
import tensaitflite "github.com/mattn/tensai/encoding/tflite"

// 学習後:
err := tensaitflite.MarshalFile("model.tflite", model)
```

対応レイヤー: Dense, Conv2D (VALID/SAME パディング), MaxPool2D, BatchNorm (Mul+Add に畳み込み), Dropout (除去), Softmax, ReLU/LeakyReLU/Sigmoid/Tanh。エクスポートされた畳み込みは TFLite の NHWC レイアウトに従います — エクスポート済みモデルには NHWC 入力を与えてください。重みの並べ替えはエクスポート時に処理されます。出力は LiteRT インタプリタ上で `Predict` と相対誤差 ~1e-7 で一致します (`encoding/tflite/verify_litert.py` 参照)。

[go-tflite](https://github.com/mattn/go-tflite) と組み合わせるときは import にエイリアスを付けてください (あちらもパッケージ名が `tflite` です):

```go
model := tflite.NewModelFromFile("mnist.tflite")
interpreter := tflite.NewInterpreter(model, nil)
interpreter.AllocateTensors()
copy(interpreter.GetInputTensor(0).Float32s(), image) // 28*28 floats, NHWC
interpreter.Invoke()
scores := interpreter.GetOutputTensor(0).Float32s() // 10 logits
```

## ONNX エクスポート

```go
import tensaionnx "github.com/mattn/tensai/encoding/onnx"

err := tensaionnx.MarshalFile("model.onnx", model)
```

レイヤー対応は TFLite エクスポートと同じですが、レイアウトの罠がありません: ONNX の畳み込みは NCHW で、これは tensai 自身のチャンネル優先行レイアウトそのものなので、エクスポートされたモデルは tensai と同じフラット化した行を `[1, C, H, W]` テンソルとして消費します。onnxruntime に対して相対誤差 ~1e-7 で検証済みです (`encoding/onnx/verify_onnxruntime.py` 参照)。

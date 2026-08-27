# Model Formats

tensai speaks four model formats, all with in-tree encoders and decoders — the FlatBuffers writer, the protobuf encoder, and both checkpoint readers are implemented in the repository, so the default build stays dependency-free.

| Package | Direction | Format |
|---|---|---|
| `encoding/safetensors` | read + write | The checkpoint format published model weights usually ship in |
| `encoding/gguf` | read | llama.cpp's model container, including the quantized ladder |
| `encoding/tflite` | write | `.tflite` flatbuffers for the TFLite/LiteRT runtimes |
| `encoding/onnx` | write | ONNX (opset 13, FP32) |

## safetensors

`Open` parses only the header; each `Tensor` call reads just that tensor's bytes, so single tensors come out of multi-gigabyte checkpoints without loading the rest. F32 loads as-is and F16/BF16/F64 convert to tensai's float32:

```go
import "github.com/mattn/tensai/encoding/safetensors"

f, err := safetensors.Open("model.safetensors")
defer f.Close()
w, err := f.Tensor("model.layers.0.attention.wq.weight") // *tensai.Tensor
```

`Save`/`SaveFile` write F32 checkpoints that the reference implementation reads back bit-for-bit; interoperability is verified in both directions.

## GGUF

`encoding/gguf` reads llama.cpp's container the same lazy way: `Open` parses the typed metadata (`String`/`Int`/`Float`/`KV`) and the tensor directory, and each `Tensor` call reads and dequantizes just that tensor. Supported types: F32/F16/BF16 plus the block-quantized Q8_0, Q4_0/Q4_1, Q5_0/Q5_1, the K-quants Q2_K through Q6_K, the nonlinear IQ4_NL, and gpt-oss's MXFP4 — so the whole ladder of checkpoints usually published for llama.cpp opens directly, verified block-exact against real llama.cpp conversions.

`Names`, `Info`, and `Metadata` inspect a checkpoint without loading it. Dimensions come back row-major like every other reader here.

!!! warning "RoPE row order"
    One caveat inherited from the format: llama.cpp's converter permutes attention q/k projection rows into its interleaved RoPE order, which consumers pairing GGUF weights with half-split RoPE must undo.

## TFLite export

```go
import tensaitflite "github.com/mattn/tensai/encoding/tflite"

// after training:
err := tensaitflite.MarshalFile("model.tflite", model)
```

Supported layers: Dense, Conv2D (VALID/SAME padding), MaxPool2D, BatchNorm (folded into Mul+Add), Dropout (dropped), Softmax, and the ReLU/LeakyReLU/Sigmoid/Tanh activations. Exported convolutions follow TFLite's NHWC layout — feed the exported model NHWC input; weight reordering is handled during export. Outputs match `Predict` to ~1e-7 relative error on the LiteRT interpreter (see `encoding/tflite/verify_litert.py`).

Alias the import when combining with [go-tflite](https://github.com/mattn/go-tflite), which also names its package `tflite`:

```go
model := tflite.NewModelFromFile("mnist.tflite")
interpreter := tflite.NewInterpreter(model, nil)
interpreter.AllocateTensors()
copy(interpreter.GetInputTensor(0).Float32s(), image) // 28*28 floats, NHWC
interpreter.Invoke()
scores := interpreter.GetOutputTensor(0).Float32s() // 10 logits
```

## ONNX export

```go
import tensaionnx "github.com/mattn/tensai/encoding/onnx"

err := tensaionnx.MarshalFile("model.onnx", model)
```

Same layer support as the TFLite export, but no layout gotcha: ONNX convolutions are NCHW, which is exactly tensai's channel-major row layout, so the exported model consumes the same flattened rows tensai does, as a `[1, C, H, W]` tensor. Verified against onnxruntime to ~1e-7 relative error (see `encoding/onnx/verify_onnxruntime.py`).

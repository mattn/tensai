# Examples

Every example is runnable from the repository root with `go run`:

| Example | Command | What it shows |
|---|---|---|
| helloworld | `go run ./_example/helloworld` | Smallest possible program: add two values on the graph |
| dataset | `go run ./_example/dataset` | Dataset workflow: shuffle, split, standardize, batches |
| xor | `go run ./_example/xor` | XOR training (MSE + a softmax sanity check) |
| fizzbuzz | `go run ./_example/fizzbuzz` | FizzBuzz as classification |
| spiral | `go run ./_example/spiral` | 3-class spiral classification |
| iris | `go run ./_example/iris` | Iris classification |
| mnist | `go run ./_example/mnist` | MNIST classifier (`-model dense`, `cnn`, or `knn`) with save/load |
| charrnn | `go run ./_example/charrnn` | Character-level LSTM text generation on the autograd engine |
| plasma | `go run ./_example/plasma` | Terminal plasma rendered by a neural network — a live SIMD benchmark |
| dot | `go run ./_example/dot` | Graphviz DOT export of the z = x + y graph |
| tensor | `go run ./_example/tensor` | Tour of the n-d Tensor: broadcasting, batched MatMul, attention |
| wgpu | `go run -tags wgpu ./_example/wgpu` | WebGPU MatMul: adapter info, CPU cross-check, GPU vs CPU sweep |
| gpt2 | `GOEXPERIMENT=simd go run ./_example/gpt2` | The published GPT-2 (124M) checkpoint generating text in pure Go |
| qwen | `GOEXPERIMENT=simd go run ./_example/qwen -q8` | Qwen2.5-0.5B-Instruct (and eight other families) chatting in pure Go |

The gpt2 example downloads the GPT-2 checkpoint (~550MB) on first run; qwen downloads Qwen2.5-0.5B-Instruct (~1GB).

## MNIST

The MNIST example downloads the standard IDX gzip files into `_example/mnist/data` when they are missing (set `MNIST_DIR` to use another cache directory; both raw IDX files and `.gz` variants are accepted):

```bash
go run ./_example/mnist                                  # dense MLP
go run ./_example/mnist -model cnn                       # Conv2D/MaxPool2D/Dropout + AdamW
go run ./_example/mnist -model knn                       # no-training k-NN baseline
go run ./_example/mnist -model cnn -export mnist.tflite  # export to TFLite
```

On the 5000-sample subset the k-NN baseline scores ~91% against ~92% for the MLP and ~95% for the CNN. Both trained variants finish by saving the model and re-scoring it after a reload, and `-export` writes a TFLite flatbuffer that scores identically on the LiteRT interpreter — see [Model Formats](formats.md).

## charrnn

Trains a character-level LSTM on an embedded public-domain text, saves the parameters with `SaveParamsFile`, restores them into a fresh model, and generates a sample from the reloaded parameters.

## plasma

Animates a demoscene-style plasma in the terminal where the plasma function is a randomly weighted network (a CPPN) evaluated for every pixel of every frame as one batch. The status line shows the per-frame network time: ~32 fps portable, ~100 fps with `GOEXPERIMENT=simd`. Try different `-seed` values for different effects.

## wgpu

Prints the adapter, cross-checks the GPU result against the CPU kernel, and `-sweep` walks a ladder of matrix sizes marking where the GPU overtakes the CPU — see [GPU (WebGPU)](guide/gpu.md).

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
| tinygpt | `GOEXPERIMENT=simd go run ./_example/tinygpt` | Character-level transformer trained from scratch on the n-d autograd engine (`-gpu` trains on the device) |
| plasma | `go run ./_example/plasma` | Terminal plasma rendered by a neural network — a live SIMD benchmark |
| dot | `go run ./_example/dot` | Graphviz DOT export of the z = x + y graph |
| tensor | `go run ./_example/tensor` | Tour of the n-d Tensor: broadcasting, batched MatMul, attention |
| wgpu | `go run -tags wgpu ./_example/wgpu` | WebGPU MatMul: adapter info, CPU cross-check, GPU vs CPU sweep |
| gpt2 | `GOEXPERIMENT=simd go run ./_example/gpt2` | The published GPT-2 (124M) checkpoint generating text in pure Go |
| flappy | `GOEXPERIMENT=simd go run ./_example/flappy` | Flappy Bird played by asking a model each step through `Engine.Score`, against a random flapper and a one-line heuristic: which question a scored token can decide, and `-screen` to watch |

The gpt2 example downloads the GPT-2 checkpoint (~550MB) on first run. For instruction-tuned models — nine families, from Qwen2.5-0.5B up to 7B — use the `tensai` command: see [LLM Inference](llm.md).

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

## tinygpt

Trains a small character-level transformer -- token and position embeddings, two pre-norm blocks with four-head causal attention and a GELU feed-forward, a final norm and an output projection -- on the same embedded text charrnn uses, then samples from it. About 106k parameters and a minute of training with `GOEXPERIMENT=simd`, after which it reproduces whole sentences of the corpus. The whole model is written against the n-dimensional autograd engine: activations are `(batch, sequence, model)` tensors, the per-head split is a `Reshape` plus a `Transpose`, and a `Tape` recycles each step's buffers. Flags: `-iters`, `-lr`, `-temp`, `-n`, `-seed`, plus `-model`, `-heads`, `-blocks`, `-batch` and `-seq` to change the shape.

`-gpu` (on a wgpu build) trains the whole block on the device: values, gradients and the Adam update stay there and only the loss comes back each step. Whether it is faster depends on the shape — at the default size the tensors are too small to keep a GPU busy and the AVX2 kernels win, while a wider model crosses over. On an AMD 780M, 24ms/step against 72ms with `-gpu` at the default size, and 282ms against 129ms at `-model 256 -heads 8 -batch 16 -seq 64`. The losses match to the digit either way.

## flappy

A Flappy Bird played several ways: a random flapper, a one-line heuristic
(flap when below the middle of the opening), and a language model asked
each step, its state written out as a sentence and the answer read from
`Engine.Score`. Nothing is trained. The question is the variable. The plain
player is asked "should the bird flap?" and answers yes or no; `-hint` adds
to its state where the bird is relative to the opening; `-compare` asks only
that comparison, "is the bird below the middle", yes or no. `-larger` asks
the same comparison as "which number is larger, 57 or 63?" with the two
numbers as the options, so the answer is a number the model writes and not
a yes it leans to; it is asked both ways round and averaged. `-rows` does
that on heights rounded to one digit, with `heur/rows` as the most that
rounding allows.

The result is the point. On a Ryzen 7735HS, 400 steps to a win:

| player | pipes | steps | per step |
|---|---|---|---|
| random | 0.3 | 16 | 0 |
| heuristic | 20 (win) | 400 | 0 |
| heur/rows | 16 | 325 | 0 |
| Qwen2.5-0.5B | 0 | 12 | 265ms |
| Qwen2.5-0.5B, hint | 0.3 | 19 | 291ms |
| Qwen2.5-0.5B, larger | 14 | 284 | 382ms |
| Qwen2.5-0.5B, rows | 16 | 325 | 191ms |
| Gemma-3-1B | 0 | 11 | 681ms |
| Gemma-3-1B, larger | 20 (win) | 400 | 693ms |
| Gemma-3-1B, rows | 16 | 325 | 421ms |
| K2-Horizon-7B | 0 | 9 | 4.7s |
| K2-Horizon-7B, hint | 0 | 9 | 5.6s |
| K2-Horizon-7B, comparison only | 0 | 11 | 2.8s |

Asked yes or no, no model plays, hinted or not, and the comparison row says
why: asked whether 65 is below 63 the 7B answers yes at 98%, and 87 below
63 at 88%. A single scored token carries the question's bias (yes, here)
and not a numeric comparison. Asked which number is larger, with the
numbers as the options, a 1B plays the heuristic's game to the step: 390
comparisons in a winning game, none wrong, down to 61 against 62. The same
mechanism that picks beer over coffee after work has nothing to say to yes
or no about a number, and everything to say when the number is the answer.
The physics stays in the code either way; what moved is the question the
model can answer.

```bash
GOEXPERIMENT=simd go run ./_example/flappy -episodes 3 -hint -compare -larger -rows
GOEXPERIMENT=simd go run ./_example/flappy -model ~/.cache/tensai/gemma-3-1b-it-Q8_0.gguf -larger -nobase -screen
GOEXPERIMENT=simd go run ./_example/flappy -show        # every decision, with its probability
GOEXPERIMENT=simd go run ./_example/flappy -nomodel     # the baselines only
```

`-screen` draws the game in the terminal as it is played, one frame per
step, the table under the last frame; `-nobase` skips the baselines.

## plasma

Animates a demoscene-style plasma in the terminal where the plasma function is a randomly weighted network (a CPPN) evaluated for every pixel of every frame as one batch. The status line shows the per-frame network time: ~32 fps portable, ~100 fps with `GOEXPERIMENT=simd`. Try different `-seed` values for different effects.

## wgpu

Prints the adapter, cross-checks the GPU result against the CPU kernel, and `-sweep` walks a ladder of matrix sizes marking where the GPU overtakes the CPU — see [GPU (WebGPU)](guide/gpu.md).

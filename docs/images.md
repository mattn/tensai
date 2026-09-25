# Image Generation

`tensai image` draws a picture from a prompt with [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), in the same pure Go as the rest: no cgo, no C compiler, and the same kernels that train a XOR network.

```
$ tensai image -size 256 -steps 20 "a calico cat asleep on a stack of books"
prompt: 18 tokens in 6s
transformer: loaded in 0s
step 20/20 in 3m39s
wrote out.png, 256x256
```

## What runs

Three models, in order, and only one is held at a time because each is seven gigabytes:

1. **The prompt encoder.** The transformer reads the prompt as the hidden states of a vision-language model, not as tokens, so the language half of Qwen3-VL 8B runs first — 36 layers of grouped-query attention. What it hands on is the last layer's output *before* the final norm, with the template's system preamble dropped. It is then released.
2. **The denoising transformer.** 32 single-stream blocks over a joint sequence: the prompt's tokens, then the image's latent grid. A block is attention and a SwiGLU feed-forward around layer norms that carry no weights of their own; the scale and gate they lack come from the timestep. Twenty of these passes walk the latent from noise to an image.
3. **The decoder.** A 2D autoencoder that turns each latent position into a sixteen-pixel square, with an alpha channel the checkpoint generates natively.

Every piece is checked against the reference implementation — diffusers for the transformer, the schedule and the decoder, transformers for the prompt encoder — and agrees to between 6.5e-8 and 2.4e-4, which is float32 rounding. The reference does not fit in memory at full depth, so the checks run it cut to a few blocks; what surrounds them is the same at any depth.

## Getting the checkpoint

```bash
tensai image "a calico cat asleep on a stack of books"
tensai image -model Comfy-Org/Qwen-Image-2.1 "a calico cat asleep on a stack of books"
```

`-model` names a checkpoint the way `run` and `chat` do: a repo, a name `tensai models` prints, or a path. Two repos download on first use, and a later run that finds a file missing (an interrupted download) fetches just that file; with everything in place nothing goes over the network.

- `Qwen/Qwen-Image-2.1`, the default: the diffusers checkpoint, about 31GB, cached as `~/.cache/tensai/Qwen/Qwen-Image-2.1` (a download from before tensai kept the org, under `~/.cache/tensai/Qwen-Image-2.1`, is still used). The weight files come from each component's index rather than a list, so a repository that re-splits them still resolves.
- `Comfy-Org/Qwen-Image-2.1`: ComfyUI's repackaging, cached as `~/.cache/tensai/Comfy-Org/Qwen-Image-2.1`. Its int8 files come to about 17GB, since the transformer and the prompt encoder are already eight bits. They hold each weight as int8 with a scale per row after rotating every 256 input columns by a Hadamard matrix, and the loader undoes both before quantizing its own way, so the two checkpoints run the same code. ComfyUI ships neither the tokenizer nor the VAE's latent statistics, and those two small files come from Qwen's repo.

A path may point at a directory in either layout: `text_encoder`, `transformer`, `vae` and `processor` for diffusers, or `text_encoders`, `diffusion_models` and `vae` with `processor/tokenizer.json` and `vae/config.json` beside them for ComfyUI, which picks the `_int8_convrot` files and falls back to `_bf16`. `-fetch`, which used to do the downloading, is still accepted and does nothing.

## Width and memory

The checkpoint ships as bfloat16, which is 14GB for the transformer alone, so the weights quantize as they load. Eight bits is the default and four is `-q4`:

| Width | Transformer | Velocity error, eight blocks |
|---|---|---|
| `-f32` | 28 GB | — |
| default | 7.0 GB | 9.1% |
| `-q4` | 3.5 GB | 13.5% |

The error is measured against the same model in float and does not compound with depth: one block costs 7.2% at eight bits and 12.8% at four, eight blocks 9.1% and 13.5%. Both draw the picture the prompt asked for; four bits interprets the detail differently.

Those figures predate rotation. Before quantizing, every weight is now rotated by a Hadamard matrix in groups of 256 input columns, and each product's input is rotated the same way; the matrix is orthogonal, so the product is unchanged, but the few outlier columns that set a row's scale are spread across their group, which is where eight bits lose the most. It is the scheme ComfyUI's int8 files use, so theirs arrive already in this form. On the first block with the reference inputs it halves the error at eight bits, 6.95% to 3.48%, and takes four bits from 19.3% to 16.2%; through four blocks of the whole model the velocity error at eight bits falls from 7.8% to 2.4% on the CPU and from 4.8% to 2.1% with `-gpu`. On the device the feed-forward's output projection reads a product made there, so it is rotated there too, by one product with H over its rows viewed in groups, a few per cent of the projection itself.

Quantizing seven billion parameters takes minutes and gives the same answer every time, so the result is written beside the checkpoint as `tensai-q8.cache` (or `tensai-q4.cache`) and mapped back afterwards. That makes the first run slow and every one after it quick: the prompt encoder starts in 7 seconds rather than 5m23s, and the transformer in 2 rather than 7m7s. The caches are tied to the checkpoint by the names, sizes and modification times of the `.safetensors` beside them, so a redownload rebuilds them.

## Size and time

`-size` is in pixels and rounds down to a multiple of 32; the decoder works in sixteen-pixel squares and the checkpoint's layout groups those in twos. On a 16-core laptop with the AVX2 build:

| Size | Latent tokens | A step | Twenty steps |
|---|---|---|---|
| 256x256 | 256 | 11s | 3m46s |
| 512x512 | 1024 | 50s | ~17m |

The step grows faster than the token count, because attention is a square in the sequence length.

## The GPU

`-gpu`, in a build with `-tags wgpu24`, moves every block's feed-forward onto the device: three projections and a gate, seventy per cent of a block's arithmetic, asking nothing of the position scheme or the mask. A 512x512 step falls from 50s to 31s, and the answer gets *closer* to what the float weights say — 4.7% against the CPU's 7.8% — because the device quantizes the activations of each product more finely.

Attention scores, softmax, and value aggregation also run on the GPU. The text prefix uses causal attention and the image queries read the full sequence; queries are tiled to bound score memory. Keys and values are uploaded once per block and shared by all query tiles. By default attention projections, norms, and three-axis rotary embedding remain on the CPU. Feed-forward dispatches are batched into one submission. No additional resident weights are needed.

The weights stay resident for the whole run, which is where the care goes. A buffer on this device is capped at 128MiB — int8 weights are under it and float ones are not — and past somewhere around five and a half gigabytes resident the driver drops the device, silently: allocations keep reporting success and the process falls over later. Nothing surfaces that, so `-gpu-budget` counts what is about to be uploaded and stops before it. Eight-bit feed-forward weights are 4.5GiB, above the default of 4, so `-gpu` alone fills the budget block by block and leaves the rest on the CPU, saying how many of them went up; `-q4` (about 2.5GiB including scale tables) or a raised budget takes all of them.

## Optional acceleration

`-gpu-projections` requires `-gpu` and streams Q/K/V and attention-output weights one at a time, for the blocks whose feed-forward is already resident. It includes the same ConvRot input rotation as the CPU path and reserves extra weight budget for the largest streamed projection. `-gpu` turns it on, and a budget with no room for it says so and leaves the projections on the CPU; `-gpu-projections=false` keeps them there on purpose, which can be the faster choice on smaller images or other devices.

`-gpu-vae` runs decoder convolutions on the GPU, uploading one filter at a time and tiling feature maps within the device storage limit. `-gpu` turns it on as well, and `-gpu-vae=false` declines it. It can be used independently of `-gpu`; the transformer releases its device before decoding. Norms, activations, upsampling and decoder attention remain on the CPU. Float accumulation order differs, so output pixels need not be bit-identical.

For fewer denoising passes, `-turbo-lora` accepts the [Viggle Qwen-Image-2.1 v0.2.1 six-step adapter](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo). Download the adapter separately (rank 128 is about 680 MB; rank 256 about 1.3 GB):

```bash
curl -fL -o qwen-turbo-r128.safetensors \
  https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors

tensai image -size 512 -gpu -gpu-budget 5 -gpu-projections -gpu-vae \
  -turbo-lora qwen-turbo-r128.safetensors "a calico cat asleep on a stack of books"
```

This selects six steps unless `-steps` was explicitly supplied, in which case it must be 6. It requires `-cfg 1` and no negative prompt. The adapter's six raw sigma nodes receive the base model's resolution shift without terminal stretching. Adapter updates stay separate from the quantized base weights, and their inputs retain ConvRot rotation. GPU adapter weights are streamed instead of kept resident; on the CPU they occupy float32 memory (about 1.3 GB for rank 128, 2.6 GB for rank 256).

The adapter is a preview and changes the generated image. It is an explicit speed/quality choice, not a lossless replacement for the default twenty-step model. Supplying `-steps 6` without the adapter does not enable this mode.

In one 512x512 comparison on a Ryzen 7 7735HS / integrated Radeon through WSL Direct3D12, with the SIMD `wgpu24` build, cached int8 base weights, prompt `ドラゴンボール` and seed 1:

| Configuration | Total elapsed |
|---|---:|
| `-gpu -gpu-budget 5`, default 20 steps | 562.91 s |
| Above plus `-gpu-projections -gpu-vae -turbo-lora` (v0.2.1 rank 128, 6 steps) | 221.14 s |

A separate one-block benchmark at the same token count measured 872 ms with CPU attention projections and 569 ms with streamed GPU projections (three timed iterations). Against the float reference through four blocks, relative velocity error was 2.36% on the quantized CPU path, 2.10% with the existing GPU path and 1.39% with streamed projections. This check uses no LoRA.

These are single runs, including loading and PNG output, rather than a guarantee for other drivers or prompts. Both images were inspected; the composition was similar but details differed. Separately, a synthetic 512px decoder benchmark with weights already loaded measured 34.3 s on CPU and 25.4 s on GPU; the full pipeline has additional loading and memory pressure.

## Flags

```
tensai image [flags] <prompt>

  -model string   a repo, a cached name, or a path (default "Qwen/Qwen-Image-2.1")
  -o string       where to write the picture (default "out.png")
  -size int       width and height in pixels (default 256)
  -steps int      denoising steps (default 20)
  -seed int       noise seed (default 1)
  -q4             quantize to four bits instead of eight
  -f32            keep the weights as floats, which needs about 42GB
  -negative str   what to steer away from; needs -cfg above 1
  -cfg float      how far to steer away from it (default 1, off)
  -gpu            run feed-forward and attention on the GPU
  -gpu-budget num gigabytes of weights the GPU may hold (default 4)
  -gpu-projections stream attention projections to the GPU (on with -gpu)
  -gpu-vae        run decoder convolutions on the GPU (on with -gpu)
  -turbo-lora str path to a Viggle six-step LoRA (sets steps to 6)
  -cpuprofile str write a CPU profile to this file
  -q              print nothing but errors
```

## Guidance

`-cfg` above 1 turns on classifier-free guidance: each step asks the
transformer twice, once for what the prompt wants and once for what
`-negative` does, and follows the difference past the first. It sharpens
the picture — colours separate, texture comes up — at the price of
doubling what a step costs, which is why it is off by default.

```bash
tensai image -cfg 4 -negative "blurry, low quality, watermark" \
  "a calico cat asleep on a stack of books"
```

A 256x256 run goes from 3m46s to 6m29s. Both prompts go through the
encoder in one load, so guidance costs nothing extra there.

## What is not here

Text-to-image only. The checkpoint also edits images, which is not implemented. The prompt encoder runs its language half, so the vision tower is unused, and the transformer's key-value cache — which the reference reuses across steps, since the prompt modulates from a timestep of zero and never changes — is not built either.

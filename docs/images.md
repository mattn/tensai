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
tensai image -fetch "a calico cat asleep on a stack of books"
```

`-fetch` downloads the components under `~/.cache/tensai/Qwen-Image-2.1`, about 31GB in all, and then draws. It resumes what an interrupted run left behind, and the weight files come from each component's index rather than a list, so a repository that re-splits them still resolves. A second `-fetch` finds everything in place and costs nothing.

`-model` takes a name under the cache or a path to any directory holding `text_encoder`, `transformer`, `vae` and `processor`.

## Width and memory

The checkpoint ships as bfloat16, which is 14GB for the transformer alone, so the weights quantize as they load. Eight bits is the default and four is `-q4`:

| Width | Transformer | Velocity error, eight blocks |
|---|---|---|
| `-f32` | 28 GB | — |
| default | 7.0 GB | 9.1% |
| `-q4` | 3.5 GB | 13.5% |

The error is measured against the same model in float and does not compound with depth: one block costs 7.2% at eight bits and 12.8% at four, eight blocks 9.1% and 13.5%. Both draw the picture the prompt asked for; four bits interprets the detail differently.

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

What is left on the CPU is attention and the norms. The transformer's rotary embedding has three position axes and rotates adjacent pairs, which is neither convention the shipped GPU kernels carry, so moving that needs a kernel of its own.

The weights stay resident for the whole run, which is where the care goes. A buffer on this device is capped at 128MiB — int8 weights are under it and float ones are not — and past somewhere around five and a half gigabytes resident the driver drops the device, silently: allocations keep reporting success and the process falls over later. Nothing surfaces that, so `-gpu-budget` counts what is about to be uploaded and refuses first. Eight-bit feed-forward weights are 4.5GiB, above the default of 4, so `-gpu` alone asks for `-q4` (2.3GiB) or a raised budget.

## Flags

```
tensai image [flags] <prompt>

  -model string   a name under the cache, or a path (default "Qwen-Image-2.1")
  -o string       where to write the picture (default "out.png")
  -size int       width and height in pixels (default 256)
  -steps int      denoising steps (default 20)
  -seed int       noise seed (default 1)
  -q4             quantize to four bits instead of eight
  -f32            keep the weights as floats, which needs about 42GB
  -negative str   what to steer away from; needs -cfg above 1
  -cfg float      how far to steer away from it (default 1, off)
  -fetch          download the checkpoint first, about 31GB
  -gpu            run the feed-forward on the GPU
  -gpu-budget num gigabytes of weights the GPU may hold (default 4)
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

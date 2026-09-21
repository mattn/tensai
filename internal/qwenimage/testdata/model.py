# Dumps a reference run of the whole denoising transformer, cut to a few
# blocks so it fits in memory: everything around the blocks is the same
# at any depth. See gen.py for the environment.
#
#   venv/bin/python model.py ~/.cache/tensai/Qwen-Image-2.1/transformer . 2 13 6 5
#
import sys, os, json, torch
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

tdir, out_dir, layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
text_len, h, w = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

index = json.load(open(f"{tdir}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
keep = lambda n: (not n.startswith("transformer_blocks.")) or int(n.split(".")[1]) < layers
shards, state = {}, {}
for name, shard in index.items():
    if not keep(name):
        continue
    if shard not in shards:
        shards[shard] = load_file(os.path.join(tdir, shard))
    state[name] = shards[shard][name].float()
shards.clear()

model = QwenImage21Transformer2DModel(num_layers=layers)
model.load_state_dict(state)
model.eval().float()

g = torch.Generator().manual_seed(11)
latents = torch.randn(1, h * w, 64, generator=g)
text = torch.randn(1, text_len, 4096, generator=g) * 0.5
t = torch.tensor([0.7])
# Text-to-image: no condition images, so the vision-language mask is
# False over the prompt, and the pipeline appends one slot per 2x2 group
# of target latents.
img_mask = torch.zeros(1, text_len + h * w // 4, dtype=torch.bool)
img_mask[:, text_len:] = True

with torch.no_grad():
    out = model(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=text,
        img_shapes=[[(1, h, w)]],
        img_mask=img_mask,
        return_dict=False,
    )[0][:, -h * w:]

tag = f"{layers}_{text_len}_{h}_{w}"
for name, v in [("lat", latents), ("txt", text), ("vel", out)]:
    v.contiguous().numpy().astype('<f4').tofile(f"{out_dir}/model_{name}_{tag}.f32")
print("layers", layers, "velocity", tuple(out.shape), "rms %.5f" % out.pow(2).mean().sqrt())

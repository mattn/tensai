# Dumps a reference run of one transformer block: random inputs, the
# block's real weights, and what diffusers makes of them. See gen.py for
# the environment this needs. Run from this directory:
#
#   venv/bin/python block.py ~/.cache/tensai/Qwen-Image-2.1/transformer . 0 48
#
import sys, torch
from safetensors.torch import load_file
import json, os
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

tdir, out_dir, layer, seq = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dim, heads, head_dim = 4096, 32, 128

index = json.load(open(f"{tdir}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
prefix = f"transformer_blocks.{layer}."
shards = {}
state = {}
for name, shard in index.items():
    if not name.startswith(prefix):
        continue
    if shard not in shards:
        shards[shard] = load_file(os.path.join(tdir, shard))
    state[name[len(prefix):]] = shards[shard][name].float()

block = QwenImage21TransformerBlock(dim=dim, num_attention_heads=heads, attention_head_dim=head_dim)
block.load_state_dict(state)
block.eval().float()

g = torch.Generator().manual_seed(7)
x = torch.randn(1, seq, dim, generator=g) * 0.5
# Two modulation rows, as the model builds them: the sampled timestep's
# and t=0's, with the second half of the sequence reading the first.
modulation = torch.randn(2, 4 * dim, generator=g) * 0.1
mask = torch.zeros(seq, dtype=torch.bool)
mask[seq // 2:] = True
# Rotary angles, one per head pair, drawn straight rather than built from
# the position scheme: the block only ever sees the angles.
ang = torch.randn(seq, head_dim // 2, generator=g)
rot = torch.polar(torch.ones_like(ang), ang)

with torch.no_grad():
    y = block(hidden_states=x, modulation=modulation, rotary_emb=rot, target_token_mask=mask)

for name, t in [("x", x), ("mod", modulation), ("cos", torch.cos(ang)), ("sin", torch.sin(ang)), ("y", y)]:
    t.contiguous().numpy().astype('<f4').tofile(f"{out_dir}/block_{name}_{seq}.f32")
print("block", layer, "seq", seq, "out", tuple(y.shape), "mean %.6f" % y.mean())

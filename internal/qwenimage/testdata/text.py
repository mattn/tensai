# Dumps a reference run of the prompt encoder, cut to a few layers so it
# fits in memory: the wiring is the same at any depth. Also reports how
# many tokens of the template's preamble the pipeline drops. See gen.py
# for the environment, plus `pip install transformers`.
#
#   venv/bin/python text.py ~/.cache/tensai/Qwen-Image-2.1 . 4
#
import sys, os, json, torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

root, out_dir, layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
tdir = os.path.join(root, "text_encoder")

sys_prompt = "Comprehend and analyze the provided prompt."
template = (
    f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
tok = AutoTokenizer.from_pretrained(os.path.join(root, "processor"))
text = template.format("a red cube on a white table")
ids = tok(text, return_tensors="pt").input_ids

# What the pipeline drops: the system message through the chat template.
sys_ids = tok.apply_chat_template(
    [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}],
    tokenize=True, return_dict=False,
)
drop = len(sys_ids[0]) if isinstance(sys_ids[0], list) else len(sys_ids)

cfg = AutoConfig.from_pretrained(tdir).text_config
cfg.num_hidden_layers = layers
model = Qwen3VLTextModel(cfg)

index = json.load(open(f"{tdir}/model.safetensors.index.json"))["weight_map"]
def keep(n):
    if not n.startswith("model.language_model."):
        return False
    parts = n.split(".")
    if parts[2] == "norm":
        return False  # neutralized below, and it may live in a later shard
    return parts[2] != "layers" or int(parts[3]) < layers
shards, state = {}, {}
for name, shard in index.items():
    if not keep(name):
        continue
    if shard not in shards:
        shards[shard] = load_file(os.path.join(tdir, shard))
    state[name[len("model.language_model."):]] = shards[shard][name].float()
shards.clear()
state["norm.weight"] = torch.ones(cfg.hidden_size)
model.load_state_dict(state)
model.eval().float()

# The pipeline reads the last decoder layer's output, before the final
# norm. Which entry of hidden_states that is moved between transformers
# releases, so do what the pipeline does: a hook that hands the norm's
# input back as its output, which neutralizes it either way.
handle = model.norm.register_forward_hook(lambda m, args, output: args[0])
try:
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
finally:
    handle.remove()
hidden = out.hidden_states[-1]

tag = f"{layers}"
ids.to(torch.int32).numpy().astype('<i4').tofile(f"{out_dir}/text_ids_{tag}.i32")
hidden.contiguous().numpy().astype('<f4').tofile(f"{out_dir}/text_hidden_{tag}.f32")
print("layers", layers, "tokens", ids.shape[1], "drop", drop, "hidden", tuple(hidden.shape), "rms %.5f" % hidden.pow(2).mean().sqrt())

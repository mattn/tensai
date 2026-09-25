# Dumps a reference run of Qwen2-Audio's audio side for one 16kHz mono WAV:
# the samples, the log-mel features, and what the encoder and projector
# make of them, the encoder cut to a few layers so torch runs it quickly
# (the wiring is the same at any depth). The .f32 files are not in the
# repo; regenerate them before running the comparison tests:
#
#   python3 -m venv venv
#   venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
#   venv/bin/pip install transformers safetensors soundfile
#   venv/bin/python ref.py ~/.cache/tensai/Qwen/Qwen2-Audio-7B-Instruct speech.wav . 2
#
import sys, os, json
import numpy as np, soundfile as sf, torch
from safetensors import safe_open
from transformers import WhisperFeatureExtractor
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

root, wav, out_dir, layers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def dump(name, a):
    np.ascontiguousarray(a, dtype=np.float32).tofile(os.path.join(out_dir, name))

samples, rate = sf.read(wav, dtype="float32")
assert rate == 16000 and samples.ndim == 1, (rate, samples.shape)
dump("samples.f32", samples)

fe = WhisperFeatureExtractor.from_pretrained(root)
feat = fe(samples, sampling_rate=16000, return_attention_mask=True, padding="max_length", return_tensors="pt")
mel, frames = feat.input_features, int(feat.attention_mask.sum())
dump("mel.f32", mel[0].numpy())  # (128, 3000)

cfg = json.load(open(os.path.join(root, "config.json")))
ac = Qwen2AudioEncoderConfig(**{**cfg["audio_config"], "encoder_layers": layers})
enc = Qwen2AudioEncoder(ac).eval()
proj = torch.nn.Linear(ac.d_model, 4096)

index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
state, pstate = {}, {}
for name, shard in index.items():
    if name.startswith("audio_tower."):
        key = name[len("audio_tower."):]
        if key.startswith("layers.") and int(key.split(".")[1]) >= layers:
            continue
        with safe_open(os.path.join(root, shard), "pt") as f:
            state[key] = f.get_tensor(name).float()
    elif name.startswith("multi_modal_projector.linear."):
        with safe_open(os.path.join(root, shard), "pt") as f:
            pstate[name.split(".")[-1]] = f.get_tensor(name).float()
enc.load_state_dict(state)
proj.load_state_dict(pstate)

feat_len, out_len = enc._get_feat_extract_output_lengths(torch.tensor(frames))
feat_len, out_len = int(feat_len), int(out_len)
seq = (mel.shape[-1] - 2) // 2 + 1
mask = torch.zeros(1, 1, seq, seq)
mask[..., feat_len:] = float("-inf")
with torch.no_grad():
    hidden = enc(mel, attention_mask=mask).last_hidden_state
    audio = proj(hidden)[0, :out_len]
dump(f"audio_{layers}.f32", audio.numpy())  # (out_len, 4096)
print(f"frames {frames}, encoder tokens {feat_len}, audio tokens {out_len}")

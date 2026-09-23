# Dumps the rotary angles and the block-causal key limits diffusers
# builds for a text-to-image sequence. See gen.py for the environment.
#
#   venv/bin/python layout.py . 13 6 5
#
import sys, torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Rope,
    QwenImage21Transformer2DModel,
    _qwenimage21_prefix_segments,
)

out_dir, text_len, h, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seq = text_len + h * w

# Text-to-image: no condition images, so the mask is False over the
# prompt and True over the target image's latents.
image_pad_mask = torch.zeros(seq, dtype=torch.bool)
image_pad_mask[text_len:] = True
img_shapes = [(1, h, w)]

rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
freqs = rope(img_shapes, image_pad_mask, torch.device("cpu"))
torch.cos(torch.angle(freqs)).contiguous().numpy().astype('<f4').tofile(f"{out_dir}/rope_cos_{text_len}_{h}_{w}.f32")
torch.sin(torch.angle(freqs)).contiguous().numpy().astype('<f4').tofile(f"{out_dir}/rope_sin_{text_len}_{h}_{w}.f32")

image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)
prefix_len = int((~target_token_mask).sum())
segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

# Turn the segments into the per-query key limit the Go side carries: a
# text query reads through itself, a condition-image query through the
# end of its block, and a target query the whole sequence.
limit = [seq] * seq
for start, end, is_text in segments:
    for i in range(start, end):
        limit[i] = i + 1 if is_text else end
torch.tensor(limit, dtype=torch.int32).numpy().astype('<i4').tofile(f"{out_dir}/rope_limit_{text_len}_{h}_{w}.i32")
target_token_mask.to(torch.int32).numpy().astype('<i4').tofile(f"{out_dir}/rope_target_{text_len}_{h}_{w}.i32")
print("seq", seq, "freqs", tuple(freqs.shape), "segments", segments)

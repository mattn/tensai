# Dumps the reference the Go decoder is checked against: a fixed latent
# and what diffusers makes of it. The .f32 files it writes are not in the
# repo (they only mean anything beside the 1.3 GB checkpoint), so
# regenerate them before running the comparison test:
#
#   python3 -m venv venv
#   venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
#   venv/bin/pip install "git+https://github.com/huggingface/diffusers"
#   venv/bin/python gen.py ~/.cache/tensai/Qwen-Image-2.1/vae . 16
#
import sys, torch
from diffusers import AutoencoderKLQwenImage21

vae_dir, out_dir, hw = sys.argv[1], sys.argv[2], int(sys.argv[3])
vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float32)
vae.eval()

g = torch.Generator().manual_seed(1234)
z = torch.randn(1, 64, 1, hw, hw, generator=g, dtype=torch.float32)

with torch.no_grad():
    out = vae.decode(z).sample  # (1, 4, 1, hw*16, hw*16)

z.numpy().astype('<f4').tofile(f"{out_dir}/latent_{hw}.f32")
out.numpy().astype('<f4').tofile(f"{out_dir}/decoded_{hw}.f32")
print("latent", tuple(z.shape), "decoded", tuple(out.shape))
print("decoded stats: min %.6f max %.6f mean %.6f" % (out.min(), out.max(), out.mean()))

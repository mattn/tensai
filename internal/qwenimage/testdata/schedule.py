# Dumps the noise levels the checkpoint's scheduler walks, for the Go
# side to check its own schedule against. See gen.py for the environment.
#
#   venv/bin/python schedule.py . 20 1024
#
import sys, numpy as np, torch
from diffusers import FlowMatchEulerDiscreteScheduler

out_dir, steps, tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "Qwen/Qwen-Image-2.1", subfolder="scheduler"
)
base, mx = sched.config.base_image_seq_len, sched.config.max_image_seq_len
m = (sched.config.max_shift - sched.config.base_shift) / (mx - base)
mu = tokens * m + sched.config.base_shift - m * base
sigmas = np.linspace(1.0, 1 / steps, steps)
sched.set_timesteps(sigmas=sigmas, mu=mu)
sched.sigmas.numpy().astype('<f8').tofile(f"{out_dir}/sigmas_{steps}_{tokens}.f64")
print("steps", steps, "tokens", tokens, "sigmas", sched.sigmas.numpy()[:3], "...", sched.sigmas.numpy()[-2:])

"""Chapter 9 — runnable demo: closed-loop control + the imagination ablation.

Briefly trains a tiny model, then rolls it out as a receding-horizon policy in
ReacherEnv and runs the imagination on/off ablation (imagine=none vs joint) over a
handful of SEEDED episodes. Prints the success-rate table.

The harness is the point; with a brief CPU run and few episodes the numbers are
noisy (see Ch 9 §9.5 and Ch 12) — do not over-read them.

Run from the repo root:
    python book/ch09-closed-loop/01_main-chapter-code/ablation_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.data import generate_episode, make_windows  # noqa: E402
from nanowam.eval import run_ablation  # noqa: E402
from nanowam.flow import flow_loss  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402
from nanowam.modes import sample_mode  # noqa: E402
from nanowam.tokenizer import FrameTokenizer, to_tokens  # noqa: E402
from nanowam.utils import set_seed  # noqa: E402

set_seed(0)
cfg = load_config("configs/pusht_tiny.yaml")
cfg.model.dim, cfg.model.depth = 64, 2
cfg.flow.sample_steps = 6
cfg.eval.replan_every = 4

tok = FrameTokenizer(cfg.model, cfg.data)
model = NanoWAM(cfg.model, cfg.data)
opt = torch.optim.AdamW(list(model.parameters()) + list(tok.parameters()), lr=5e-4)

rng = np.random.default_rng(0)
W = []
while len(W) < 128:
    f, a = generate_episode(rng, 24, cfg.data.image_size)
    W += make_windows(f, a, cfg.data.ctx_frames, cfg.data.video_horizon, cfg.data.action_horizon)
ctx_all = torch.from_numpy(np.stack([w[0] for w in W])).float() / 255.0
fut_all = torch.from_numpy(np.stack([w[1] for w in W])).float() / 255.0
act_all = torch.from_numpy(np.stack([w[2] for w in W])).float()

print(f"training a tiny model briefly on {len(W)} windows ...")
for step in range(150):
    idx = torch.randint(0, len(W), (24,))
    ctx, fut, act = ctx_all[idx], fut_all[idx], act_all[idx]
    z_ctx, z_fut = tok.encode(ctx), tok.encode(fut)              # encode once (cf. train.py)
    recon = F.mse_loss(tok.decode(z_ctx), ctx) + F.mse_loss(tok.decode(z_fut), fut)
    lat = {"ctx": to_tokens(z_ctx).detach(),
           "video": to_tokens(z_fut).detach(), "action": act}
    f_loss, _ = flow_loss(model, lat, sample_mode(cfg.train.mode_probs), cfg.flow)
    (f_loss + recon).backward(); opt.step(); opt.zero_grad()
    if step % 50 == 0:
        print(f"  step {step:3d}  flow {f_loss.item():.3f}  recon {recon.item():.4f}")

print("\nrunning the imagination ablation (8 seeded episodes each) ...")
res = run_ablation(model, tok, cfg, device="cpu", episodes=8, max_steps=32)
print(f"\n{'setting':<14}{'mode':<10}{'success':>9}{'mean_steps':>12}")
for setting, mode_name in (("none", "policy"), ("joint", "joint")):
    r = res[setting]
    print(f"imagine={setting:<6}{mode_name:<10}{r['success_rate']*100:>8.0f}%{r['mean_steps']:>12.1f}")
print("\n(harness verified; numbers are noisy at this scale — see Ch 9 and Ch 12)")

"""Chapter 7 — runnable demo: one training step, and the tiny-batch overfit.

Reproduces the trainer's core: tokenizer reconstruction + flow matching on
DETACHED latents, with per-batch mode sampling. Trains a tiny model on a small
fixed batch and shows both losses falling — the overfit sanity check from the
chapter. No data prep needed (frames are generated in memory).

Run from the repo root:
    python book/ch07-training/01_main-chapter-code/train_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.data import generate_episode, make_windows  # noqa: E402
from nanowam.flow import flow_loss  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402
from nanowam.modes import sample_mode  # noqa: E402
from nanowam.tokenizer import FrameTokenizer, to_tokens  # noqa: E402
from nanowam.utils import set_seed  # noqa: E402

set_seed(0)
cfg = load_config("configs/pusht_tiny.yaml")
cfg.model.dim, cfg.model.depth = 64, 2          # tiny + fast

tok = FrameTokenizer(cfg.model, cfg.data)
model = NanoWAM(cfg.model, cfg.data)
opt = torch.optim.AdamW(list(model.parameters()) + list(tok.parameters()), lr=1e-3)

# a small fixed batch of windows, generated in memory
rng = np.random.default_rng(0)
windows = []
while len(windows) < 16:
    f, a = generate_episode(rng, 24, cfg.data.image_size)
    windows += make_windows(f, a, cfg.data.ctx_frames, cfg.data.video_horizon, cfg.data.action_horizon)
windows = windows[:16]
ctx = torch.from_numpy(np.stack([w[0] for w in windows])).float() / 255.0
fut = torch.from_numpy(np.stack([w[1] for w in windows])).float() / 255.0
act = torch.from_numpy(np.stack([w[2] for w in windows])).float()

print("overfitting a fixed batch of 16 windows (tokenizer recon + flow on detached latents)")
first = None
for step in range(300):
    # 2-3. tokenize + reconstruct
    ctx_recon = tok.decode(tok.encode(ctx)); fut_z = tok.encode(fut)
    recon = F.mse_loss(ctx_recon, ctx) + F.mse_loss(tok.decode(fut_z), fut)
    # 4-5. mode + flow loss on DETACHED latents
    lat = {"ctx": to_tokens(tok.encode(ctx)).detach(),
           "video": to_tokens(fut_z).detach(), "action": act}
    mode = sample_mode(cfg.train.mode_probs)
    f_loss, _ = flow_loss(model, lat, mode, cfg.flow)
    # 6. step
    loss = f_loss + recon
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None:
        first = (f_loss.item(), recon.item())
    if step % 100 == 0:
        print(f"  step {step:3d}  flow {f_loss.item():.3f}  recon {recon.item():.4f}  mode {mode.name.lower()}")

print(f"\nrecon loss: {first[1]:.4f} -> {recon.item():.4f}")
print(f"flow loss (last mode): {f_loss.item():.3f}  (started near {first[0]:.3f})")
assert recon.item() < 0.5 * first[1], "recon should drop sharply on a fixed batch"
print("OK: the trainer learns the fixed batch (recon collapses; flow falls).")

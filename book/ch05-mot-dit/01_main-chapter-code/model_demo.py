"""Chapter 5 — runnable demo: a NanoWAM forward pass and the zero-init property.

Builds the MoT Diffusion Transformer, runs one forward pass, checks the per-stream
velocity shapes, and confirms that a freshly initialized model predicts *exactly
zero* velocity (AdaLN-zero + zero-init heads). Also prints the parameter count and
shows that per-stream FFN routing is what makes the model an MoT.

Run from the repo root:
    python book/ch05-mot-dit/01_main-chapter-code/model_demo.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402

cfg = load_config("configs/pusht_tiny.yaml")
model = NanoWAM(cfg.model, cfg.data)
print(f"NanoWAM: {model.num_params() / 1e6:.2f}M params "
      f"(dim={cfg.model.dim}, depth={cfg.model.depth}, heads={cfg.model.heads})")

B = 2
C = cfg.model.latent_channels
ctx = torch.randn(B, model.n_ctx, C)
vid = torch.randn(B, model.n_video, C)
act = torch.randn(B, model.n_action, cfg.data.action_dim)
tau = torch.rand(B)
mode = torch.zeros(B, dtype=torch.long)

v_vid, v_act = model(ctx, vid, act, tau, mode)
print(f"video tokens in {tuple(vid.shape)} -> velocity {tuple(v_vid.shape)}")
print(f"action chunk in {tuple(act.shape)} -> velocity {tuple(v_act.shape)}")
assert v_vid.shape == vid.shape and v_act.shape == act.shape

# zero-init property: a fresh model predicts exactly zero velocity
vmax = max(v_vid.abs().max().item(), v_act.abs().max().item())
print(f"max |velocity| at init: {vmax:.2e}   (expected ~0 from AdaLN-zero + zero-init heads)")
assert vmax == 0.0

# the MoT signature: three separate per-stream feed-forward networks per block
n_ffn = len(model.blocks[0].ffn)
print(f"per-block feed-forward networks (one per stream): {n_ffn}")
assert n_ffn == 3
print("OK: forward pass, zero-init start, and per-stream FFN routing all verified.")

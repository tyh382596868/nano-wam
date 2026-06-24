"""Chapter 11 — runnable demo: causal mask, exact KV cache, and dreaming.

No training needed -- this demonstrates the *mechanism*:
  1. a causal model blocks the future (perturbing a later frame leaves an earlier
     output unchanged),
  2. the cached path (encode_prefix + forward_suffix) is NUMERICALLY EQUAL to the
     uncached forward (atol 1e-5),
  3. dream_rollout produces n_blocks * Hv valid frames autoregressively.

Run from the repo root:
    python book/ch11-causal-kv-cache/01_main-chapter-code/causal_demo.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402
from nanowam.modes import Mode  # noqa: E402
from nanowam.rollout import dream_rollout  # noqa: E402
from nanowam.tokenizer import FrameTokenizer  # noqa: E402

torch.manual_seed(0)
cfg = load_config("configs/causal_tiny.yaml")
cfg.model.dim, cfg.model.depth = 64, 3          # tiny; causal requires Ha == Hv (config has 4,4)
model = NanoWAM(cfg.model, cfg.data).eval()
print(f"causal model: Ha={cfg.data.action_horizon} == Hv={cfg.data.video_horizon}, "
      f"mask {tuple(model.full_mask.shape)}")

B, C = 2, cfg.model.latent_channels
ctx = torch.randn(B, model.n_ctx, C)
vid = torch.randn(B, model.n_video, C)
act = torch.randn(B, model.n_action, cfg.data.action_dim)
tau = torch.rand(B)
mode_id = torch.full((B,), int(Mode.JOINT), dtype=torch.long)

# 1. causality: perturbing the LAST frame must not change the FIRST frame's output
with torch.no_grad():
    v0, _ = model(ctx, vid, act, tau, mode_id)
    vid2 = vid.clone(); vid2[:, -model.P:] += 5.0           # perturb the last frame's tokens
    v1, _ = model(ctx, vid2, act, tau, mode_id)
delta = (v0[:, :model.P] - v1[:, :model.P]).abs().max().item()
print(f"causality: max change to FIRST frame from perturbing LAST frame = {delta:.2e}  (expect ~0)")
assert delta < 1e-5

# 2. exact KV cache: encode_prefix + forward_suffix == single-pass forward
with torch.no_grad():
    v_full = model(ctx, vid, act, tau, mode_id)
    prefix = model.encode_prefix(ctx, mode_id)
    v_cached = model.forward_suffix(vid, act, tau, mode_id, prefix)
err = max((a - b).abs().max().item() for a, b in zip(v_full, v_cached))
print(f"KV cache exactness: max |cached - uncached| = {err:.2e}  (expect < 1e-5)")
assert err < 1e-5

# 3. autoregressive dreaming: n_blocks * Hv valid frames
tok = FrameTokenizer(cfg.model, cfg.data)
init = torch.rand(cfg.data.ctx_frames, 3, cfg.data.image_size, cfg.data.image_size)
n_blocks = 5
frames = dream_rollout(model, tok, init, n_blocks, cfg, device="cpu")
print(f"dream: {n_blocks} blocks -> {frames.shape[0]} frames {tuple(frames.shape[1:])}, "
      f"range [{frames.min():.2f}, {frames.max():.2f}]")
assert frames.shape[0] == n_blocks * cfg.data.video_horizon
assert frames.min() >= 0 and frames.max() <= 1

print("\nOK: causal mask blocks the future; the KV cache is exact; dreaming runs unbounded.")

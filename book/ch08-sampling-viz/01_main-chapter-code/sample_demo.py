"""Chapter 8 — runnable demo: generate -> decode -> compare to a baseline.

Briefly trains a tiny model, samples future frames in WORLD mode (conditioned on
the ground-truth actions), decodes the predicted latents back to pixels, prints
the frame MSE next to the copy-last-frame baseline, and saves a ground-truth-vs-
prediction strip. The point is the full visualization pipeline + honest baseline,
not quality (a brief CPU run is undertrained).

Run from the repo root:
    python book/ch08-sampling-viz/01_main-chapter-code/sample_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.data import generate_episode, make_windows  # noqa: E402
from nanowam.flow import flow_loss, sample  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402
from nanowam.modes import Mode, sample_mode  # noqa: E402
from nanowam.tokenizer import FrameTokenizer, to_tokens, from_tokens  # noqa: E402
from nanowam.utils import set_seed  # noqa: E402

set_seed(0)
cfg = load_config("configs/pusht_tiny.yaml")
cfg.model.dim, cfg.model.depth = 64, 2          # tiny + fast (quality is not the point)
cfg.flow.sample_steps = 8

tok = FrameTokenizer(cfg.model, cfg.data)
model = NanoWAM(cfg.model, cfg.data)
opt = torch.optim.AdamW(list(model.parameters()) + list(tok.parameters()), lr=5e-4)

# in-memory dataset
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
    loss = f_loss + recon
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"  step {step:3d}  flow {f_loss.item():.3f}  recon {recon.item():.4f}")

# sample WORLD mode on a few held-out-ish windows and decode
model.eval(); tok.eval()
h = cfg.data.image_size // cfg.model.patch_size
with torch.no_grad():
    ctx, fut, act = ctx_all[:4], fut_all[:4], act_all[:4]
    out = sample(model, {"ctx": to_tokens(tok.encode(ctx)), "action": act}, Mode.WORLD, cfg.flow)
    pred = tok.decode(from_tokens(out["video"], cfg.data.video_horizon, h, h)).clamp(0, 1)

v_mse = F.mse_loss(pred, fut).item()
b_mse = F.mse_loss(ctx[:, -1:].expand_as(fut), fut).item()   # copy-last-frame baseline
print(f"\nfuture-frame MSE: model {v_mse:.4f}  vs  copy-last-frame {b_mse:.4f}")
print("(a brief CPU run is undertrained; the pipeline is the point — see Ch 12 for scale)")

# save a GT (top) vs prediction (bottom) strip for sample 0
def strip(x):  # (T,3,H,W) -> HWC uint8 row
    return np.moveaxis(np.concatenate(list(x.numpy()), axis=2), 0, -1)
img = np.concatenate([strip(fut[0]), strip(pred[0])], axis=0)
try:
    import imageio.v2 as imageio
    out_png = Path(__file__).with_name("world_strip.png")
    imageio.imwrite(out_png, (img * 255).astype(np.uint8))
    print(f"saved {out_png} (top = ground-truth future, bottom = prediction)")
except Exception as e:
    print(f"(install imageio to save the strip: {e})")

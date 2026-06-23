"""Chapter 3 — runnable demo: train the tokenizer to reconstruct reacher frames.

Trains the conv autoencoder on the reconstruction loss alone (no WAM yet) for a
few hundred steps and saves an input-vs-reconstruction strip. You should see the
reconstruction go from noise to recognizable squares, and the printed MSE fall.

Run from the repo root:
    python book/ch03-tokenizer/01_main-chapter-code/tokenizer_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.data import generate_episode  # noqa: E402
from nanowam.tokenizer import FrameTokenizer  # noqa: E402

cfg = load_config("configs/pusht_tiny.yaml")
cfg.model.latent_channels = 4
tok = FrameTokenizer(cfg.model, cfg.data)
opt = torch.optim.AdamW(tok.parameters(), lr=2e-3)

# a small fixed batch of reacher frames to overfit
rng = np.random.default_rng(0)
frames = np.concatenate([generate_episode(rng, 16, cfg.data.image_size)[0] for _ in range(8)])
frames = torch.from_numpy(frames).float() / 255.0  # (N, 3, H, W) in [0,1]

print(f"training tokenizer on {len(frames)} frames ...")
for step in range(400):
    idx = torch.randint(0, len(frames), (32,))
    x = frames[idx]
    recon = tok(x)
    loss = F.mse_loss(recon, x)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"  step {step:4d}  recon MSE {loss.item():.4f}")
print(f"  final       recon MSE {loss.item():.4f}")

# save an input (top) vs reconstruction (bottom) strip for 6 frames
with torch.no_grad():
    x = frames[:6]
    r = tok(x).clamp(0, 1)
top = np.concatenate(list(x.numpy()), axis=2)
bot = np.concatenate(list(r.numpy()), axis=2)
strip = np.moveaxis(np.concatenate([top, bot], axis=1), 0, -1)  # HWC
try:
    import imageio.v2 as imageio
    out = Path(__file__).with_name("recon_strip.png")
    imageio.imwrite(out, (strip * 255).astype(np.uint8))
    print(f"saved {out} (top = input, bottom = reconstruction)")
except Exception as e:
    print(f"(install imageio to save the strip: {e})")

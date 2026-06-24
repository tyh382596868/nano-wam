"""Chapter 4 — runnable demo: rectified flow on a 2-D toy.

Learns a velocity field that flows a Gaussian into a target distribution (eight
Gaussians on a circle) using the exact three-step flow-matching loss and an Euler
sampler from the chapter. Prints how far samples sit from the nearest mode before
vs after training, and saves a noise -> generated -> target scatter.

Reuses nanowam.flow.interpolate so the *mechanism* is the library's, not a copy.

Run from the repo root:
    python book/ch04-flow-matching/01_main-chapter-code/flow_2d_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.flow import interpolate  # noqa: E402  (the same interpolant the WAM uses)

torch.manual_seed(0)


CENTERS = np.stack([[2 * np.cos(a), 2 * np.sin(a)]
                    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])  # 8 modes


def sample_target(n):
    """Eight Gaussians arranged on a circle (the canonical flow-matching toy)."""
    idx = np.random.randint(0, len(CENTERS), n)
    pts = CENTERS[idx] + 0.1 * np.random.randn(n, 2)
    return torch.tensor(pts, dtype=torch.float32)


def time_features(tau, n_freq=6):
    """Sinusoidal features of tau in [0,1] — the same trick the WAM uses for tau."""
    freqs = 2 ** torch.arange(n_freq, dtype=torch.float32) * np.pi
    a = tau[:, None] * freqs[None]
    return torch.cat([tau[:, None], torch.sin(a), torch.cos(a)], dim=-1)


class VelocityNet(nn.Module):
    """v(x, tau): (B,2) + sinusoidal(tau) -> (B,2)."""

    def __init__(self, hidden=256, n_freq=6):
        super().__init__()
        self.n_freq = n_freq
        in_dim = 2 + (1 + 2 * n_freq)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, tau):
        return self.net(torch.cat([x, time_features(tau, self.n_freq)], dim=-1))


def dist_to_modes(pts):
    """Mean distance of points to the nearest of the 8 mode centers."""
    c = torch.tensor(CENTERS, dtype=torch.float32)        # (8, 2)
    d = torch.cdist(pts, c)                               # (N, 8)
    return d.min(dim=1).values.mean().item()


@torch.no_grad()
def sample(model, n, steps=50):
    x = torch.randn(n, 2)
    dt = 1.0 / steps
    for i in range(steps):
        tau = torch.full((n,), i * dt)
        x = x + dt * model(x, tau)
    return x


model = VelocityNet()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

before = dist_to_modes(sample(model, 2000))
print(f"mean distance to nearest mode BEFORE training: {before:.3f}")

for step in range(4000):
    x1 = sample_target(512)                       # data at tau=1
    eps = torch.randn_like(x1)                    # noise at tau=0
    tau = torch.rand(x1.shape[0])
    x_tau, u = interpolate(x1, eps, tau)          # <-- the library's interpolant
    loss = F.mse_loss(model(x_tau, tau), u)       # regress predicted velocity onto u
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"  step {step:4d}  flow loss {loss.item():.4f}")

after = dist_to_modes(sample(model, 2000))
print(f"mean distance to nearest mode AFTER  training: {after:.3f}  (was {before:.3f})")

# save a noise -> generated -> target scatter
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = sample(model, 2000).numpy()
    tgt = sample_target(2000).numpy()
    noise = torch.randn(2000, 2).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, pts, title in zip(axes, [noise, gen, tgt], ["noise (tau=0)", "generated", "target"]):
        ax.scatter(pts[:, 0], pts[:, 1], s=2)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    out = Path(__file__).with_name("flow_2d.png")
    fig.tight_layout(); fig.savefig(out, dpi=90)
    print(f"saved {out}")
except Exception as e:
    print(f"(install matplotlib to save the scatter: {e})")

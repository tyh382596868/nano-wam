"""Chapter 6 — runnable demo: one network, four modes.

Shows the stream mask for each of the four modes and confirms that every mode
produces a finite, differentiable flow loss on a fresh model — the contract the
training loop (Chapter 7) relies on. Also samples modes from the config's
distribution to show the UniDiffuser-style per-batch schedule.

Run from the repo root:
    python book/ch06-modes/01_main-chapter-code/modes_demo.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.flow import flow_loss  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402
from nanowam.modes import Mode, MODE_TABLE, sample_mode  # noqa: E402

cfg = load_config("configs/pusht_tiny.yaml")
cfg.model.dim, cfg.model.depth = 64, 2          # tiny + fast for the demo
model = NanoWAM(cfg.model, cfg.data)

B, P, C = 4, model.P, cfg.model.latent_channels
batch = {
    "ctx":    torch.randn(B, model.n_ctx, C),
    "video":  torch.randn(B, model.n_video, C),
    "action": torch.randn(B, model.n_action, cfg.data.action_dim),
}

print("mode      noise_video  noise_action   loss      what it is")
print("-" * 70)
what = {Mode.POLICY: "act from context",
        Mode.WORLD: "predict frames from actions",
        Mode.INVERSE: "infer action from observed future",
        Mode.JOINT: "imagine frames + actions"}
for mode in Mode:
    m = MODE_TABLE[mode]
    loss, _ = flow_loss(model, batch, mode, cfg.flow)
    loss.backward()                                   # must be differentiable
    assert torch.isfinite(loss)
    print(f"{mode.name:<9} {str(m.noise_video):<12} {str(m.noise_action):<13} "
          f"{loss.item():<9.3f} {what[mode]}")

# the per-batch mode schedule (UniDiffuser-style)
counts = Counter(sample_mode(cfg.train.mode_probs).name for _ in range(2000))
print("\nsampled modes over 2000 draws (config mode_probs):")
for name, c in counts.most_common():
    print(f"  {name:<8} {c/2000:.2f}")
print("\nOK: all four modes give a finite, differentiable loss from one network.")

"""Chapter 2 — runnable demo: episodes, windows, and the data contract.

Generates one reacher episode, slices it into (context, future, action) windows,
prints the shapes, and saves a frame strip so you can *see* the agent drift toward
the goal. No external data needed.

Run from the repo root:
    python book/ch02-window-contract/01_main-chapter-code/reacher_demo.py
"""
import sys
from pathlib import Path

import numpy as np

# make `import nanowam` work when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.data import generate_episode, make_windows  # noqa: E402

K, Hv, Ha, SIZE = 2, 4, 8, 64

rng = np.random.default_rng(0)
frames, actions = generate_episode(rng, length=24, size=SIZE)
print(f"episode: frames {frames.shape} uint8, actions {actions.shape} float32")

windows = make_windows(frames, actions, K, Hv, Ha)
ctx, future, action = windows[0]
print(f"windows from this episode: {len(windows)}")
print(f"  ctx    {ctx.shape}   future {future.shape}   action {action.shape}")

# sanity: the action sign should match the agent's motion direction
print(f"  first action (normalized displacement): {action[0]}")

# save a frame strip (context | future) if imageio is available
try:
    import imageio.v2 as imageio

    strip = np.concatenate(list(np.concatenate([ctx, future], axis=0)), axis=2)  # (3,H,W*)
    strip = np.moveaxis(strip, 0, -1)  # HWC for imageio
    out = Path(__file__).with_name("window_strip.png")
    imageio.imwrite(out, strip)
    print(f"saved {out} (left {K} = context, right {Hv} = future)")
except Exception as e:  # imageio optional
    print(f"(install imageio to save the strip: {e})")

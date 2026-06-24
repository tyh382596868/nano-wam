"""Chapter 10 — runnable demo: the EpisodeSource adapter, without downloading data.

Implements a tiny mock EpisodeSource, runs it through make_windows, normalizes the
actions and saves stats, shards to disk, and reads it back with WindowDataset --
confirming the round-trip and the same contract real data would use. Also
exercises to_chw_uint8 on an odd-sized image and shows the lazy lerobot import.

No external data and no training needed.

Run from the repo root:
    python book/ch10-real-data/01_main-chapter-code/adapter_demo.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.data import WindowDataset, make_windows  # noqa: E402
from nanowam.sources import EpisodeSource, LeRobotSource, to_chw_uint8  # noqa: E402

cfg = load_config("configs/pusht_tiny.yaml")
S = cfg.data.image_size

# 1. frame coercion: odd-sized HWC uint8 -> (3, S, S) uint8
odd = (np.random.rand(40, 50, 3) * 255).astype(np.uint8)
out = to_chw_uint8(odd, S)
print(f"to_chw_uint8: {odd.shape} HWC uint8 -> {out.shape} {out.dtype}")
assert out.shape == (3, S, S)


# 2. a mock source standing in for a real dataset
class MockSource(EpisodeSource):
    def iter_episodes(self):
        rng = np.random.default_rng(0)
        for _ in range(3):
            frames = (rng.random((20, 3, S, S)) * 255).astype(np.uint8)
            actions = rng.standard_normal((20, cfg.data.action_dim)).astype(np.float32) * 5.0
            yield frames, actions


windows = []
for frames, actions in MockSource().iter_episodes():
    windows += make_windows(frames, actions, cfg.data.ctx_frames,
                            cfg.data.video_horizon, cfg.data.action_horizon)
print(f"mock source -> {len(windows)} windows (same contract as the reacher)")

with tempfile.TemporaryDirectory() as root:
    cfg.data.root = root
    # 3. normalize actions + save stats, then shard
    acts = np.concatenate([w[2] for w in windows]).reshape(-1, cfg.data.action_dim)
    mean = acts.mean(0, keepdims=True).astype(np.float32)
    std = (acts.std(0, keepdims=True) + 1e-6).astype(np.float32)
    np.savez(Path(root) / "stats.npz", action_mean=mean, action_std=std)
    print(f"action stats: mean={mean.ravel()} std={std.ravel()}")

    (Path(root) / "train").mkdir()
    ctx = np.stack([w[0] for w in windows])
    fut = np.stack([w[1] for w in windows])
    act = ((np.stack([w[2] for w in windows]) - mean) / std).astype(np.float32)
    np.savez_compressed(Path(root) / "train" / "shard_000.npz", ctx=ctx, future=fut, action=act)

    # 4. read it back through the SAME WindowDataset the trainer uses
    ds = WindowDataset(cfg.data, "train")
    item = ds[0]
    print(f"WindowDataset item: ctx {tuple(item['ctx_frames'].shape)} "
          f"future {tuple(item['future_frames'].shape)} actions {tuple(item['actions'].shape)}")
    print(f"loaded normalization stats: mean shape {ds.action_mean.shape}")
    assert item["ctx_frames"].shape == (cfg.data.ctx_frames, 3, S, S)
    assert ds.action_mean is not None

# 5. lazy import: constructing LeRobotSource does not import lerobot; iterating errors clearly
src = LeRobotSource(cfg.data, "lerobot/pusht")
try:
    next(src.iter_episodes())
except ImportError as e:
    print(f"lazy lerobot import (expected without the package): {e}")

print("\nOK: real data plugs into the exact same window/shard pipeline.")

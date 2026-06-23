"""Toy windowed dataset.

nano-wam's default M2 data source is a fully self-contained **procedural** task
("reacher"): a white agent square moves toward a gray goal square on a black
background. It renders to frames and has a 2D action (the per-step displacement),
so the WAM has a real world (agent moves) and real actions to predict — with
*zero* external sim dependencies, so training/CI run anywhere. The pushT / LeRobot
adapters (the real targets in DESIGN.md §7) plug in behind the same window
interface at M5.

Shard format (written by scripts/prepare_data.py) under `<root>/<split>/`:
    shard_*.npz with
        ctx     (n, K,  3, H, W) uint8
        future  (n, Hv, 3, H, W) uint8
        action  (n, Ha, Da)      float32

WindowDataset yields, per item:
    {"ctx_frames": (K,3,H,W) float[0,1],
     "future_frames": (Hv,3,H,W) float[0,1],
     "actions": (Ha, Da) float}
"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig


# --------------------------------------------------------------------------- #
# Procedural "reacher" generator (used by prepare_data.py)
# --------------------------------------------------------------------------- #

def render_frame(agent: np.ndarray, goal: np.ndarray, size: int) -> np.ndarray:
    """Render one (3, H, W) uint8 frame: white agent, gray goal, black bg."""
    img = np.zeros((3, size, size), dtype=np.uint8)
    r = max(2, size // 12)

    def stamp(pos, value):
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        x0, x1 = max(0, cx - r), min(size, cx + r)
        y0, y1 = max(0, cy - r), min(size, cy + r)
        img[:, y0:y1, x0:x1] = value

    stamp(goal, 110)    # gray goal (draw first so agent overlays it)
    stamp(agent, 255)   # white agent
    return img


def generate_episode(rng: np.random.Generator, length: int, size: int):
    """One episode. Returns frames (L,3,H,W) uint8, actions (L,Da=2) float32.

    Action at step t is the normalized displacement applied to reach t+1, so an
    inverse/forward-dynamics relationship genuinely exists in the data.
    """
    margin = size // 6
    agent = rng.uniform(margin, size - margin, size=2)
    goal = rng.uniform(margin, size - margin, size=2)
    speed = size / 16.0

    frames, actions = [], []
    for _ in range(length):
        frames.append(render_frame(agent, goal, size))
        direction = goal - agent
        dist = np.linalg.norm(direction) + 1e-6
        step = direction / dist * min(speed, dist)
        step = step + rng.normal(0, speed * 0.05, size=2)  # small noise
        actions.append((step / speed).astype(np.float32))   # ~[-1, 1]
        agent = np.clip(agent + step, 0, size - 1)
    return np.stack(frames), np.stack(actions)


def make_windows(frames: np.ndarray, actions: np.ndarray, K: int, Hv: int, Ha: int):
    """Slide windows over one episode. Yields (ctx, future, action) arrays.

    A window at start t needs: K context frames [t-K .. t), Hv future frames
    [t .. t+Hv), and Ha actions [t .. t+Ha) taken from frame t onward.
    """
    L = len(frames)
    horizon = max(Hv, Ha)
    out = []
    for t in range(K, L - horizon + 1):
        ctx = frames[t - K:t]
        future = frames[t:t + Hv]
        action = actions[t:t + Ha]
        out.append((ctx, future, action))
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class WindowDataset(Dataset):
    """Reads sharded windows from `<cfg.root>/<split>` into memory."""

    def __init__(self, cfg: DataConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        shard_dir = os.path.join(cfg.root, split)
        shards: List[str] = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not shards:
            raise FileNotFoundError(
                f"No shards in {shard_dir}. Run scripts/prepare_data.py first."
            )
        ctx, future, action = [], [], []
        for s in shards:
            d = np.load(s)
            ctx.append(d["ctx"]); future.append(d["future"]); action.append(d["action"])
        self.ctx = np.concatenate(ctx)
        self.future = np.concatenate(future)
        self.action = np.concatenate(action)

    def __len__(self) -> int:
        return len(self.ctx)

    def __getitem__(self, idx: int):
        return {
            "ctx_frames": torch.from_numpy(self.ctx[idx]).float() / 255.0,
            "future_frames": torch.from_numpy(self.future[idx]).float() / 255.0,
            "actions": torch.from_numpy(self.action[idx]).float(),
        }

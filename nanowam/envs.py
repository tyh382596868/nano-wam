"""Interactive environment for closed-loop evaluation (M4).

`ReacherEnv` is the step-able counterpart of the procedural generator in
data.py: a white agent square moves toward a gray goal. Actions use the same
normalized convention the dataset stores (displacement / speed, ~[-1, 1]), so a
policy trained on the offline windows can be rolled out here directly.
"""
from __future__ import annotations

import numpy as np

from .config import DataConfig
from .data import render_frame


class ReacherEnv:
    """Minimal reach-the-goal env. Frames are (3, H, W) float in [0, 1]."""

    def __init__(self, cfg: DataConfig, seed: int = 0):
        self.size = cfg.image_size
        self.speed = self.size / 16.0          # matches generate_episode
        self.success_radius = self.size / 10.0
        self.rng = np.random.default_rng(seed)
        self.agent = None
        self.goal = None

    def reset(self) -> np.ndarray:
        margin = self.size // 6
        self.agent = self.rng.uniform(margin, self.size - margin, size=2)
        self.goal = self.rng.uniform(margin, self.size - margin, size=2)
        return self._frame()

    def step(self, action: np.ndarray):
        """Apply a normalized action; return (frame, done, dist_to_goal)."""
        step = np.asarray(action, dtype=np.float64) * self.speed
        self.agent = np.clip(self.agent + step, 0, self.size - 1)
        dist = float(np.linalg.norm(self.agent - self.goal))
        return self._frame(), dist < self.success_radius, dist

    def _frame(self) -> np.ndarray:
        return render_frame(self.agent, self.goal, self.size).astype(np.float32) / 255.0

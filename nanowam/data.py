"""Toy windowed dataset.

Yields training windows from preprocessed episodes (written by
scripts/prepare_data.py):

    item = {
        "ctx_frames":    (K,  3, H, W) float in [0,1],
        "future_frames": (Hv, 3, H, W),
        "actions":       (Ha, Da),
        "goal":          (Lg,) long  | (d,) float   (depends on goal_embed),
    }

Default source is pushT (Da=2). An M5 LeRobot adapter can drop a LIBERO slice in
behind the same window interface. See DESIGN.md §7.

STATUS: stub. Bodies land in M2.
"""
from __future__ import annotations

from torch.utils.data import Dataset

from .config import DataConfig


class WindowDataset(Dataset):
    """Reads sharded windows from `cfg.root`; returns the dict above."""

    def __init__(self, cfg: DataConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        # TODO(M2): index the shards written by prepare_data.py.
        raise NotImplementedError("WindowDataset is a stub (M2).")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError

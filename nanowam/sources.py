"""Episode sources — the M5 real-data adapter layer.

Everything downstream (windowing, sharding, training) consumes *episodes*: a
pair (frames, actions) where
    frames  : (L, 3, H, W) uint8        — H = W = cfg.image_size
    actions : (L, Da)       float32
An `EpisodeSource.iter_episodes()` yields these one episode at a time, so the
same `make_windows` + shard writer (see scripts/prepare_data.py) handles the
procedural toy task and real robot datasets identically.

Adapters:
  - ReacherSource : the self-contained procedural task (default).
  - LeRobotSource : any dataset in the LeRobot format (pushT, LIBERO slices, ...).
    `lerobot` is imported lazily so the rest of nano-wam needs no heavy deps.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from .config import DataConfig
from .data import generate_episode

Episode = Tuple[np.ndarray, np.ndarray]  # (frames uint8 (L,3,H,W), actions f32 (L,Da))


def to_chw_uint8(img, size: int) -> np.ndarray:
    """Coerce one image (HWC/CHW, uint8 or float[0,1], np or torch) to (3,size,size) uint8."""
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(np.asarray(img))
    if t.ndim == 2:                       # grayscale -> 3ch
        t = t.unsqueeze(0).repeat(3, 1, 1)
    if t.shape[0] not in (1, 3):          # HWC -> CHW
        t = t.permute(2, 0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    t = t.float()
    if t.max() <= 1.0 + 1e-6:             # float [0,1] -> [0,255]
        t = t * 255.0
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear",
                          align_corners=False).squeeze(0)
    return t.clamp(0, 255).byte().cpu().numpy()


class EpisodeSource:
    """Base interface. Subclasses implement `iter_episodes`."""

    def iter_episodes(self) -> Iterator[Episode]:
        raise NotImplementedError


class ReacherSource(EpisodeSource):
    """Procedural reach-the-goal task (no external deps)."""

    def __init__(self, cfg: DataConfig, n_episodes: int, ep_len: int, seed: int):
        self.cfg = cfg
        self.n_episodes = n_episodes
        self.ep_len = ep_len
        self.seed = seed

    def iter_episodes(self) -> Iterator[Episode]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_episodes):
            yield generate_episode(rng, self.ep_len, self.cfg.image_size)


class LeRobotSource(EpisodeSource):
    """Adapter for datasets in the LeRobot format (pushT, LIBERO slices, ...).

    Frames are resized to cfg.image_size; the action vector is taken as-is (its
    width must equal cfg.action_dim — assert with a clear message). Episodes are
    delimited via the dataset's episode index. `lerobot` is imported lazily.
    """

    def __init__(self, cfg: DataConfig, repo_id: str, image_key: Optional[str] = None,
                 action_key: str = "action", max_episodes: Optional[int] = None):
        self.cfg = cfg
        self.repo_id = repo_id
        self.image_key = image_key
        self.action_key = action_key
        self.max_episodes = max_episodes

    def _load(self):
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "LeRobotSource needs the `lerobot` package: pip install lerobot"
            ) from e
        return LeRobotDataset(self.repo_id)

    def _episode_ranges(self, ds) -> Iterable[Tuple[int, int]]:
        """Yield (start, end) frame indices per episode, version-tolerant."""
        idx = getattr(ds, "episode_data_index", None)
        if idx is not None and "from" in idx and "to" in idx:
            starts = [int(x) for x in idx["from"]]
            ends = [int(x) for x in idx["to"]]
            return list(zip(starts, ends))
        # fallback: group consecutive rows by episode_index
        ranges, start, cur = [], 0, None
        for i in range(len(ds)):
            ep = int(ds[i]["episode_index"])
            if cur is None:
                cur = ep
            elif ep != cur:
                ranges.append((start, i)); start, cur = i, ep
        ranges.append((start, len(ds)))
        return ranges

    def _pick_image_key(self, sample) -> str:
        if self.image_key:
            return self.image_key
        for k in sample:
            if k.startswith("observation.image"):
                return k
        raise KeyError("no 'observation.image*' key found; pass --image-key explicitly")

    def iter_episodes(self) -> Iterator[Episode]:
        ds = self._load()
        size = self.cfg.image_size
        img_key = self._pick_image_key(ds[0])
        ranges = self._episode_ranges(ds)
        if self.max_episodes:
            ranges = list(ranges)[: self.max_episodes]

        for start, end in ranges:
            frames, actions = [], []
            for i in range(start, end):
                row = ds[i]
                frames.append(to_chw_uint8(row[img_key], size))
                a = np.asarray(row[self.action_key], dtype=np.float32).reshape(-1)
                assert a.shape[0] == self.cfg.action_dim, (
                    f"dataset action dim {a.shape[0]} != config action_dim "
                    f"{self.cfg.action_dim}; update configs"
                )
                actions.append(a)
            if len(frames) >= self.cfg.ctx_frames + max(self.cfg.video_horizon,
                                                        self.cfg.action_horizon):
                yield np.stack(frames), np.stack(actions)

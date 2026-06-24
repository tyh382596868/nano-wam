"""Prepare pushT (LeRobot v3.0 format) windows — no `lerobot` package needed.

The `lerobot` PyPI package can't be installed against this box's torch 2.3.1
(its torchvision pin conflicts), and its old `lerobot.common.*` reader can't
read v3.0 datasets anyway. This standalone script ingests the v3.0 layout
directly:

  - meta/info.json                      dataset spec (features, fps, paths)
  - data/chunk-*/file-*.parquet         tabular rows (action, episode_index, ...)
  - meta/episodes/chunk-*/file-*.parquet  per-episode global index ranges
  - videos/observation.image/.../*.mp4  AV1-encoded frames (decoded via PyAV)

Frames map 1:1 to parquet rows by global `index`; episode boundaries come from
the episodes-meta `dataset_from_index`/`dataset_to_index`. Output is byte-for-byte
the same shard format `scripts/prepare_data.py` writes, so WindowDataset, train,
sample and eval are unchanged.

Usage (route HF through the mirror if HF is blocked):
    HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_pusht_v3.py \
        --config configs/lerobot_pusht.yaml --normalize
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root
from nanowam.config import load_config
from nanowam.data import make_windows


def _resize_chw_uint8(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (3,size,size) uint8."""
    import cv2
    if frame_hwc.shape[0] != size or frame_hwc.shape[1] != size:
        frame_hwc = cv2.resize(frame_hwc, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame_hwc.transpose(2, 0, 1))


def decode_video(mp4_path: str, size: int) -> np.ndarray:
    """Decode an (AV1) mp4 to (N, 3, size, size) uint8 via PyAV (software dav1d)."""
    import av
    frames = []
    container = av.open(mp4_path)
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")  # (H,W,3) uint8
        frames.append(_resize_chw_uint8(rgb, size))
    container.close()
    return np.stack(frames)


def load_episodes(repo_id: str, cfg, max_episodes=None):
    """Yield (frames (L,3,H,W) uint8, actions (L,Da) f32) per episode."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    size = cfg.data.image_size
    dl = lambda f: hf_hub_download(repo_id, f, repo_type="dataset")

    df = pd.read_parquet(dl("data/chunk-000/file-000.parquet")).sort_values("index")
    actions = np.stack(df["action"].values).astype(np.float32)          # (N, Da)
    assert actions.shape[1] == cfg.data.action_dim, (
        f"dataset action dim {actions.shape[1]} != config action_dim {cfg.data.action_dim}")

    frames = decode_video(dl("videos/observation.image/chunk-000/file-000.mp4"), size)
    n = min(len(frames), len(actions))
    if len(frames) != len(actions):
        print(f"[pusht_v3] WARNING: {len(frames)} frames vs {len(actions)} rows; truncating to {n}")
    frames, actions = frames[:n], actions[:n]

    ep_meta = pd.read_parquet(dl("meta/episodes/chunk-000/file-000.parquet"))
    ranges = list(zip(ep_meta["dataset_from_index"].astype(int),
                      ep_meta["dataset_to_index"].astype(int)))
    if max_episodes:
        ranges = ranges[:max_episodes]
    print(f"[pusht_v3] {len(frames)} frames, {len(ranges)} episodes, "
          f"action range {actions.min(0)}..{actions.max(0)}")
    for start, end in ranges:
        end = min(end, n)
        if end - start >= cfg.data.ctx_frames + max(cfg.data.video_horizon, cfg.data.action_horizon):
            yield frames[start:end], actions[start:end]


def write_shards(windows, out_dir, mean, std, shard_size):
    os.makedirs(out_dir, exist_ok=True)
    n_shards = 0
    for i in range(0, len(windows), shard_size):
        chunk = windows[i:i + shard_size]
        ctx = np.stack([w[0] for w in chunk])
        future = np.stack([w[1] for w in chunk])
        action = np.stack([w[2] for w in chunk]).astype(np.float32)
        if mean is not None:
            action = (action - mean) / std
        np.savez_compressed(os.path.join(out_dir, f"shard_{n_shards:03d}.npz"),
                            ctx=ctx, future=future, action=action)
        n_shards += 1
    print(f"[pusht_v3]   wrote {n_shards} shard(s) to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare pushT (LeRobot v3.0) windows.")
    p.add_argument("--config", required=True)
    p.add_argument("--repo-id", default="lerobot/pusht")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--normalize", action="store_true", help="normalize actions; save stats.npz")
    p.add_argument("--shard-size", type=int, default=2048)
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"[pusht_v3] repo={args.repo_id} image={cfg.data.image_size} "
          f"window=ctx{cfg.data.ctx_frames}+future{cfg.data.video_horizon}, "
          f"actions {cfg.data.action_horizon}x{cfg.data.action_dim}")

    windows = []
    n_ep = 0
    for frames, actions in load_episodes(args.repo_id, cfg, args.max_episodes):
        windows.extend(make_windows(frames, actions, cfg.data.ctx_frames,
                                    cfg.data.video_horizon, cfg.data.action_horizon))
        n_ep += 1
    print(f"[pusht_v3] {n_ep} episode(s) -> {len(windows)} windows")
    if not windows:
        raise SystemExit("no windows produced (episodes too short for the horizons?)")

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(windows)
    n_val = max(1, int(len(windows) * args.val_frac))
    val, train = windows[:n_val], windows[n_val:]

    mean = std = None
    if args.normalize:
        acts = np.concatenate([w[2].reshape(-1, cfg.data.action_dim) for w in train])
        mean = acts.mean(0, keepdims=True).astype(np.float32)
        std = (acts.std(0, keepdims=True) + 1e-6).astype(np.float32)
        os.makedirs(cfg.data.root, exist_ok=True)
        np.savez(os.path.join(cfg.data.root, "stats.npz"), action_mean=mean, action_std=std)
        print(f"[pusht_v3] saved action stats: mean={mean.ravel()} std={std.ravel()}")

    write_shards(train, os.path.join(cfg.data.root, "train"), mean, std, args.shard_size)
    write_shards(val, os.path.join(cfg.data.root, "val"), mean, std, args.shard_size)


if __name__ == "__main__":
    main()

"""Generate sharded training windows for WindowDataset, from any EpisodeSource.

Sources (see nanowam/sources.py):
  --source reacher   self-contained procedural task (default; no deps)
  --source lerobot   any LeRobot-format dataset (pushT, LIBERO slices, ...),
                     selected with --repo-id (needs `pip install lerobot`)

Windows from all episodes are pooled, shuffled, and split into train/val. With
--normalize, per-dimension action mean/std are computed on train and applied to
both splits; the stats are saved to <root>/stats.npz for deployment de-norm.

Usage:
    python scripts/prepare_data.py --config configs/pusht_tiny.yaml --episodes 200
    python scripts/prepare_data.py --config configs/lerobot_pusht.yaml \
        --source lerobot --repo-id lerobot/pusht --normalize
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config
from nanowam.data import make_windows
from nanowam.sources import LeRobotSource, ReacherSource


def build_source(args, cfg):
    if args.source == "reacher":
        return ReacherSource(cfg.data, args.episodes, args.ep_len, seed=cfg.seed)
    if args.source == "lerobot":
        if not args.repo_id:
            raise SystemExit("--source lerobot requires --repo-id")
        return LeRobotSource(cfg.data, args.repo_id, image_key=args.image_key,
                             action_key=args.action_key, max_episodes=args.max_episodes)
    raise SystemExit(f"unknown source {args.source}")


def collect_windows(source, cfg):
    windows = []
    n_ep = 0
    for frames, actions in source.iter_episodes():
        windows.extend(make_windows(
            frames, actions, cfg.data.ctx_frames, cfg.data.video_horizon, cfg.data.action_horizon))
        n_ep += 1
    print(f"[prepare_data] {n_ep} episode(s) -> {len(windows)} windows")
    return windows


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
    print(f"[prepare_data]   wrote {n_shards} shard(s) to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build nano-wam training windows.")
    p.add_argument("--config", required=True)
    p.add_argument("--source", choices=["reacher", "lerobot"], default="reacher")
    p.add_argument("--episodes", type=int, default=200, help="reacher: episodes to generate")
    p.add_argument("--ep-len", type=int, default=24, help="reacher: steps per episode")
    p.add_argument("--repo-id", default=None, help="lerobot: HF dataset id")
    p.add_argument("--image-key", default=None, help="lerobot: observation image key")
    p.add_argument("--action-key", default="action", help="lerobot: action key")
    p.add_argument("--max-episodes", type=int, default=None, help="lerobot: cap episodes")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--normalize", action="store_true", help="normalize actions; save stats.npz")
    p.add_argument("--shard-size", type=int, default=2048)
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"[prepare_data] source='{args.source}' image={cfg.data.image_size} "
          f"window=ctx{cfg.data.ctx_frames}+future{cfg.data.video_horizon}, "
          f"actions {cfg.data.action_horizon}x{cfg.data.action_dim}")

    windows = collect_windows(build_source(args, cfg), cfg)
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
        print(f"[prepare_data] saved action stats: mean={mean.ravel()} std={std.ravel()}")

    write_shards(train, os.path.join(cfg.data.root, "train"), mean, std, args.shard_size)
    write_shards(val, os.path.join(cfg.data.root, "val"), mean, std, args.shard_size)


if __name__ == "__main__":
    main()

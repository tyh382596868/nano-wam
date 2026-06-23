"""Generate sharded training windows for WindowDataset.

Default source is the self-contained procedural "reacher" task (see
nanowam/data.py). The pushT / LeRobot adapters (DESIGN.md §7) will write the same
shard format at M5.

Usage:
    python scripts/prepare_data.py --config configs/pusht_tiny.yaml
    python scripts/prepare_data.py --config configs/pusht_tiny.yaml --episodes 200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config
from nanowam.data import generate_episode, make_windows


def build_split(cfg, split, n_episodes, ep_len, seed, shard_size):
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(cfg.data.root, split)
    os.makedirs(out_dir, exist_ok=True)

    windows = []
    for _ in range(n_episodes):
        frames, actions = generate_episode(rng, ep_len, cfg.data.image_size)
        windows.extend(make_windows(
            frames, actions, cfg.data.ctx_frames, cfg.data.video_horizon, cfg.data.action_horizon
        ))
    rng.shuffle(windows)

    n_shards = 0
    for i in range(0, len(windows), shard_size):
        chunk = windows[i:i + shard_size]
        ctx = np.stack([w[0] for w in chunk])
        future = np.stack([w[1] for w in chunk])
        action = np.stack([w[2] for w in chunk]).astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, f"shard_{n_shards:03d}.npz"),
                            ctx=ctx, future=future, action=action)
        n_shards += 1
    print(f"[prepare_data] {split}: {len(windows)} windows -> {n_shards} shard(s) in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nano-wam training windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=200, help="train episodes")
    parser.add_argument("--ep-len", type=int, default=24, help="steps per episode")
    parser.add_argument("--shard-size", type=int, default=2048)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[prepare_data] source='{cfg.data.name}' image={cfg.data.image_size} "
          f"window=ctx{cfg.data.ctx_frames}+future{cfg.data.video_horizon}, "
          f"actions {cfg.data.action_horizon}x{cfg.data.action_dim}")
    build_split(cfg, "train", args.episodes, args.ep_len, seed=cfg.seed, shard_size=args.shard_size)
    build_split(cfg, "val", max(8, args.episodes // 10), args.ep_len, seed=cfg.seed + 1,
                shard_size=args.shard_size)


if __name__ == "__main__":
    main()

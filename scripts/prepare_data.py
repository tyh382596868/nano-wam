"""Convert episodes into sharded training windows for WindowDataset.

Reads an episode source (default: pushT expert demos; M5: a small LeRobot/LIBERO
slice) and slides a window of (ctx_frames + video_horizon) frames with an
aligned action_horizon, writing shards under data.root. See DESIGN.md §7.

STATUS: stub. Bodies land in M2.

Usage:
    python scripts/prepare_data.py --config configs/pusht_tiny.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nano-wam training windows.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[prepare_data] would build windows for '{cfg.data.name}' into {cfg.data.root}")
    print(f"  window = ctx {cfg.data.ctx_frames} + future {cfg.data.video_horizon} frames, "
          f"actions {cfg.data.action_horizon}x{cfg.data.action_dim}")
    raise NotImplementedError("prepare_data is a stub (M2).")


if __name__ == "__main__":
    main()

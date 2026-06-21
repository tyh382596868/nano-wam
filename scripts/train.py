"""Train nano-wam.

Loop sketch (see DESIGN.md §5):
    for step in range(max_steps):
        batch = next(loader)                      # tokenize frames -> latents
        mode  = sample_mode(cfg.train.mode_probs) # UniDiffuser-style
        mask  = MODE_TABLE[mode]
        loss  = flow_loss(model, batch, mask, cfg.flow)
        loss.backward(); clip; opt.step()
        log / checkpoint

STATUS: stub. Bodies land in M2.

Usage:
    python scripts/train.py --config configs/pusht_tiny.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nano-wam.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[train] config loaded: out_dir={cfg.train.out_dir} "
          f"dim={cfg.model.dim} depth={cfg.model.depth} max_steps={cfg.train.max_steps}")
    # TODO(M2): set_seed -> WindowDataset/DataLoader -> FrameTokenizer + NanoWAM
    #           -> AdamW + warmup -> training loop with flow_loss -> ckpt.
    raise NotImplementedError("train loop is a stub (M2).")


if __name__ == "__main__":
    main()

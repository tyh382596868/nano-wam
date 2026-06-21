"""Evaluate / roll out nano-wam, with the FastWAM imagination toggle.

  --imagine none   : policy mode, action stream only (no video denoising)
  --imagine joint  : denoise video + action together; also dumps predicted frames

Deployment uses receding horizon: sample an action chunk, execute the first
`eval.replan_every` actions, then replan. See DESIGN.md §6.

STATUS: stub. Bodies land in M3/M4.

Usage:
    python scripts/sample.py --config configs/pusht_tiny.yaml --imagine none
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample / evaluate nano-wam.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None, help="checkpoint to load")
    parser.add_argument("--imagine", choices=["none", "joint"], default=None,
                        help="override eval.imagine (the FastWAM ablation toggle)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    imagine = args.imagine or cfg.eval.imagine
    print(f"[sample] episodes={cfg.eval.episodes} imagine={imagine} "
          f"steps={cfg.flow.sample_steps} replan_every={cfg.eval.replan_every}")
    # TODO(M3/M4): load ckpt -> build env -> receding-horizon rollout via
    #              flow.sample under POLICY (imagine=none) or JOINT (imagine=joint)
    #              -> report success rate; with imagine=joint dump predicted frames.
    raise NotImplementedError("sample/eval is a stub (M3/M4).")


if __name__ == "__main__":
    main()

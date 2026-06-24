"""Closed-loop pushT evaluation in gym-pusht (the M-real sim test).

Loads a checkpoint trained on the pushT LeRobot dataset and rolls it out as a
receding-horizon policy in the gym-pusht simulator, running the FastWAM
imagination on/off ablation (policy vs joint). Reports success rate, mean
coverage, and mean steps-to-success.

Usage:
    MUJOCO_GL=egl python scripts/eval_pusht.py --config configs/lerobot_pusht.yaml --episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root
from nanowam.config import load_config
from nanowam.model import NanoWAM
from nanowam.tokenizer import FrameTokenizer
from nanowam.pusht_eval import run_pusht_ablation
from nanowam.utils import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop pushT eval in gym-pusht.")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None, help="default: <out_dir>/ckpt_final.pt")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    tokenizer = FrameTokenizer(cfg.model, cfg.data).to(device).eval()
    model = NanoWAM(cfg.model, cfg.data).to(device).eval()
    ckpt = args.ckpt or f"{cfg.train.out_dir}/ckpt_final.pt"
    if os.path.exists(ckpt):
        step = load_checkpoint(ckpt, model, tokenizer)
        print(f"[eval_pusht] loaded {ckpt} (step {step})")
    else:
        print(f"[eval_pusht] WARNING: no checkpoint at {ckpt}; using random weights")

    res = run_pusht_ablation(model, tokenizer, cfg, device,
                             episodes=args.episodes, max_steps=args.max_steps)
    n = res["none"]["n"]
    print(f"\n[eval_pusht] gym-pusht closed-loop imagination ablation ({n} episodes)")
    print(f"{'setting':<14}{'mode':<10}{'success':>10}{'mean_cov':>10}{'mean_steps':>12}")
    for setting, mode_name in (("none", "policy"), ("joint", "joint")):
        r = res[setting]
        print(f"imagine={setting:<6}{mode_name:<10}{r['success_rate']*100:>9.1f}%"
              f"{r['mean_coverage']:>10.3f}{r['mean_steps']:>12.1f}")


if __name__ == "__main__":
    main()

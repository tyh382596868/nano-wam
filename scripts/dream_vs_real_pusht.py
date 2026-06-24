"""Closed-loop pushT: model's imagined future vs the actually-executed sim frames.

Rolls out one joint-mode episode in gym-pusht. At each replan step the model
denoises BOTH the future-frame (world) stream and the action stream; we decode
the imagined frames, execute the actions in the sim, and render what actually
happened. Saves a side-by-side gif/grid (top = imagined, bottom = executed).

Usage:
    python scripts/dream_vs_real_pusht.py --config configs/lerobot_pusht.yaml \
        --seed 10000 --max-steps 80
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanowam.config import load_config
from nanowam.flow import sample
from nanowam.model import NanoWAM
from nanowam.modes import Mode
from nanowam.pusht_eval import _frame_from_obs, load_action_stats
from nanowam.tokenizer import FrameTokenizer, to_tokens, from_tokens
from nanowam.utils import load_checkpoint


def to_uint8(frame_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) float[0,1] -> (H,W,3) uint8."""
    x = np.clip(frame_chw, 0, 1) * 255.0
    return np.moveaxis(x.astype(np.uint8), 0, -1)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    K = cfg.data.ctx_frames
    Hv = cfg.data.video_horizon
    replan = cfg.eval.replan_every
    size = cfg.data.image_size
    h = size // cfg.model.patch_size

    tokenizer = FrameTokenizer(cfg.model, cfg.data).to(device).eval()
    model = NanoWAM(cfg.model, cfg.data).to(device).eval()
    ckpt = args.ckpt or f"{cfg.train.out_dir}/ckpt_final.pt"
    step = load_checkpoint(ckpt, model, tokenizer)
    print(f"[dream_vs_real] loaded {ckpt} (step {step})")

    mean, std = load_action_stats(cfg)

    import gymnasium as gym
    import gym_pusht  # noqa: F401
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", observation_width=96, observation_height=96)
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)

    obs, info = env.reset(seed=args.seed)
    buf = [_frame_from_obs(obs["pixels"], size)] * K

    imagined, executed = [], []
    steps = 0
    best_cov = float(info.get("coverage", 0.0))
    while steps < args.max_steps:
        ctx = torch.from_numpy(np.stack(buf[-K:])).float().unsqueeze(0).to(device)
        ctx_tok = to_tokens(tokenizer.encode(ctx))
        out = sample(model, {"ctx": ctx_tok}, Mode.JOINT, cfg.flow)
        pred = tokenizer.decode(from_tokens(out["video"], Hv, h, h))[0].cpu().numpy()  # (Hv,3,H,W)
        actions = out["action"][0].cpu().numpy() * std + mean
        actions = np.clip(actions, low, high)

        for j in range(min(replan, Hv, len(actions))):
            obs, rew, term, trunc, info = env.step(actions[j].astype(np.float32))
            real = _frame_from_obs(obs["pixels"], size)
            buf.append(real)
            imagined.append(to_uint8(pred[j]))   # imagined future frame j
            executed.append(to_uint8(real))      # what actually happened at step j
            best_cov = max(best_cov, float(info.get("coverage", 0.0)))
            steps += 1
            done = bool(info.get("is_success", False)) or term
            if done or trunc or steps >= args.max_steps:
                break
        if done or trunc:
            break
    env.close()
    print(f"[dream_vs_real] {steps} steps, best_coverage={best_cov:.3f}, success={done}")

    # stitch: gif (top=imagined, bottom=executed), and a strip grid every few frames
    H, W = size, size
    vsep = np.full((3, W, 3), 80, np.uint8)
    gif = [np.concatenate([imagined[t], vsep, executed[t]], axis=0) for t in range(len(imagined))]

    stride = max(1, len(imagined) // 12)
    idxs = list(range(0, len(imagined), stride))
    hsep = np.full((H, 2, 3), 80, np.uint8)
    top = []
    bot = []
    for t in idxs:
        top.append(imagined[t]); top.append(hsep)
        bot.append(executed[t]); bot.append(hsep)
    grid = np.concatenate([np.concatenate(top, axis=1),
                           np.full((6, np.concatenate(top, axis=1).shape[1], 3), 80, np.uint8),
                           np.concatenate(bot, axis=1)], axis=0)

    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)
    import imageio.v2 as imageio
    gif_path = os.path.join(out_dir, f"dream_vs_real_seed{args.seed}.gif")
    grid_path = os.path.join(out_dir, f"dream_vs_real_seed{args.seed}_grid.png")
    imageio.mimsave(gif_path, gif, duration=0.25, loop=0)
    imageio.imwrite(grid_path, grid)
    print(f"[dream_vs_real] wrote {gif_path} and {grid_path}")
    print("[dream_vs_real] layout: TOP = model-imagined future, BOTTOM = actual sim execution")


if __name__ == "__main__":
    main()

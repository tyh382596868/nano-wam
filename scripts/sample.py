"""Sample / visualize nano-wam rollouts.

M3 scope: qualitative generation. Given context frames from the val set, denoise
future-frame latents (and/or actions) and decode them back to pixels, then dump a
side-by-side GT-vs-prediction gif + png grid and print simple metrics.

Modes:
  --mode world   condition on ctx + ground-truth actions, predict future frames
  --mode joint   condition on ctx only, predict future frames AND actions
  --mode policy  predict the action chunk only (no imagination)

The FastWAM-style `--imagine {none,joint}` toggle maps to policy / joint and is
wired for the M4 closed-loop eval (success rate); here it just selects what to
generate. See DESIGN.md §6.

Usage:
    python scripts/sample.py --config configs/pusht_tiny.yaml --mode world
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config
from nanowam.data import WindowDataset
from nanowam.flow import sample
from nanowam.model import NanoWAM
from nanowam.modes import Mode
from nanowam.tokenizer import FrameTokenizer, to_tokens, from_tokens
from nanowam.utils import load_checkpoint

IMAGINE_TO_MODE = {"none": "policy", "joint": "joint"}


def decode_video_tokens(tokenizer, tokens, T, h, w):
    """(B, T*h*w, C) latent tokens -> (B, T, 3, H, W) frames in [0, 1]."""
    return tokenizer.decode(from_tokens(tokens, T, h, w))


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(.., 3, H, W) float[0,1] -> (.., H, W, 3) uint8."""
    x = frames.clamp(0, 1).mul(255).byte().cpu().numpy()
    return np.moveaxis(x, -3, -1)


def save_comparison(gt: torch.Tensor, pred: torch.Tensor, out_dir: str, tag: str):
    """gt/pred: (N, Hv, 3, H, W). Writes a png grid and an animated gif."""
    os.makedirs(out_dir, exist_ok=True)
    g, p = to_uint8(gt), to_uint8(pred)            # (N, Hv, H, W, 3)
    N, Hv, H, W, _ = g.shape
    sep = np.full((H, 2, 3), 80, dtype=np.uint8)   # gray separator

    # png grid: rows = samples; each row = [gt frames | sep | pred frames]
    rows = []
    for n in range(N):
        gt_row = np.concatenate(list(g[n]), axis=1)
        pred_row = np.concatenate(list(p[n]), axis=1)
        rows.append(np.concatenate([gt_row, sep, pred_row], axis=1))
    grid = np.concatenate(rows, axis=0)

    # gif: per future timestep, stack samples vertically, gt-above-pred
    gif_frames = []
    vsep = np.full((2, W, 3), 80, dtype=np.uint8)
    for t in range(Hv):
        cols = [np.concatenate([g[n, t], vsep, p[n, t]], axis=0) for n in range(N)]
        gif_frames.append(np.concatenate(cols, axis=1))

    png_path = os.path.join(out_dir, f"{tag}_grid.png")
    gif_path = os.path.join(out_dir, f"{tag}_rollout.gif")
    try:
        import imageio.v2 as imageio
        imageio.imwrite(png_path, grid)
        imageio.mimsave(gif_path, gif_frames, duration=0.3, loop=0)
        print(f"[sample] wrote {png_path} and {gif_path}  (top=GT, bottom=pred)")
    except Exception as e:  # imageio optional; still report we got here
        print(f"[sample] could not write images ({e}); grid shape {grid.shape}")
    return png_path, gif_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample / visualize nano-wam.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None, help="checkpoint (default: <out_dir>/ckpt_final.pt)")
    parser.add_argument("--mode", choices=["world", "joint", "policy"], default="world")
    parser.add_argument("--imagine", choices=["none", "joint"], default=None,
                        help="FastWAM toggle; overrides --mode (none->policy, joint->joint)")
    parser.add_argument("--n", type=int, default=4, help="number of samples to visualize")
    parser.add_argument("--closed-loop", action="store_true",
                        help="run the M4 closed-loop imagination ablation instead of open-loop viz")
    parser.add_argument("--dream", type=int, default=0, metavar="N",
                        help="causal model only: autoregressively dream N blocks into a long gif")
    parser.add_argument("--episodes", type=int, default=None, help="episodes for --closed-loop")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode_name = IMAGINE_TO_MODE[args.imagine] if args.imagine else args.mode
    mode = Mode[mode_name.upper()]
    device = torch.device(args.device)

    tokenizer = FrameTokenizer(cfg.model, cfg.data).to(device).eval()
    model = NanoWAM(cfg.model, cfg.data).to(device).eval()
    ckpt = args.ckpt or f"{cfg.train.out_dir}/ckpt_final.pt"
    if os.path.exists(ckpt):
        step = load_checkpoint(ckpt, model, tokenizer)
        print(f"[sample] loaded {ckpt} (step {step})")
    else:
        print(f"[sample] WARNING: no checkpoint at {ckpt}; using random weights")

    if args.dream:
        from nanowam.rollout import dream_rollout
        ds = WindowDataset(cfg.data, "val")
        init = ds[0]["ctx_frames"].to(device)               # (K, 3, H, W)
        frames = dream_rollout(model, tokenizer, init, args.dream, cfg, Mode.JOINT, device)
        print(f"[sample] dreamed {args.dream} blocks -> {frames.shape[0]} frames")
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        try:
            import imageio.v2 as imageio
            gif = os.path.join(cfg.train.out_dir, "dream_long.gif")
            imageio.mimsave(gif, list(to_uint8(frames)), duration=0.2, loop=0)
            print(f"[sample] wrote {gif} ({frames.shape[0]} frames)")
        except Exception as e:
            print(f"[sample] could not write dream gif ({e})")
        return

    if args.closed_loop:
        from nanowam.eval import run_ablation
        res = run_ablation(model, tokenizer, cfg, device, episodes=args.episodes)
        print(f"[eval] closed-loop imagination ablation ({res['none']['n']} episodes)")
        print(f"{'setting':<14}{'mode':<10}{'success':>10}{'mean_steps':>12}")
        for setting, mode_name in (("none", "policy"), ("joint", "joint")):
            r = res[setting]
            print(f"imagine={setting:<6}{mode_name:<10}{r['success_rate']*100:>9.1f}%{r['mean_steps']:>12.1f}")
        return

    ds = WindowDataset(cfg.data, "val")
    idxs = list(range(min(args.n, len(ds))))
    ctx = torch.stack([ds[i]["ctx_frames"] for i in idxs]).to(device)
    fut = torch.stack([ds[i]["future_frames"] for i in idxs]).to(device)
    act = torch.stack([ds[i]["actions"] for i in idxs]).to(device)
    h = cfg.data.image_size // cfg.model.patch_size

    with torch.no_grad():
        ctx_tok = to_tokens(tokenizer.encode(ctx))
        cond = {"ctx": ctx_tok}
        if mode == Mode.WORLD:
            cond["action"] = act
        out = sample(model, cond, mode, cfg.flow)

        print(f"[sample] mode={mode_name} steps={cfg.flow.sample_steps} n={len(idxs)}")
        if "action" in out:
            a_mse = torch.mean((out["action"] - act) ** 2).item()
            print(f"[sample] action MSE vs GT: {a_mse:.4f}")
        if "video" in out:
            pred = decode_video_tokens(tokenizer, out["video"], cfg.data.video_horizon, h, h)
            v_mse = torch.mean((pred - fut) ** 2).item()
            # trivial baseline: repeat the last context frame across the horizon
            base = ctx[:, -1:].expand_as(fut)
            b_mse = torch.mean((base - fut) ** 2).item()
            print(f"[sample] future-frame MSE: model {v_mse:.4f}  vs  copy-last-frame {b_mse:.4f}")
            save_comparison(fut, pred, cfg.train.out_dir, f"{mode_name}")


if __name__ == "__main__":
    main()

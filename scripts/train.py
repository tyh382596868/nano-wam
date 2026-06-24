"""Train nano-wam.

The tokenizer is trained by reconstruction; the WAM is trained by rectified-flow
matching on the (detached) latents — the standard latent-diffusion split, so the
flow objective can't collapse the encoder. A UniDiffuser-style per-batch mode
sample lets one checkpoint serve policy / world / joint / inverse. See DESIGN.md
§5.

Checkpointing (M7): with train.max_ckpts > 0 the interval checkpoints are capped
at that many, keeping the ones that score best in the closed-loop sim (plus the
latest, for preemption-safe resume). With train.eval_every > 0 the sim eval runs
every that-many steps and its score ranks the checkpoints; see
nanowam/ckpt_manager.py. ckpt_best.pt always points at the best so far.

Usage:
    python scripts/train.py --config configs/pusht_tiny.yaml
    python scripts/train.py --config configs/pusht_dlc.yaml --resume runs/pusht_dlc/ckpt_020000.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root, no install
from nanowam.config import load_config
from nanowam.ckpt_manager import CheckpointManager, score_checkpoint
from nanowam.data import WindowDataset
from nanowam.flow import flow_loss
from nanowam.model import NanoWAM
from nanowam.modes import sample_mode
from nanowam.tokenizer import FrameTokenizer, to_tokens
from nanowam.utils import load_checkpoint, save_checkpoint, set_seed


def lr_scale(step: int, warmup: int) -> float:
    return min(1.0, (step + 1) / max(1, warmup))


def encode_to_tokens(tokenizer, frames):
    """frames (B,T,3,H,W) -> (recon_decoded, latent_tokens (B, T*P, C))."""
    z = tokenizer.encode(frames)            # (B,T,C,h,w)
    recon = tokenizer.decode(z)             # (B,T,3,H,W)
    return recon, to_tokens(z)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nano-wam.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device(args.device)
    print(f"[train] device={device} out_dir={cfg.train.out_dir} "
          f"dim={cfg.model.dim} depth={cfg.model.depth} max_steps={cfg.train.max_steps}")

    ds = WindowDataset(cfg.data, "train")
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                        num_workers=2, persistent_workers=True)
    print(f"[train] {len(ds)} windows")

    tokenizer = FrameTokenizer(cfg.model, cfg.data).to(device)
    model = NanoWAM(cfg.model, cfg.data).to(device)
    print(f"[train] params: tokenizer={sum(p.numel() for p in tokenizer.parameters())/1e6:.2f}M "
          f"wam={model.num_params()/1e6:.2f}M")

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(tokenizer.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_scale(s, cfg.train.warmup_steps))

    start = 0
    if args.resume:
        start = load_checkpoint(args.resume, model, tokenizer, opt)
        for _ in range(start):
            sched.step()
        print(f"[train] resumed from step {start}")

    ckpt_mgr = CheckpointManager(cfg.train.out_dir, cfg.train.max_ckpts)
    ckpt_mgr.restore()

    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def maybe_eval_and_track(step: int) -> None:
        path = f"{cfg.train.out_dir}/ckpt_{step:06d}.pt"
        save_checkpoint(path, model, tokenizer, opt, step)
        score, extra = float("-inf"), None
        if cfg.train.eval_every and step % cfg.train.eval_every == 0:
            try:
                score, r = score_checkpoint(model, tokenizer, cfg, device,
                                            cfg.train.eval_episodes, cfg.train.eval_max_steps)
                extra = r
                print(f"[eval] step {step:6d} score {score:.4f} "
                      f"success {r['success_rate']*100:.1f}% "
                      f"cov {r.get('mean_coverage', float('nan')):.3f} "
                      f"steps {r.get('mean_steps', float('nan')):.1f}")
            except Exception as e:  # eval must never kill a long training run
                print(f"[eval] step {step:6d} SKIPPED ({type(e).__name__}: {e})")
            finally:
                model.train(); tokenizer.train()
        ckpt_mgr.add(step, path, score, extra)
        print(f"[train] saved {path} (kept {len(ckpt_mgr.entries)}, best {ckpt_mgr.best_score:.4f})")

    model.train(); tokenizer.train()
    step = start
    data_iter = iter(loader)
    while step < cfg.train.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        ctx_frames = batch["ctx_frames"].to(device)
        future_frames = batch["future_frames"].to(device)
        actions = batch["actions"].to(device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            ctx_recon, ctx_tok = encode_to_tokens(tokenizer, ctx_frames)
            fut_recon, fut_tok = encode_to_tokens(tokenizer, future_frames)
            recon_loss = F.mse_loss(ctx_recon, ctx_frames) + F.mse_loss(fut_recon, future_frames)

            lat = {"ctx": ctx_tok.detach(), "video": fut_tok.detach(), "action": actions}
            mode = sample_mode(cfg.train.mode_probs)
            f_loss, metrics = flow_loss(model, lat, mode, cfg.flow)
            loss = f_loss + recon_loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(tokenizer.parameters()), cfg.train.grad_clip)
        scaler.step(opt); scaler.update(); sched.step()

        if step % cfg.train.log_every == 0:
            print(f"[train] step {step:6d} loss {loss.item():.4f} "
                  f"flow {metrics['loss'].item():.4f} recon {recon_loss.item():.4f} "
                  f"mode {mode.name.lower()}")
        if step > start and step % cfg.train.ckpt_every == 0:
            maybe_eval_and_track(step)
        step += 1

    save_checkpoint(f"{cfg.train.out_dir}/ckpt_final.pt", model, tokenizer, opt, step)
    print("[train] done")


if __name__ == "__main__":
    main()

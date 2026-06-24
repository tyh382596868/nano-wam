"""Autoregressive long-horizon rollout (the LingBot-VA path, nano edition).

A causal WAM predicts one block of `video_horizon` future frames; to dream past
that horizon we roll out **block-autoregressively**: predict a block, decode it,
slide the context window onto the last K predicted frames, and predict the next
block — for as many blocks as you like. Each block uses the KV-cache fast path
(`flow.sample(..., use_cache=True)`), so the clean context's per-layer K/V are
computed once per block instead of once per Euler step.

`dream_rollout` returns the full (n_blocks * Hv, 3, H, W) frame sequence. With
mode=JOINT the model also imagines the actions, so the dream is autonomous (no
external action sequence needed).
"""
from __future__ import annotations

import torch

from .config import Config
from .flow import sample
from .modes import Mode
from .tokenizer import to_tokens, from_tokens


@torch.no_grad()
def dream_rollout(model, tokenizer, init_ctx_frames: torch.Tensor, n_blocks: int,
                  cfg: Config, mode: Mode = Mode.JOINT, device="cpu") -> torch.Tensor:
    """Roll out n_blocks * video_horizon frames autoregressively.

    init_ctx_frames: (K, 3, H, W) in [0, 1]. Returns (n_blocks*Hv, 3, H, W).
    """
    assert getattr(model, "causal", False), "dream_rollout needs a causal model"
    model.eval(); tokenizer.eval()
    K = cfg.data.ctx_frames
    Hv = cfg.data.video_horizon
    h = cfg.data.image_size // cfg.model.patch_size

    ctx = init_ctx_frames.to(device).unsqueeze(0)  # (1, K, 3, H, W)
    frames_out = []
    for _ in range(n_blocks):
        ctx_tok = to_tokens(tokenizer.encode(ctx))
        out = sample(model, {"ctx": ctx_tok}, mode, cfg.flow, use_cache=True)
        block = tokenizer.decode(from_tokens(out["video"], Hv, h, h))[0]  # (Hv, 3, H, W)
        block = block.clamp(0, 1)
        frames_out.append(block)
        # slide the context window onto the last K predicted frames
        ctx = block[-K:].unsqueeze(0)

    return torch.cat(frames_out, dim=0)

"""Rectified flow matching — noising, loss, and ODE sampler.

Convention (data at tau=1, noise at tau=0):
    x_tau = (1 - tau) * eps + tau * x1      eps ~ N(0, I)
    target velocity  u = x1 - eps
The network predicts v_theta; loss is MSE to u on the noised+supervised streams
only (per the active Mode). Inference integrates dx/dtau = v_theta from 0 -> 1.

Per-mode stream handling (see DESIGN.md §5). The boolean mask says which streams
are noised+supervised; the *conditioning* content of an un-noised stream depends
on the mode:
    - WORLD   : action stream fed clean (the action we condition the world on)
    - INVERSE : video stream fed clean (the observed future frames)
    - POLICY  : video stream fed as zeros (no future available at deploy time)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import FlowConfig
from .modes import Mode, MODE_TABLE


def interpolate(x1: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor):
    """Form x_tau = (1-tau)*eps + tau*x1 and target velocity u = x1 - eps.

    `tau` is (B,); it is broadcast over the remaining dims of x1.
    """
    view = (-1,) + (1,) * (x1.dim() - 1)
    t = tau.view(view)
    x_tau = (1 - t) * eps + t * x1
    u = x1 - eps
    return x_tau, u


def _zeros_like(x):
    return torch.zeros_like(x)


def build_stream_inputs(batch: dict, mode: Mode, tau: torch.Tensor, generator=None):
    """Construct (video_in, action_in) plus targets/supervision flags for a mode.

    `batch` holds *clean* latents: 'video' (B, Hv*P, C), 'action' (B, Ha, Da).
    Returns dict with model inputs and per-stream (target, supervised) info.
    """
    mask = MODE_TABLE[mode]
    out = {}

    # video stream
    if mask.noise_video:
        eps = torch.randn(batch["video"].shape, device=batch["video"].device, generator=generator)
        x_v, u_v = interpolate(batch["video"], eps, tau)
        out["video_in"], out["u_video"], out["sup_video"] = x_v, u_v, True
    else:
        cond_v = batch["video"] if mode == Mode.INVERSE else _zeros_like(batch["video"])
        out["video_in"], out["u_video"], out["sup_video"] = cond_v, None, False

    # action stream
    if mask.noise_action:
        eps = torch.randn(batch["action"].shape, device=batch["action"].device, generator=generator)
        x_a, u_a = interpolate(batch["action"], eps, tau)
        out["action_in"], out["u_action"], out["sup_action"] = x_a, u_a, True
    else:
        out["action_in"], out["u_action"], out["sup_action"] = batch["action"], None, False

    return out


def flow_loss(
    model,
    batch: dict,
    mode: Mode,
    cfg: FlowConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, dict]:
    """Joint rectified-flow loss for one batch under a Mode.

    `batch` keys: 'ctx' (B, K*P, C), 'video' (B, Hv*P, C), 'action' (B, Ha, Da),
    optional 'goal' (B, d). Only the streams noised by the mode are supervised.
    Returns (scalar_loss, metrics).
    """
    device = batch["ctx"].device
    B = batch["ctx"].shape[0]
    tau = torch.rand(B, device=device, generator=generator)
    s = build_stream_inputs(batch, mode, tau, generator)

    mode_id = torch.full((B,), int(mode), device=device, dtype=torch.long)
    v_video, v_action = model(
        batch["ctx"], s["video_in"], s["action_in"], tau, mode_id, batch.get("goal")
    )

    loss = batch["ctx"].new_zeros(())
    metrics = {}
    if s["sup_video"]:
        lv = F.mse_loss(v_video, s["u_video"])
        loss = loss + cfg.loss_video_weight * lv
        metrics["loss_video"] = lv.detach()
    if s["sup_action"]:
        la = F.mse_loss(v_action, s["u_action"])
        loss = loss + cfg.loss_action_weight * la
        metrics["loss_action"] = la.detach()
    metrics["loss"] = loss.detach()
    return loss, metrics


@torch.no_grad()
def sample(
    model,
    cond: dict,
    mode: Mode,
    cfg: FlowConfig,
    steps: Optional[int] = None,
    use_cache: bool = False,
):
    """Euler ODE sampler. Denoises the streams the mode flags, tau 0 -> 1.

    `cond` keys: 'ctx', optional 'goal', and clean conditioning for un-noised
    streams ('action' for WORLD, 'video' for INVERSE). Shapes for the noised
    streams are taken from the model. Returns {'video': ..., 'action': ...} for
    whichever streams were denoised.

    `use_cache=True` (causal model only) computes the clean context's per-layer
    K/V once via `encode_prefix` and reuses it across every Euler step — the
    KV-cache fast path. Numerically equivalent to the uncached forward.
    """
    mask = MODE_TABLE[mode]
    steps = steps or cfg.sample_steps
    device = cond["ctx"].device
    B = cond["ctx"].shape[0]
    C = model.model_cfg.latent_channels
    Da = model.data_cfg.action_dim
    goal = cond.get("goal")
    mode_id = torch.full((B,), int(mode), device=device, dtype=torch.long)

    if mask.noise_video:
        x_v = torch.randn(B, model.n_video, C, device=device)
    else:
        x_v = cond["video"] if mode == Mode.INVERSE else torch.zeros(B, model.n_video, C, device=device)
    if mask.noise_action:
        x_a = torch.randn(B, model.n_action, Da, device=device)
    else:
        x_a = cond["action"]

    cached = use_cache and getattr(model, "causal", False)
    prefix = model.encode_prefix(cond["ctx"], mode_id, goal) if cached else None

    dt = 1.0 / steps
    for i in range(steps):
        tau = torch.full((B,), i * dt, device=device)
        if cached:
            v_video, v_action = model.forward_suffix(x_v, x_a, tau, mode_id, prefix, goal)
        else:
            v_video, v_action = model(cond["ctx"], x_v, x_a, tau, mode_id, goal)
        if mask.noise_video:
            x_v = x_v + dt * v_video
        if mask.noise_action:
            x_a = x_a + dt * v_action

    out = {}
    if mask.noise_video:
        out["video"] = x_v
    if mask.noise_action:
        out["action"] = x_a
    return out

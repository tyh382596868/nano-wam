"""Rectified flow matching — noising, loss, and ODE sampler.

Per stream, draw tau ~ U(0,1), Gaussian eps, interpolant x_tau = (1-tau)*eps +
tau*x1, target velocity u = x1 - eps. The network predicts v_theta; loss is MSE
to u on the noised+supervised streams only (per the active StreamMask).

Inference integrates dx/dtau = v_theta from tau: 0 -> 1 with Euler steps.
See DESIGN.md §5 and §6.

STATUS: stub. Bodies land in M2 (loss) and M3 (sampler).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .config import FlowConfig
from .modes import StreamMask


def interpolate(x1: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor):
    """Form x_tau and target velocity u = x1 - eps. Broadcasts tau over x dims."""
    raise NotImplementedError("interpolate is a stub (M2).")


def flow_loss(
    model,
    batch,
    mask: StreamMask,
    cfg: FlowConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute the joint rectified-flow loss for one batch under a StreamMask.

    Noises only the streams flagged in `mask`; passes the rest as clean
    conditioning. Returns (scalar_loss, metrics_dict).
    """
    raise NotImplementedError("flow_loss is a stub (M2).")


@torch.no_grad()
def sample(
    model,
    cond,
    mask: StreamMask,
    cfg: FlowConfig,
    steps: Optional[int] = None,
):
    """Euler ODE sampler. Denoises the streams flagged in `mask` from tau 0->1.

    `cond` carries clean context/goal (and clean action or video for the
    world/inverse modes). Returns the denoised streams. With mask noising only
    the action stream you get the imagination-free policy (FastWAM); noising both
    streams gives joint forecasting.
    """
    raise NotImplementedError("sample is a stub (M3).")

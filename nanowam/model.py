"""MoT Diffusion Transformer — the core of nano-wam.

Three token streams share one self-attention but use per-stream FFN + AdaLN:
  - context tokens  : tokenized history frames (always clean conditioning)
  - video tokens    : tokenized future frames  (world stream, may be noised)
  - action tokens   : projected action chunk   (policy stream, may be noised)

Conditioning (flow time tau, mode embedding, goal embed) enters via AdaLN-zero.
See DESIGN.md §4 for the block definition and §3 for shapes.

STATUS: stub. Public API + shape contracts are fixed; bodies land in M1.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig, DataConfig

# Stream ids used to route per-stream weights.
STREAM_CTX, STREAM_VIDEO, STREAM_ACTION = 0, 1, 2


class AdaLNZero(nn.Module):
    """Per-stream adaptive LayerNorm, zero-initialized (identity at init)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        raise NotImplementedError("AdaLNZero is a stub (M1).")


class MoTBlock(nn.Module):
    """One Mixture-of-Transformers block: shared attention, per-stream FFN/AdaLN."""

    def __init__(self, cfg: ModelConfig, cond_dim: int):
        super().__init__()
        self.cfg = cfg
        # TODO(M1): shared MHA; per-stream {AdaLN, FFN}; optional routed QKV
        #           when cfg.route_attention is True.
        raise NotImplementedError("MoTBlock is a stub (M1).")

    def forward(
        self,
        x: torch.Tensor,          # (B, T, d) concatenated tokens
        stream_ids: torch.Tensor, # (T,) long in {0,1,2}
        cond: torch.Tensor,       # (B, cond_dim) tau+mode+goal embedding
    ) -> torch.Tensor:
        raise NotImplementedError


class NanoWAM(nn.Module):
    """The world-action model.

    forward() consumes already-tokenized + already-noised stream latents and
    predicts per-stream rectified-flow velocities. Tokenization (frames<->latents)
    lives in FrameTokenizer; flow noising/loss/sampling live in flow.py.
    """

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        # TODO(M1): patch embed for video latents; linear embed for actions;
        #           context embed; positional embeddings; cond MLP (tau, mode,
        #           goal); N x MoTBlock; per-stream readout heads.
        raise NotImplementedError("NanoWAM is a stub (M1).")

    def forward(
        self,
        ctx_latents: torch.Tensor,       # (B, K*P, C) clean context tokens
        video_latents: torch.Tensor,     # (B, Hv*P, C) possibly-noised
        action_latents: torch.Tensor,    # (B, Ha, Da) possibly-noised
        tau: torch.Tensor,               # (B,) or (B, n_streams) flow time
        mode: torch.Tensor,              # (B,) long mode id (see modes.py)
        goal: Optional[torch.Tensor] = None,  # (B, Lg) or (B, d)
    ):
        """Returns (v_video, v_action) velocity predictions matching the inputs."""
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

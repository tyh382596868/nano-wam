"""MoT Diffusion Transformer — the core of nano-wam.

Three token streams share one self-attention but use per-stream FFN + AdaLN:
  - context tokens  : tokenized history frames (always clean conditioning)
  - video tokens    : tokenized future frames  (world stream, may be noised)
  - action tokens   : projected action chunk   (policy stream, may be noised)

Conditioning (flow time tau, mode embedding, goal embed) enters via AdaLN-zero.
See DESIGN.md §3 (shapes) and §4 (block definition).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, DataConfig
from .modes import Mode

# Stream ids used to route per-stream weights.
STREAM_CTX, STREAM_VIDEO, STREAM_ACTION = 0, 1, 2
N_STREAMS = 3


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift. shift/scale are (B, T, d)."""
    return x * (1 + scale) + shift


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of a continuous flow time t in [0, 1]. (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad odd dims
        emb = F.pad(emb, (0, 1))
    return emb


class Attention(nn.Module):
    """Multi-head self-attention. QKV shared across streams, or routed per stream."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.heads
        self.dim = cfg.dim
        self.routed = cfg.route_attention
        if self.routed:
            self.qkv = nn.ModuleList(nn.Linear(cfg.dim, 3 * cfg.dim) for _ in range(N_STREAMS))
        else:
            self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x: torch.Tensor, stream_ids: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        if self.routed:
            qkv = x.new_zeros(B, T, 3 * d)
            for s in range(N_STREAMS):
                m = stream_ids == s
                if m.any():
                    qkv[:, m] = self.qkv[s](x[:, m])
        else:
            qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        hd = d // self.heads
        q, k, v = (t.reshape(B, T, self.heads, hd).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # full (non-causal) self-attn
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class MoTBlock(nn.Module):
    """One Mixture-of-Transformers block: shared attention, per-stream FFN/AdaLN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = Attention(cfg)
        # per-stream FFN
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(N_STREAMS)
        )
        # per-stream AdaLN-zero modulation: cond -> (shift1,scale1,gate1,shift2,scale2,gate2)
        self.adaln = nn.ModuleList(nn.Linear(d, 6 * d) for _ in range(N_STREAMS))
        for lin in self.adaln:  # zero-init -> block starts as identity
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _stream_modulation(self, stream_ids: torch.Tensor, cond: torch.Tensor, B: int, T: int, d: int):
        """Build per-token (B, T, d) modulation tensors from per-stream AdaLN."""
        parts = [cond.new_zeros(B, T, d) for _ in range(6)]
        for s in range(N_STREAMS):
            m = stream_ids == s
            if not m.any():
                continue
            chunks = self.adaln[s](cond).chunk(6, dim=-1)  # each (B, d)
            for i, c in enumerate(chunks):
                parts[i][:, m] = c.unsqueeze(1)
        return parts  # shift1, scale1, gate1, shift2, scale2, gate2

    def forward(self, x: torch.Tensor, stream_ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self._stream_modulation(stream_ids, cond, B, T, d)
        x = x + g1 * self.attn(modulate(self.norm1(x), sh1, sc1), stream_ids)
        # per-stream FFN
        h = modulate(self.norm2(x), sh2, sc2)
        ffn_out = h.new_zeros(B, T, d)
        for s in range(N_STREAMS):
            m = stream_ids == s
            if m.any():
                ffn_out[:, m] = self.ffn[s](h[:, m])
        x = x + g2 * ffn_out
        return x


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
        d = model_cfg.dim
        C = model_cfg.latent_channels
        Da = data_cfg.action_dim
        P = (data_cfg.image_size // model_cfg.patch_size) ** 2
        self.P = P
        self.n_ctx = data_cfg.ctx_frames * P
        self.n_video = data_cfg.video_horizon * P
        self.n_action = data_cfg.action_horizon

        # input projections (per stream)
        self.ctx_proj = nn.Linear(C, d)
        self.video_proj = nn.Linear(C, d)
        self.action_proj = nn.Linear(Da, d)

        # learned positional embeddings (per stream, fixed horizons)
        self.ctx_pos = nn.Parameter(torch.zeros(1, self.n_ctx, d))
        self.video_pos = nn.Parameter(torch.zeros(1, self.n_video, d))
        self.action_pos = nn.Parameter(torch.zeros(1, self.n_action, d))
        for p in (self.ctx_pos, self.video_pos, self.action_pos):
            nn.init.normal_(p, std=0.02)

        # conditioning: time MLP + mode embedding (+ optional goal)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.mode_emb = nn.Embedding(len(Mode), d)
        if model_cfg.goal_embed == "none":
            self.goal_proj = None
        else:
            # accepts a (B, d) goal vector; richer goal encoders land with multi-task data
            self.goal_proj = nn.Linear(d, d)

        self.blocks = nn.ModuleList(MoTBlock(model_cfg) for _ in range(model_cfg.depth))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)

        # output heads (per stream), zero-init -> ~0 velocity at start
        self.video_head = nn.Linear(d, C)
        self.action_head = nn.Linear(d, Da)
        for head in (self.video_head, self.action_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _cond(self, tau: torch.Tensor, mode: torch.Tensor, goal: Optional[torch.Tensor], B: int):
        if tau.dim() > 1:  # (B, n_streams) -> shared tau for M1: use the first column
            tau = tau[:, 0]
        c = self.time_mlp(timestep_embedding(tau, self.model_cfg.dim)) + self.mode_emb(mode)
        if goal is not None and self.goal_proj is not None:
            c = c + self.goal_proj(goal)
        return c  # (B, d)

    def forward(
        self,
        ctx_latents: torch.Tensor,       # (B, K*P, C) clean context tokens
        video_latents: torch.Tensor,     # (B, Hv*P, C) possibly-noised
        action_latents: torch.Tensor,    # (B, Ha, Da) possibly-noised
        tau: torch.Tensor,               # (B,) or (B, n_streams) flow time
        mode: torch.Tensor,              # (B,) long mode id (see modes.py)
        goal: Optional[torch.Tensor] = None,  # (B, d)
    ):
        """Returns (v_video, v_action) velocity predictions matching the inputs."""
        B = ctx_latents.shape[0]
        ctx = self.ctx_proj(ctx_latents) + self.ctx_pos
        vid = self.video_proj(video_latents) + self.video_pos
        act = self.action_proj(action_latents) + self.action_pos

        x = torch.cat([ctx, vid, act], dim=1)  # (B, T, d)
        stream_ids = torch.cat([
            torch.full((self.n_ctx,), STREAM_CTX, device=x.device, dtype=torch.long),
            torch.full((self.n_video,), STREAM_VIDEO, device=x.device, dtype=torch.long),
            torch.full((self.n_action,), STREAM_ACTION, device=x.device, dtype=torch.long),
        ])

        cond = self._cond(tau, mode, goal, B)
        for blk in self.blocks:
            x = blk(x, stream_ids, cond)
        x = self.final_norm(x)

        vid_out = x[:, self.n_ctx:self.n_ctx + self.n_video]
        act_out = x[:, self.n_ctx + self.n_video:]
        v_video = self.video_head(vid_out)    # (B, Hv*P, C)
        v_action = self.action_head(act_out)  # (B, Ha, Da)
        return v_video, v_action

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

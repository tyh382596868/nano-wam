"""MoT Diffusion Transformer — the core of nano-wam.

Three token streams share one self-attention but use per-stream FFN + AdaLN:
  - context tokens  : tokenized history frames (always clean conditioning)
  - video tokens    : tokenized future frames  (world stream, may be noised)
  - action tokens   : projected action chunk   (policy stream, may be noised)

Conditioning enters via AdaLN-zero. The flow time `tau` modulates only the
*noised* streams (video/action); the clean context stream is conditioned by
mode/goal alone, so its K/V are independent of the denoising step — which is what
makes the KV cache (below) valid.

Causal mode (model_cfg.causal, requires action_horizon == video_horizon): a
block-causal mask over interleaved (action_i, frame_i) time-steps. Context is
visible to all and attends only to itself; step i attends to context + steps <= i.
This enables `encode_prefix` / `forward_suffix`: cache the clean context's
per-layer K/V once and reuse it across every Euler step (and across an
autoregressive rollout). See DESIGN.md §3-§4 and nanowam/rollout.py.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, DataConfig
from .modes import Mode

# Stream ids used to route per-stream weights.
STREAM_CTX, STREAM_VIDEO, STREAM_ACTION = 0, 1, 2
N_STREAMS = 3

KV = Tuple[torch.Tensor, torch.Tensor]  # (k, v) each (B, heads, T, head_dim)


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
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class Attention(nn.Module):
    """Multi-head self-attention. QKV shared across streams, or routed per stream.

    Supports an optional `prefix_kv` (cached keys/values to prepend) and can
    `return_kv` the current tokens' keys/values for caching.
    """

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

    def forward(self, x, stream_ids, attn_mask=None, prefix_kv: Optional[KV] = None,
                return_kv: bool = False):
        B, T, d = x.shape
        if self.routed:
            qkv = x.new_zeros(B, T, 3 * d)
            for s in range(N_STREAMS):
                m = stream_ids == s
                if m.any():
                    qkv[:, m] = self.qkv[s](x[:, m]).to(qkv.dtype)
        else:
            qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        hd = d // self.heads
        q, k, v = (t.reshape(B, T, self.heads, hd).transpose(1, 2) for t in (q, k, v))
        cur_kv = (k, v) if return_kv else None
        if prefix_kv is not None:
            pk, pv = prefix_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, d)
        out = self.proj(out)
        return (out, cur_kv) if return_kv else out


class MoTBlock(nn.Module):
    """One Mixture-of-Transformers block: shared attention, per-stream FFN/AdaLN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = Attention(cfg)
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(N_STREAMS)
        )
        # per-stream AdaLN-zero modulation
        self.adaln = nn.ModuleList(nn.Linear(d, 6 * d) for _ in range(N_STREAMS))
        for lin in self.adaln:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _stream_modulation(self, stream_ids, cond_ps, B, T, d):
        """cond_ps: (B, N_STREAMS, d) per-stream conditioning. -> 6 x (B, T, d)."""
        parts = [cond_ps.new_zeros(B, T, d) for _ in range(6)]
        for s in range(N_STREAMS):
            m = stream_ids == s
            if not m.any():
                continue
            chunks = self.adaln[s](cond_ps[:, s]).chunk(6, dim=-1)
            for i, c in enumerate(chunks):
                parts[i][:, m] = c.unsqueeze(1).to(parts[i].dtype)
        return parts

    def forward(self, x, stream_ids, cond_ps, attn_mask=None, prefix_kv=None, return_kv=False):
        B, T, d = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self._stream_modulation(stream_ids, cond_ps, B, T, d)
        attn_out = self.attn(modulate(self.norm1(x), sh1, sc1), stream_ids,
                             attn_mask=attn_mask, prefix_kv=prefix_kv, return_kv=return_kv)
        kv = None
        if return_kv:
            attn_out, kv = attn_out
        x = x + g1 * attn_out
        h = modulate(self.norm2(x), sh2, sc2)
        ffn_out = h.new_zeros(B, T, d)
        for s in range(N_STREAMS):
            m = stream_ids == s
            if m.any():
                ffn_out[:, m] = self.ffn[s](h[:, m]).to(ffn_out.dtype)
        x = x + g2 * ffn_out
        return (x, kv) if return_kv else x


class NanoWAM(nn.Module):
    """The world-action model. See module docstring + DESIGN.md."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.causal = getattr(model_cfg, "causal", False)
        d = model_cfg.dim
        C = model_cfg.latent_channels
        Da = data_cfg.action_dim
        P = (data_cfg.image_size // model_cfg.patch_size) ** 2
        self.P = P
        self.n_ctx = data_cfg.ctx_frames * P
        self.n_video = data_cfg.video_horizon * P
        self.n_action = data_cfg.action_horizon

        if self.causal:
            assert data_cfg.action_horizon == data_cfg.video_horizon, (
                "causal mode interleaves (action_i, frame_i): set action_horizon == video_horizon"
            )

        self.ctx_proj = nn.Linear(C, d)
        self.video_proj = nn.Linear(C, d)
        self.action_proj = nn.Linear(Da, d)

        self.ctx_pos = nn.Parameter(torch.zeros(1, self.n_ctx, d))
        self.video_pos = nn.Parameter(torch.zeros(1, self.n_video, d))
        self.action_pos = nn.Parameter(torch.zeros(1, self.n_action, d))
        for p in (self.ctx_pos, self.video_pos, self.action_pos):
            nn.init.normal_(p, std=0.02)

        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.mode_emb = nn.Embedding(len(Mode), d)
        self.goal_proj = None if model_cfg.goal_embed == "none" else nn.Linear(d, d)

        self.blocks = nn.ModuleList(MoTBlock(model_cfg) for _ in range(model_cfg.depth))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.video_head = nn.Linear(d, C)
        self.action_head = nn.Linear(d, Da)
        for head in (self.video_head, self.action_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        if self.causal:
            self.register_buffer("full_mask", self._build_causal_mask(), persistent=False)

    # --- conditioning -------------------------------------------------------
    def _cond_parts(self, tau, mode, goal):
        if tau.dim() > 1:
            tau = tau[:, 0]
        static = self.mode_emb(mode)
        if goal is not None and self.goal_proj is not None:
            static = static + self.goal_proj(goal)
        time = self.time_mlp(timestep_embedding(tau, self.model_cfg.dim))
        return static, time  # each (B, d)

    def _cond_per_stream(self, static, time):
        """(B, N_STREAMS, d): ctx gets no flow-time; video/action do."""
        return torch.stack([static, static + time, static + time], dim=1)

    # --- causal mask --------------------------------------------------------
    def _time_index(self):
        """Per-token time-step index for the [ctx ; video ; action] order."""
        Hv = self.data_cfg.video_horizon
        idx = [-1] * self.n_ctx
        idx += [i for i in range(Hv) for _ in range(self.P)]   # frame i at step i
        idx += [j for j in range(self.n_action)]               # action j at step j
        return torch.tensor(idx, dtype=torch.long)

    def _build_causal_mask(self):
        t = self._time_index()
        return (t[None, :] <= t[:, None])  # (T, T) bool: query q attends key k iff t_k <= t_q

    # --- embedding ----------------------------------------------------------
    def _embed_ctx(self, ctx_latents):
        return self.ctx_proj(ctx_latents) + self.ctx_pos

    def _embed_suffix(self, video_latents, action_latents):
        vid = self.video_proj(video_latents) + self.video_pos
        act = self.action_proj(action_latents) + self.action_pos
        return torch.cat([vid, act], dim=1)

    def _suffix_stream_ids(self, device):
        return torch.cat([
            torch.full((self.n_video,), STREAM_VIDEO, device=device, dtype=torch.long),
            torch.full((self.n_action,), STREAM_ACTION, device=device, dtype=torch.long),
        ])

    def _readout(self, suffix):
        suffix = self.final_norm(suffix)
        vid_out = suffix[:, :self.n_video]
        act_out = suffix[:, self.n_video:]
        return self.video_head(vid_out), self.action_head(act_out)

    # --- forward ------------------------------------------------------------
    def forward(self, ctx_latents, video_latents, action_latents, tau, mode, goal=None):
        """Single-pass forward. Returns (v_video, v_action)."""
        B = ctx_latents.shape[0]
        device = ctx_latents.device
        ctx = self._embed_ctx(ctx_latents)
        suffix = self._embed_suffix(video_latents, action_latents)
        x = torch.cat([ctx, suffix], dim=1)
        stream_ids = torch.cat([
            torch.full((self.n_ctx,), STREAM_CTX, device=device, dtype=torch.long),
            self._suffix_stream_ids(device),
        ])
        static, time = self._cond_parts(tau, mode, goal)
        cond_ps = self._cond_per_stream(static, time)
        mask = self.full_mask if self.causal else None
        for blk in self.blocks:
            x = blk(x, stream_ids, cond_ps, attn_mask=mask)
        return self._readout(x[:, self.n_ctx:])

    # --- cached (causal-only) path -----------------------------------------
    def encode_prefix(self, ctx_latents, mode, goal=None) -> List[KV]:
        """Run the clean context through all blocks; return per-layer ctx K/V.

        Valid only in causal mode: context attends to itself and is not modulated
        by the flow time, so its K/V are constant across the denoising trajectory.
        """
        assert self.causal, "encode_prefix requires a causal model"
        static, time = self._cond_parts(
            torch.zeros(ctx_latents.shape[0], device=ctx_latents.device), mode, goal)
        cond_ps = self._cond_per_stream(static, time)  # ctx column ignores `time`
        sids = torch.full((self.n_ctx,), STREAM_CTX, device=ctx_latents.device, dtype=torch.long)
        x = self._embed_ctx(ctx_latents)
        cache = []
        for blk in self.blocks:
            x, kv = blk(x, sids, cond_ps, attn_mask=None, return_kv=True)
            cache.append(kv)
        return cache

    def forward_suffix(self, video_latents, action_latents, tau, mode, prefix_cache, goal=None):
        """Forward only the noised suffix, attending to a cached context prefix."""
        assert self.causal, "forward_suffix requires a causal model"
        device = video_latents.device
        x = self._embed_suffix(video_latents, action_latents)
        sids = self._suffix_stream_ids(device)
        static, time = self._cond_parts(tau, mode, goal)
        cond_ps = self._cond_per_stream(static, time)
        # suffix queries attend to keys [ctx ; suffix]; reuse the full mask's suffix rows
        mask = self.full_mask[self.n_ctx:, :]  # (Ts, T)
        for blk, kv in zip(self.blocks, prefix_cache):
            x = blk(x, sids, cond_ps, attn_mask=mask, prefix_kv=kv)
        return self._readout(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

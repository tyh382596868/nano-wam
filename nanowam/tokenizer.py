"""Frame tokenizer — nano-wam's stand-in for the Wan VAE.

A tiny *deterministic* conv autoencoder (no KL/VAE sampling) mapping a low-res
RGB frame to a small latent grid and back. Default: (3, 64, 64) <-> (C, 8, 8)
with C = ModelConfig.latent_channels, downsample factor = ModelConfig.patch_size.

Because patch_size == downsample factor, each latent grid cell is exactly one
token for the WAM (P = (image_size / patch_size)**2). Helpers `to_tokens` /
`from_tokens` convert between the (B, T, C, h, w) grid layout and the
(B, T*h*w, C) token layout the model consumes.

The tokenizer is trained jointly with the WAM (M2). See DESIGN.md §2.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig, DataConfig


def to_tokens(grids: torch.Tensor) -> torch.Tensor:
    """(B, T, C, h, w) -> (B, T*h*w, C). Spatial cells become tokens."""
    B, T, C, h, w = grids.shape
    return grids.permute(0, 1, 3, 4, 2).reshape(B, T * h * w, C)


def from_tokens(tokens: torch.Tensor, T: int, h: int, w: int) -> torch.Tensor:
    """(B, T*h*w, C) -> (B, T, C, h, w). Inverse of `to_tokens`."""
    B, N, C = tokens.shape
    assert N == T * h * w, f"token count {N} != T*h*w {T*h*w}"
    return tokens.reshape(B, T, h, w, C).permute(0, 1, 4, 2, 3).contiguous()


class FrameTokenizer(nn.Module):
    """Conv AE. Encodes frames to latents, decodes latents to frames.

    Leading batch/time dims are handled transparently: inputs may be
    (B, 3, H, W) or (B, T, 3, H, W); the latent mirrors the leading dims.
    """

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super().__init__()
        self.cfg = model_cfg
        self.data_cfg = data_cfg
        C = model_cfg.latent_channels
        f = model_cfg.patch_size  # downsample factor (e.g. 8 -> three stride-2 stages)
        n_down = int(round(math.log2(f)))
        assert 2 ** n_down == f, f"patch_size must be a power of 2, got {f}"
        self.latent_hw = data_cfg.image_size // f

        chans = [3, 32, 64, 128][: n_down + 1]
        if len(chans) < n_down + 1:  # very deep downsample: keep widest width
            chans = chans + [chans[-1]] * (n_down + 1 - len(chans))

        # Encoder: n_down stride-2 convs, then project to C channels.
        enc = []
        for i in range(n_down):
            enc += [nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1),
                    nn.GroupNorm(8, chans[i + 1]), nn.SiLU()]
        enc += [nn.Conv2d(chans[n_down], C, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        # Decoder: project from C, then n_down stride-2 transposed convs back to RGB.
        dec = [nn.Conv2d(C, chans[n_down], 3, padding=1), nn.SiLU()]
        for i in range(n_down, 0, -1):
            dec += [nn.ConvTranspose2d(chans[i], chans[i - 1] if i - 1 > 0 else 32,
                                       4, stride=2, padding=1),
                    nn.GroupNorm(8, chans[i - 1] if i - 1 > 0 else 32), nn.SiLU()]
        dec += [nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()]  # frames in [0, 1]
        self.decoder = nn.Sequential(*dec)

    def _fold(self, x: torch.Tensor):
        """Collapse leading dims into batch; return (folded, lead_shape)."""
        *lead, c, h, w = x.shape
        return x.reshape(-1, c, h, w), tuple(lead)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (..., 3, H, W) -> latents (..., C, H/f, W/f)."""
        x, lead = self._fold(frames)
        z = self.encoder(x)
        return z.reshape(*lead, *z.shape[1:])

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents (..., C, H/f, W/f) -> frames (..., 3, H, W)."""
        z, lead = self._fold(latents)
        x = self.decoder(z)
        return x.reshape(*lead, *x.shape[1:])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Reconstruct (for recon-loss warmup / sanity checks)."""
        return self.decode(self.encode(frames))

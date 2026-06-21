"""Frame tokenizer — nano-wam's stand-in for the Wan VAE.

A tiny conv encoder/decoder mapping a low-res RGB frame to a small latent grid
and back. Default: (3, 64, 64) <-> (C, 8, 8) with C = ModelConfig.latent_channels.

Design notes (see DESIGN.md §2):
  - Keep it small; this is NOT the focus of nano-wam.
  - M1 option A: train jointly with the WAM (simplest).
  - M1 option B: operate on raw downsampled pixels and skip the latent entirely
    (set latent == patchified pixels). Useful for ultra-nano debugging.

STATUS: stub. Shapes are the contract; bodies land in M1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, DataConfig


class FrameTokenizer(nn.Module):
    """Conv VAE-lite. Encodes frames to latents, decodes latents to frames."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super().__init__()
        self.cfg = model_cfg
        self.data_cfg = data_cfg
        # TODO(M1): conv down-stack to (C, H/8, W/8) and mirror up-stack.
        raise NotImplementedError("FrameTokenizer is a stub (M1).")

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, *, 3, H, W) -> latents (B, *, C, H/8, W/8)."""
        raise NotImplementedError

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents (B, *, C, H/8, W/8) -> frames (B, *, 3, H, W)."""
        raise NotImplementedError

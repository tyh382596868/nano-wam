"""Shape-contract tests — the executable half of DESIGN.md §3.

These encode the tensor shapes the implementation must satisfy. They are marked
xfail while the modules are stubs (M0); flip them on as milestones land:
  - M1: tokenizer round-trip + NanoWAM.forward velocity shapes
  - M2: flow_loss returns a scalar; sample_mode respects probs

Run: pytest -q   (pytest is optional; see comments for a no-pytest fallback)
"""
from __future__ import annotations

import pytest

from nanowam.config import Config, load_config


def test_config_defaults_match_dataclasses():
    """load_config on the default yaml should equal the dataclass defaults' shape."""
    cfg = load_config("configs/pusht_tiny.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.action_dim == 2
    assert cfg.model.dim == 256 and cfg.model.depth == 8
    # P = (image_size / patch_size) ** 2
    p = (cfg.data.image_size // cfg.model.patch_size) ** 2
    assert p == 64


def test_tokenizer_roundtrip_shapes():
    """encode/decode preserve frame shape and hit the configured latent grid."""
    import torch
    from nanowam.tokenizer import FrameTokenizer, to_tokens, from_tokens

    cfg = load_config("configs/pusht_tiny.yaml")
    tok = FrameTokenizer(cfg.model, cfg.data)

    B, T, H = 2, cfg.data.video_horizon, cfg.data.image_size
    frames = torch.rand(B, T, 3, H, H)
    z = tok.encode(frames)
    h = cfg.data.image_size // cfg.model.patch_size
    assert z.shape == (B, T, cfg.model.latent_channels, h, h)
    recon = tok.decode(z)
    assert recon.shape == frames.shape
    assert (recon >= 0).all() and (recon <= 1).all()  # sigmoid output

    # token <-> grid round-trip
    tokens = to_tokens(z)
    assert tokens.shape == (B, T * h * h, cfg.model.latent_channels)
    assert torch.allclose(from_tokens(tokens, T, h, h), z)


def test_forward_velocity_shapes():
    import torch
    from nanowam.model import NanoWAM

    cfg = load_config("configs/pusht_tiny.yaml")
    model = NanoWAM(cfg.model, cfg.data)

    B = 2
    P = (cfg.data.image_size // cfg.model.patch_size) ** 2
    C = cfg.model.latent_channels
    ctx = torch.randn(B, cfg.data.ctx_frames * P, C)
    vid = torch.randn(B, cfg.data.video_horizon * P, C)
    act = torch.randn(B, cfg.data.action_horizon, cfg.data.action_dim)
    tau = torch.rand(B)
    mode = torch.zeros(B, dtype=torch.long)

    v_vid, v_act = model(ctx, vid, act, tau, mode)
    assert v_vid.shape == vid.shape
    assert v_act.shape == act.shape


def test_forward_runs_with_routed_attention_and_goal():
    """route_attention + a goal vector must not change output shapes."""
    import torch
    from nanowam.model import NanoWAM

    cfg = load_config("configs/pusht_tiny.yaml")
    cfg.model.route_attention = True
    model = NanoWAM(cfg.model, cfg.data)

    B = 2
    P = (cfg.data.image_size // cfg.model.patch_size) ** 2
    C = cfg.model.latent_channels
    ctx = torch.randn(B, cfg.data.ctx_frames * P, C)
    vid = torch.randn(B, cfg.data.video_horizon * P, C)
    act = torch.randn(B, cfg.data.action_horizon, cfg.data.action_dim)
    goal = torch.randn(B, cfg.model.dim)
    v_vid, v_act = model(ctx, vid, act, torch.rand(B), torch.ones(B, dtype=torch.long), goal)
    assert v_vid.shape == vid.shape and v_act.shape == act.shape


@pytest.mark.xfail(reason="flow_loss is a stub until M2", strict=True)
def test_flow_loss_is_scalar():
    raise NotImplementedError

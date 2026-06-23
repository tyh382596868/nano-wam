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


def _tiny_cfg():
    cfg = load_config("configs/pusht_tiny.yaml")
    cfg.model.dim = 64
    cfg.model.depth = 2
    cfg.model.heads = 4
    return cfg


def _random_latent_batch(cfg, B=4):
    import torch
    P = (cfg.data.image_size // cfg.model.patch_size) ** 2
    C = cfg.model.latent_channels
    return {
        "ctx": torch.randn(B, cfg.data.ctx_frames * P, C),
        "video": torch.randn(B, cfg.data.video_horizon * P, C),
        "action": torch.randn(B, cfg.data.action_horizon, cfg.data.action_dim),
    }


@pytest.mark.parametrize("mode_name", ["joint", "policy", "world", "inverse"])
def test_flow_loss_is_scalar(mode_name):
    import torch
    from nanowam.flow import flow_loss
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode

    cfg = _tiny_cfg()
    model = NanoWAM(cfg.model, cfg.data)
    batch = _random_latent_batch(cfg)
    loss, metrics = flow_loss(model, batch, Mode[mode_name.upper()], cfg.flow)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()  # must be differentiable
    assert "loss" in metrics


def test_sample_shapes():
    import torch
    from nanowam.flow import sample
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode

    cfg = _tiny_cfg()
    model = NanoWAM(cfg.model, cfg.data)
    B, P, C = 2, model.P, cfg.model.latent_channels
    cond = {"ctx": torch.randn(B, model.n_ctx, C)}

    out = sample(model, cond, Mode.POLICY, cfg.flow, steps=3)
    assert out["action"].shape == (B, cfg.data.action_horizon, cfg.data.action_dim)
    assert "video" not in out  # policy does not imagine

    out = sample(model, cond, Mode.JOINT, cfg.flow, steps=3)
    assert out["video"].shape == (B, model.n_video, C)
    assert out["action"].shape == (B, cfg.data.action_horizon, cfg.data.action_dim)


def test_policy_overfit_decreases():
    """A tiny WAM should overfit action flow on a small fixed batch."""
    import numpy as np
    import torch
    from nanowam.data import generate_episode, make_windows
    from nanowam.flow import flow_loss
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode
    from nanowam.tokenizer import FrameTokenizer, to_tokens
    from nanowam.utils import set_seed

    set_seed(0)
    cfg = _tiny_cfg()
    tok = FrameTokenizer(cfg.model, cfg.data)  # fixed random tokenizer (ctx detached)
    model = NanoWAM(cfg.model, cfg.data)

    rng = np.random.default_rng(0)
    windows = []
    while len(windows) < 16:
        f, a = generate_episode(rng, 24, cfg.data.image_size)
        windows += make_windows(f, a, cfg.data.ctx_frames, cfg.data.video_horizon, cfg.data.action_horizon)
    windows = windows[:16]
    ctx = torch.from_numpy(np.stack([w[0] for w in windows])).float() / 255.0
    act = torch.from_numpy(np.stack([w[2] for w in windows])).float()
    with torch.no_grad():
        ctx_tok = to_tokens(tok.encode(ctx))
    batch = {"ctx": ctx_tok, "video": torch.zeros(16, model.n_video, cfg.model.latent_channels),
             "action": act}

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(120):
        opt.zero_grad()
        loss, _ = flow_loss(model, batch, Mode.POLICY, cfg.flow, generator=gen)
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.6 * losses[0], f"no overfit: {losses[0]:.3f} -> {losses[-1]:.3f}"

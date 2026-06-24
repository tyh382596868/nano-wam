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


def test_world_generation_decode_pipeline():
    """world sampling -> latent tokens -> decode back to valid frames."""
    import torch
    from nanowam.flow import sample
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode
    from nanowam.tokenizer import FrameTokenizer, from_tokens

    cfg = _tiny_cfg()
    tok = FrameTokenizer(cfg.model, cfg.data)
    model = NanoWAM(cfg.model, cfg.data)

    B, C = 2, cfg.model.latent_channels
    h = cfg.data.image_size // cfg.model.patch_size
    cond = {
        "ctx": torch.randn(B, model.n_ctx, C),
        "action": torch.randn(B, cfg.data.action_horizon, cfg.data.action_dim),
    }
    out = sample(model, cond, Mode.WORLD, cfg.flow, steps=3)
    assert "video" in out and "action" not in out  # world predicts frames, conditions on action

    frames = tok.decode(from_tokens(out["video"], cfg.data.video_horizon, h, h))
    assert frames.shape == (B, cfg.data.video_horizon, 3, cfg.data.image_size, cfg.data.image_size)
    assert (frames >= 0).all() and (frames <= 1).all()


def test_to_chw_uint8_resizes():
    import numpy as np
    from nanowam.sources import to_chw_uint8

    hwc = (np.random.rand(40, 50, 3) * 255).astype(np.uint8)  # wrong size, HWC uint8
    out = to_chw_uint8(hwc, 64)
    assert out.shape == (3, 64, 64) and out.dtype == np.uint8

    chw_float = np.random.rand(3, 70, 70).astype(np.float32)  # CHW float[0,1]
    out2 = to_chw_uint8(chw_float, 64)
    assert out2.shape == (3, 64, 64) and out2.max() <= 255


def test_lerobot_source_lazy_import():
    """Constructing LeRobotSource must not import lerobot; iterating errors clearly."""
    from nanowam.sources import LeRobotSource

    cfg = _tiny_cfg()
    src = LeRobotSource(cfg.data, "lerobot/pusht")  # no import at construction
    with pytest.raises(ImportError):
        next(src.iter_episodes())


def test_source_to_shards_roundtrip(tmp_path):
    import os
    import numpy as np
    from nanowam.data import make_windows, WindowDataset
    from nanowam.sources import EpisodeSource

    cfg = _tiny_cfg()
    cfg.data.root = str(tmp_path)
    S = cfg.data.image_size

    class MockSource(EpisodeSource):
        def iter_episodes(self):
            rng = np.random.default_rng(0)
            for _ in range(2):
                frames = (rng.random((20, 3, S, S)) * 255).astype(np.uint8)
                actions = rng.standard_normal((20, cfg.data.action_dim)).astype(np.float32)
                yield frames, actions

    windows = []
    for f, a in MockSource().iter_episodes():
        windows += make_windows(f, a, cfg.data.ctx_frames, cfg.data.video_horizon,
                                cfg.data.action_horizon)
    assert windows

    acts = np.concatenate([w[2] for w in windows]).reshape(-1, cfg.data.action_dim)
    mean = acts.mean(0, keepdims=True).astype(np.float32)
    std = (acts.std(0, keepdims=True) + 1e-6).astype(np.float32)
    np.savez(os.path.join(str(tmp_path), "stats.npz"), action_mean=mean, action_std=std)

    os.makedirs(os.path.join(str(tmp_path), "train"))
    ctx = np.stack([w[0] for w in windows])
    fut = np.stack([w[1] for w in windows])
    act = ((np.stack([w[2] for w in windows]) - mean) / std).astype(np.float32)
    np.savez_compressed(os.path.join(str(tmp_path), "train", "shard_000.npz"),
                        ctx=ctx, future=fut, action=act)

    ds = WindowDataset(cfg.data, "train")
    item = ds[0]
    assert item["ctx_frames"].shape == (cfg.data.ctx_frames, 3, S, S)
    assert item["future_frames"].shape == (cfg.data.video_horizon, 3, S, S)
    assert item["actions"].shape == (cfg.data.action_horizon, cfg.data.action_dim)
    assert ds.action_mean is not None and ds.action_mean.shape[-1] == cfg.data.action_dim


def test_reacher_env_step():
    import numpy as np
    from nanowam.envs import ReacherEnv

    cfg = _tiny_cfg()
    env = ReacherEnv(cfg.data, seed=0)
    frame = env.reset()
    assert frame.shape == (3, cfg.data.image_size, cfg.data.image_size)
    assert frame.min() >= 0 and frame.max() <= 1

    # an action pointing at the goal should reduce the distance
    direction = (env.goal - env.agent)
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    _, _, d0 = env.step(direction * 0.0)   # no-op to read current dist
    _, done, d1 = env.step(direction)
    assert d1 < d0 and isinstance(done, bool)


def test_closed_loop_ablation_runs():
    """The imagination ablation harness runs end-to-end and returns valid stats."""
    from nanowam.eval import run_ablation
    from nanowam.model import NanoWAM
    from nanowam.tokenizer import FrameTokenizer

    cfg = _tiny_cfg()
    cfg.flow.sample_steps = 3
    cfg.eval.replan_every = 2
    tok = FrameTokenizer(cfg.model, cfg.data)
    model = NanoWAM(cfg.model, cfg.data)

    res = run_ablation(model, tok, cfg, device="cpu", episodes=2, max_steps=6)
    assert set(res) == {"none", "joint"}
    for r in res.values():
        assert 0.0 <= r["success_rate"] <= 1.0 and r["n"] == 2


def _causal_cfg():
    cfg = _tiny_cfg()
    cfg.model.causal = True
    cfg.data.action_horizon = cfg.data.video_horizon  # required for interleaving
    return cfg


def test_causal_forward_and_train_step():
    """A causal model forwards to the right shapes and takes a flow_loss step."""
    import torch
    from nanowam.flow import flow_loss
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode

    cfg = _causal_cfg()
    model = NanoWAM(cfg.model, cfg.data)
    assert model.causal and model.full_mask.shape[0] == model.full_mask.shape[1]

    batch = _random_latent_batch(cfg)
    loss, _ = flow_loss(model, batch, Mode.JOINT, cfg.flow)
    assert torch.isfinite(loss)
    loss.backward()


def test_kv_cache_forward_equivalence():
    """encode_prefix + forward_suffix must equal the single-pass causal forward."""
    import torch
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode

    cfg = _causal_cfg()
    torch.manual_seed(0)
    model = NanoWAM(cfg.model, cfg.data).eval()
    batch = _random_latent_batch(cfg, B=2)
    tau = torch.rand(2)
    mode_id = torch.full((2,), int(Mode.JOINT), dtype=torch.long)

    with torch.no_grad():
        v_full = model(batch["ctx"], batch["video"], batch["action"], tau, mode_id)
        prefix = model.encode_prefix(batch["ctx"], mode_id)
        v_cached = model.forward_suffix(batch["video"], batch["action"], tau, mode_id, prefix)

    for a, b in zip(v_full, v_cached):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_causal_mask_blocks_future():
    """Perturbing a later-step token must not change an earlier step's output."""
    import torch
    from nanowam.model import NanoWAM
    from nanowam.modes import Mode

    cfg = _causal_cfg()
    torch.manual_seed(0)
    model = NanoWAM(cfg.model, cfg.data).eval()
    b = _random_latent_batch(cfg, B=1)
    tau = torch.rand(1)
    mode_id = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        v0, _ = model(b["ctx"], b["video"], b["action"], tau, mode_id)
        vid2 = b["video"].clone()
        vid2[:, -model.P:] += 5.0  # perturb the LAST frame's tokens
        v1, _ = model(b["ctx"], vid2, b["action"], tau, mode_id)
    # the first frame (step 0) must be unaffected by the last frame (step Hv-1)
    assert torch.allclose(v0[:, :model.P], v1[:, :model.P], atol=1e-5)


def test_dream_rollout_longhorizon():
    import torch
    from nanowam.model import NanoWAM
    from nanowam.rollout import dream_rollout
    from nanowam.tokenizer import FrameTokenizer

    cfg = _causal_cfg()
    cfg.flow.sample_steps = 3
    tok = FrameTokenizer(cfg.model, cfg.data)
    model = NanoWAM(cfg.model, cfg.data)

    init = torch.rand(cfg.data.ctx_frames, 3, cfg.data.image_size, cfg.data.image_size)
    n_blocks = 4
    frames = dream_rollout(model, tok, init, n_blocks, cfg, device="cpu")
    assert frames.shape == (n_blocks * cfg.data.video_horizon, 3,
                            cfg.data.image_size, cfg.data.image_size)
    assert (frames >= 0).all() and (frames <= 1).all()


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

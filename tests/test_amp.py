"""Regression test for the AMP (autocast) path.

The per-stream scatters in MoTBlock (AdaLN modulation, routed QKV, FFN) write
Linear/Embedding outputs into zero-init buffers with an in-place index-put. Under
`torch.autocast`, the Linear outputs are half while the buffers are float32, so
the assignment raised "Index put requires the source and destination dtypes
match". This was invisible to the CPU-only suite (autocast is a CUDA path).

These tests run only when CUDA is available; CI's CPU runner skips them.
"""
import pytest
import torch

from nanowam.config import load_config
from nanowam.flow import flow_loss
from nanowam.model import NanoWAM
from nanowam.modes import Mode
from nanowam.tokenizer import FrameTokenizer, to_tokens

CONFIGS = ["configs/pusht_tiny.yaml", "configs/causal_tiny.yaml"]
MODES = [Mode.JOINT, Mode.POLICY, Mode.WORLD, Mode.INVERSE]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP autocast is a CUDA path")
@pytest.mark.parametrize("config", CONFIGS)
def test_amp_flow_loss_all_modes(config):
    cfg = load_config(config)
    dev = "cuda"
    tok = FrameTokenizer(cfg.model, cfg.data).to(dev)
    model = NanoWAM(cfg.model, cfg.data).to(dev)
    B = 2
    ctx = torch.rand(B, cfg.data.ctx_frames, 3, 64, 64, device=dev)
    fut = torch.rand(B, cfg.data.video_horizon, 3, 64, 64, device=dev)
    act = torch.rand(B, cfg.data.action_horizon, cfg.data.action_dim, device=dev)
    for mode in MODES:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            lat = {"ctx": to_tokens(tok.encode(ctx)),
                   "video": to_tokens(tok.encode(fut)),
                   "action": act}
            loss, _ = flow_loss(model, lat, mode, cfg.flow)
        assert torch.isfinite(loss), f"non-finite AMP loss in mode {mode.name}"
        loss.backward()  # must not raise on the in-place scatters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP autocast is a CUDA path")
def test_amp_routed_attention():
    """route_attention=True exercises the QKV scatter under autocast too."""
    cfg = load_config("configs/pusht_tiny.yaml")
    cfg.model.route_attention = True
    dev = "cuda"
    tok = FrameTokenizer(cfg.model, cfg.data).to(dev)
    model = NanoWAM(cfg.model, cfg.data).to(dev)
    B = 2
    ctx = torch.rand(B, cfg.data.ctx_frames, 3, 64, 64, device=dev)
    fut = torch.rand(B, cfg.data.video_horizon, 3, 64, 64, device=dev)
    act = torch.rand(B, cfg.data.action_horizon, cfg.data.action_dim, device=dev)
    with torch.autocast("cuda"):
        lat = {"ctx": to_tokens(tok.encode(ctx)),
               "video": to_tokens(tok.encode(fut)),
               "action": act}
        loss, _ = flow_loss(model, lat, Mode.JOINT, cfg.flow)
    assert torch.isfinite(loss)
    loss.backward()

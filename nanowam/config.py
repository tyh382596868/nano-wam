"""Config dataclasses for nano-wam.

These mirror configs/pusht_tiny.yaml one-to-one. `load_config` reads a YAML file
and returns a populated `Config`. Kept dependency-light (yaml + dataclasses).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, get_type_hints


@dataclass
class DataConfig:
    name: str = "pusht"
    root: str = "data/pusht"
    image_size: int = 64
    ctx_frames: int = 2          # K
    video_horizon: int = 4       # Hv
    action_horizon: int = 8      # Ha
    action_dim: int = 2          # Da


@dataclass
class ModelConfig:
    dim: int = 256               # d
    depth: int = 8               # N MoT blocks
    heads: int = 8
    latent_channels: int = 4     # C
    patch_size: int = 8
    route_attention: bool = False  # MoT: route FFN+AdaLN only by default
    causal: bool = False           # block-causal attention (needs Ha == Hv); enables KV-cache AR
    goal_embed: str = "onehot"   # onehot | none | frozen_text


@dataclass
class FlowConfig:
    loss_video_weight: float = 1.0
    loss_action_weight: float = 1.0
    per_stream_time: bool = False
    sample_steps: int = 10


@dataclass
class TrainConfig:
    mode_probs: Dict[str, float] = field(
        default_factory=lambda: {"joint": 0.5, "policy": 0.25, "world": 0.15, "inverse": 0.10}
    )
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 20000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 50
    ckpt_every: int = 2000
    out_dir: str = "runs/pusht_tiny"


@dataclass
class EvalConfig:
    episodes: int = 20
    imagine: str = "none"        # none | joint
    replan_every: int = 4


@dataclass
class Config:
    seed: int = 1337
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _from_dict(cls, data: Dict[str, Any]):
    """Shallow-recursive dataclass builder; ignores unknown keys with a warning."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    # get_type_hints resolves the string annotations produced by
    # `from __future__ import annotations` back into real types.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        kwargs[key] = _from_dict(ftype, value) if is_dataclass(ftype) else value
    return cls(**kwargs)


def load_config(path: str) -> Config:
    """Load a YAML config file into a `Config`."""
    import yaml  # local import keeps import-time deps minimal

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)

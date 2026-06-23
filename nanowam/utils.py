"""Small helpers: seeding, checkpoint I/O, simple logging.

STATUS: stub. Bodies land in M1/M2 as needed.
"""
from __future__ import annotations

from typing import Any, Dict


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (+cuda) for reproducible toy runs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model, tokenizer, optimizer, step: int,
                    extra: Dict[str, Any] | None = None) -> None:
    import os

    import torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str, model, tokenizer=None, optimizer=None) -> int:
    """Restore model (+tokenizer/optimizer) in place; return the saved step."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if tokenizer is not None and ckpt.get("tokenizer") is not None:
        tokenizer.load_state_dict(ckpt["tokenizer"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)

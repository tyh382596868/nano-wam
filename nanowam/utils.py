"""Small helpers: seeding, checkpoint I/O, simple logging.

STATUS: stub. Bodies land in M1/M2 as needed.
"""
from __future__ import annotations

from typing import Any, Dict


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (+cuda) for reproducible toy runs."""
    raise NotImplementedError("set_seed is a stub (M1).")


def save_checkpoint(path: str, model, optimizer, step: int, extra: Dict[str, Any] | None = None) -> None:
    raise NotImplementedError("save_checkpoint is a stub (M2).")


def load_checkpoint(path: str, model, optimizer=None):
    """Returns the saved step (and restores model/optimizer in place)."""
    raise NotImplementedError("load_checkpoint is a stub (M2).")

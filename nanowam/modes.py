"""Training/inference modes — the UniDiffuser-style mask scheme.

One trained network serves several jobs depending on which streams are clean
(conditioning) vs. noised (predicted & supervised). See DESIGN.md §5.

  mode      clean (cond)            noised (predicted)
  --------  ---------------------   ------------------
  joint     ctx, goal               video + action
  policy    ctx, goal               action
  world     ctx, goal, action       video
  inverse   ctx, future video       action

A mode is described by a `StreamMask`: booleans telling the flow machinery which
streams to noise/supervise. `sample_mode` draws one according to train.mode_probs.

STATUS: stub. Enum + mask table to be finalized in M2.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict


class Mode(IntEnum):
    JOINT = 0
    POLICY = 1
    WORLD = 2
    INVERSE = 3


@dataclass(frozen=True)
class StreamMask:
    """Which streams are noised (predicted). Unnoised streams are clean cond.

    `goal_present` lets `policy`/`world` optionally drop the goal for
    classifier-free-guidance-style training later.
    """
    noise_video: bool
    noise_action: bool
    goal_present: bool = True


# Canonical mask per mode. Context tokens are ALWAYS clean conditioning.
MODE_TABLE: Dict[Mode, StreamMask] = {
    Mode.JOINT:   StreamMask(noise_video=True,  noise_action=True),
    Mode.POLICY:  StreamMask(noise_video=False, noise_action=True),
    Mode.WORLD:   StreamMask(noise_video=True,  noise_action=False),
    Mode.INVERSE: StreamMask(noise_video=False, noise_action=True),  # future video given clean
}


def sample_mode(mode_probs: Dict[str, float], generator=None) -> Mode:
    """Draw a Mode according to a {name: prob} dict (probs should sum to 1)."""
    import torch

    names = list(mode_probs.keys())
    weights = torch.tensor([mode_probs[n] for n in names], dtype=torch.float32)
    idx = int(torch.multinomial(weights, 1, generator=generator).item())
    return Mode[names[idx].upper()]

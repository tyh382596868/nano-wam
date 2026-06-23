"""Closed-loop evaluation + the FastWAM imagination ablation (M4).

Roll the WAM out as a receding-horizon policy in ReacherEnv: encode the last K
frames, sample an action chunk, execute the first `replan_every` actions, repeat.
Two settings answer "does test-time imagination help?":

    imagine="none"  -> POLICY mode: denoise the action stream only.
    imagine="joint" -> JOINT  mode: denoise future frames AND actions together,
                       then act on the action part.

`run_ablation` runs both over a set of seeded episodes and returns success rate
and mean steps-to-goal. See DESIGN.md §6.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .config import Config
from .envs import ReacherEnv
from .flow import sample
from .modes import Mode
from .tokenizer import to_tokens

IMAGINE_TO_MODE = {"none": Mode.POLICY, "joint": Mode.JOINT}


@torch.no_grad()
def rollout_episode(model, tokenizer, env: ReacherEnv, mode: Mode, cfg: Config,
                    device, max_steps: int):
    """One closed-loop episode. Returns (success: bool, steps: int)."""
    K = cfg.data.ctx_frames
    replan = cfg.eval.replan_every
    frame = env.reset()
    buf = [frame] * K  # rolling window of the last K frames

    steps = 0
    while steps < max_steps:
        ctx = torch.from_numpy(np.stack(buf[-K:])).float().unsqueeze(0).to(device)  # (1,K,3,H,W)
        ctx_tok = to_tokens(tokenizer.encode(ctx))
        out = sample(model, {"ctx": ctx_tok}, mode, cfg.flow)
        actions = out["action"][0].cpu().numpy()  # (Ha, Da)

        for j in range(min(replan, len(actions))):
            frame, done, _ = env.step(actions[j])
            buf.append(frame)
            steps += 1
            if done:
                return True, steps
            if steps >= max_steps:
                break
    return False, steps


def run_ablation(model, tokenizer, cfg: Config, device, episodes: int = None,
                 max_steps: int = None, settings=("none", "joint")) -> Dict[str, dict]:
    """Run the imagination on/off ablation. Returns {setting: {success_rate, mean_steps}}."""
    episodes = episodes or cfg.eval.episodes
    max_steps = max_steps or 6 * cfg.data.action_horizon
    model.eval(); tokenizer.eval()

    results = {}
    for setting in settings:
        mode = IMAGINE_TO_MODE[setting]
        succ, steps_to_goal = [], []
        for ep in range(episodes):
            env = ReacherEnv(cfg.data, seed=1000 + ep)  # same seeds across settings
            ok, steps = rollout_episode(model, tokenizer, env, mode, cfg, device, max_steps)
            succ.append(ok)
            if ok:
                steps_to_goal.append(steps)
        results[setting] = {
            "success_rate": float(np.mean(succ)),
            "mean_steps": float(np.mean(steps_to_goal)) if steps_to_goal else float("nan"),
            "n": episodes,
        }
    return results

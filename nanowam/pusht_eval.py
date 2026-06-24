"""Closed-loop pushT evaluation in the gym-pusht simulator + imagination ablation.

The repo's nanowam/eval.py is hardwired to the procedural ReacherEnv. pushT is a
real dataset with no in-repo simulator, so this module wraps `gym_pusht` behind
the same receding-horizon rollout used for reacher:

  encode the last K frames -> sample an action chunk (policy or joint) ->
  de-normalize it with the dataset's saved action stats -> execute the first
  `replan_every` actions in the sim -> repeat.

pushT actions are absolute target positions in [0, 512]; the model is trained on
*normalized* actions (prepare_pusht_v3.py --normalize, stats in <root>/stats.npz),
so we must de-normalize before stepping. Success = the sim's `is_success` /
coverage >= threshold. `run_pusht_ablation` runs imagine={none,joint} over seeded
episodes and reports success rate, mean coverage, and mean steps.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from .config import Config
from .flow import sample
from .modes import Mode
from .tokenizer import to_tokens

IMAGINE_TO_MODE = {"none": Mode.POLICY, "joint": Mode.JOINT}


def _frame_from_obs(pixels: np.ndarray, size: int) -> np.ndarray:
    """(H,W,3) uint8 -> (3,size,size) float[0,1]."""
    import cv2
    if pixels.shape[0] != size or pixels.shape[1] != size:
        pixels = cv2.resize(pixels, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)).astype(np.float32) / 255.0


def load_action_stats(cfg: Config):
    """Return (mean, std) of shape (Da,) from <root>/stats.npz, or (0,1) if absent."""
    path = os.path.join(cfg.data.root, "stats.npz")
    if os.path.exists(path):
        st = np.load(path)
        return st["action_mean"].reshape(-1), st["action_std"].reshape(-1)
    return np.zeros(cfg.data.action_dim, np.float32), np.ones(cfg.data.action_dim, np.float32)


@torch.no_grad()
def rollout_episode(model, tokenizer, env, mode, cfg, device, mean, std,
                    max_steps, action_low, action_high):
    """One closed-loop pushT episode. Returns (success, steps, best_coverage)."""
    K = cfg.data.ctx_frames
    replan = cfg.eval.replan_every
    size = cfg.data.image_size

    obs, info = env.reset()
    buf = [_frame_from_obs(obs["pixels"], size)] * K
    best_cov = float(info.get("coverage", 0.0))
    steps = 0
    while steps < max_steps:
        ctx = torch.from_numpy(np.stack(buf[-K:])).float().unsqueeze(0).to(device)  # (1,K,3,H,W)
        ctx_tok = to_tokens(tokenizer.encode(ctx))
        out = sample(model, {"ctx": ctx_tok}, mode, cfg.flow)
        actions = out["action"][0].cpu().numpy()                 # (Ha, Da) normalized
        actions = actions * std + mean                           # de-normalize -> abs targets
        actions = np.clip(actions, action_low, action_high)

        for j in range(min(replan, len(actions))):
            obs, rew, term, trunc, info = env.step(actions[j].astype(np.float32))
            buf.append(_frame_from_obs(obs["pixels"], size))
            best_cov = max(best_cov, float(info.get("coverage", 0.0)))
            steps += 1
            if bool(info.get("is_success", False)) or term:
                return True, steps, best_cov
            if trunc or steps >= max_steps:
                return False, steps, best_cov
    return False, steps, best_cov


def run_pusht_ablation(model, tokenizer, cfg, device, episodes=None, max_steps=300,
                       settings=("none", "joint"), obs_size=96, seed0=10000) -> Dict[str, dict]:
    """imagination on/off ablation in gym-pusht. Returns {setting: {...}}."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the env)

    episodes = episodes or cfg.eval.episodes
    mean, std = load_action_stats(cfg)
    model.eval(); tokenizer.eval()

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", observation_width=obs_size,
                   observation_height=obs_size)
    low = np.asarray(env.action_space.low); high = np.asarray(env.action_space.high)

    results = {}
    for setting in settings:
        mode = IMAGINE_TO_MODE[setting]
        succ, steps_to_goal, covs = [], [], []
        for ep in range(episodes):
            env.reset(seed=seed0 + ep)  # same seeds across settings
            ok, steps, cov = rollout_episode(model, tokenizer, env, mode, cfg, device,
                                             mean, std, max_steps, low, high)
            succ.append(ok); covs.append(cov)
            if ok:
                steps_to_goal.append(steps)
        results[setting] = {
            "success_rate": float(np.mean(succ)),
            "mean_coverage": float(np.mean(covs)),
            "mean_steps": float(np.mean(steps_to_goal)) if steps_to_goal else float("nan"),
            "n": episodes,
        }
    env.close()
    return results

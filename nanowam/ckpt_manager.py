"""Checkpoint retention + in-training closed-loop scoring.

Long DLC runs want a *bounded* set of checkpoints that keeps the ones that
actually perform best in the sim — not just the most recent. `CheckpointManager`
keeps the top-`max_keep` checkpoints by score (plus always the latest, so a
preempted job can resume), maintains a `ckpt_best.pt` copy, and appends every
evaluation to `metrics.jsonl`.

`score_checkpoint` runs the closed-loop imagination eval (gym-pusht for the
LeRobot pushT data, the procedural ReacherEnv otherwise) in JOINT mode and
returns a single scalar used for ranking, plus the raw metrics dict.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Tuple


class CheckpointManager:
    def __init__(self, out_dir: str, max_keep: int = 0):
        self.out_dir = out_dir
        self.max_keep = max_keep                       # 0 -> unlimited (keep all)
        self.entries: List[dict] = []                  # {step, path, score} on disk
        self.best_score = float("-inf")
        os.makedirs(out_dir, exist_ok=True)
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")

    def restore(self) -> None:
        """Rebuild state from metrics.jsonl after a resume (keep only files present)."""
        if not os.path.exists(self.metrics_path):
            return
        by_step: Dict[int, dict] = {}
        with open(self.metrics_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = os.path.join(self.out_dir, rec["path"])
                by_step[rec["step"]] = {"step": rec["step"], "path": p,
                                        "score": rec.get("score", float("-inf"))}
        self.entries = [e for e in by_step.values() if os.path.exists(e["path"])]
        if self.entries:
            self.best_score = max(e["score"] for e in self.entries)

    def add(self, step: int, path: str, score: float, extra: Optional[dict] = None) -> None:
        self.entries.append({"step": step, "path": path, "score": float(score)})
        rec = {"step": step, "path": os.path.basename(path), "score": float(score)}
        if extra:
            rec.update({k: extra[k] for k in extra if k != "n"})
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if score > self.best_score:
            self.best_score = score
            try:
                shutil.copyfile(path, os.path.join(self.out_dir, "ckpt_best.pt"))
            except OSError:
                pass
        self._prune()

    def _prune(self) -> None:
        if not self.max_keep or len(self.entries) <= self.max_keep:
            return
        latest_step = max(e["step"] for e in self.entries)
        ranked = sorted(self.entries, key=lambda e: (e["score"], e["step"]), reverse=True)
        keep_paths = {e["path"] for e in ranked[: self.max_keep]}
        # always protect the newest checkpoint so a preempted run can resume
        keep_paths |= {e["path"] for e in self.entries if e["step"] == latest_step}
        for e in list(self.entries):
            if e["path"] not in keep_paths and os.path.exists(e["path"]):
                try:
                    os.remove(e["path"])
                except OSError:
                    pass
        self.entries = [e for e in self.entries if e["path"] in keep_paths]


def score_checkpoint(model, tokenizer, cfg, device, episodes: int,
                     max_steps: int) -> Tuple[float, dict]:
    """Closed-loop JOINT-mode eval -> (scalar score, raw metrics). Higher is better."""
    name = (cfg.data.name or "").lower()
    model.eval()
    tokenizer.eval()
    if "lerobot" in name or "pusht_sim" in name:
        from .pusht_eval import run_pusht_ablation
        res = run_pusht_ablation(model, tokenizer, cfg, device, episodes=episodes,
                                 max_steps=max_steps, settings=("joint",))
        r = res["joint"]
        # success dominates; coverage breaks ties while success is still 0
        score = r["success_rate"] + 1e-3 * r["mean_coverage"]
    else:
        from .eval import run_ablation
        res = run_ablation(model, tokenizer, cfg, device, episodes=episodes,
                           max_steps=max_steps, settings=("joint",))
        r = res["joint"]
        ms = r["mean_steps"]
        # success dominates; fewer steps breaks ties (nan when no success)
        score = r["success_rate"] - (0.0 if ms != ms else 1e-4 * ms)
    return float(score), r

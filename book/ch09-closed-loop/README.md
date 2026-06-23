# Chapter 9 — Closed-Loop Control and the Imagination Ablation

> **This chapter covers**
> - Turning the data generator into a step-able environment
> - Using the WAM as a receding-horizon policy
> - FastWAM's question: does *test-time imagination* help?
> - Measuring success rate with imagination on vs. off

**Builds:** `ReacherEnv`, `rollout_episode`, `run_ablation`.
**Maps to:** `nanowam/envs.py`, `nanowam/eval.py`, `scripts/sample.py --closed-loop`.

## Outline
1. From offline windows to an interactive env (matching the action convention).
2. Receding horizon: sample an action chunk, execute a few, replan.
3. The ablation: `imagine=none` (policy) vs `imagine=joint` (denoise video too).
4. Running it and reading success rate + mean steps.
5. Honest caveat: why nano-CPU numbers are preliminary (revisited in Chapter 12).

## Summary & Exercises
- Summary: the WAM can act, and we can measure whether imagining helps.
- Exercises: vary `replan_every`; add a distance-to-goal reward curve; design a
  task where imagination should help more.

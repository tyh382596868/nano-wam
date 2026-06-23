# Chapter 12 — Scaling, Extensions, and the Frontier

> **This chapter covers**
> - Running nano-wam at scale with one command
> - What is verified (mechanisms) vs. what needs a GPU (quality)
> - Natural extensions: language/goal conditioning, CFG, per-stream time,
>   real-robot deployment
> - How the big research systems go beyond nano-wam

**Builds:** nothing new — you connect what you built to the frontier.
**Maps to:** `scripts/run_gpu.sh`, the repo `PROJECT.md` and `DESIGN.md`.

## Outline
1. The one-click GPU run and what changes at scale (width, depth, steps, AMP).
2. The honesty ledger: mechanisms are CPU-verified; quality/ablation numbers need
   a real run.
3. Extension hooks already in the code (goal embedding, `goal_present`/CFG,
   per-stream flow time) and what they'd unlock.
4. Deploying on a real robot: action de-normalization and the control loop.
5. Back to the frontier: re-reading DreamZero / FastWAM / LingBot-VA / Motus now
   that you've built the core — what each adds on top.

## Summary & Exercises
- Summary: you have a complete, honest WAM and a map of where to go next.
- Exercises: scale `reacher_gpu.yaml` until world prediction beats the baseline;
  pick one reference system and implement one of its ideas on top of nano-wam.

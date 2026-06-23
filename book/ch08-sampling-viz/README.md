# Chapter 8 — Sampling, Decoding, and Visualizing Rollouts

> **This chapter covers**
> - Generating future frames and actions with the trained model
> - Decoding predicted latents back into pixels
> - Building GT-vs-prediction grids and animated rollouts
> - Reading honest baselines (copy-last-frame) before trusting a number

**Builds:** the open-loop visualization path.
**Maps to:** `scripts/sample.py` (the `--mode world/joint/policy` path).

## Outline
1. Sampling in world, joint, and policy modes.
2. From latent tokens back to frames (`from_tokens` → tokenizer decode).
3. Rendering comparisons: PNG grids and GIFs.
4. Metrics with a baseline: why copy-last-frame is a stiff bar for slow motion.
5. Looking at results and forming an honest read.

## Summary & Exercises
- Summary: generate, decode, visualize, and compare to a baseline.
- Exercises: sweep `sample_steps` and watch sharpness; report action MSE in joint
  vs policy mode and interpret the gap.

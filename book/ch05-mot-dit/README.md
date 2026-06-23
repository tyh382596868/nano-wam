# Chapter 5 — Coding the MoT Diffusion Transformer

> **This chapter covers**
> - Embedding three token streams: context, video, and action
> - Self-attention and AdaLN-zero conditioning on the flow time
> - The **Mixture-of-Transformers** block: shared attention, per-stream FFN/AdaLN
> - Assembling `NanoWAM` and why the velocity heads are zero-initialized

**Builds:** `Attention`, `MoTBlock`, `NanoWAM` (the model core).
**Maps to:** `nanowam/model.py`.

## Outline
1. Three streams, one sequence: per-stream input projections + positional embeds.
2. Sinusoidal time embedding and the conditioning vector (mode, goal, flow time).
3. AdaLN-zero: modulation that starts as identity (stable training).
4. Mixture-of-Transformers: shared attention so streams talk; per-stream FFN/AdaLN
   so each modality keeps its own weights.
5. Reading out per-stream velocities; zero-init heads → zero velocity at step 0.

## Summary & Exercises
- Summary: a conditional DiT that maps noised streams to velocities.
- Exercises: toggle `route_attention`; count parameters as you scale `dim`/`depth`;
  verify the zero-init head gives zero output before training.

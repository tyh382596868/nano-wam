# Chapter 6 — Four Modes from One Network

> **This chapter covers**
> - The UniDiffuser idea: which streams are *clean* vs *noised* defines the task
> - The four modes — policy, world, inverse, joint — and their stream masks
> - Sampling a mode per batch so one checkpoint serves them all
> - Wiring modes into the flow loss

**Builds:** `Mode`, `StreamMask`, `MODE_TABLE`, `sample_mode`, `build_stream_inputs`.
**Maps to:** `nanowam/modes.py`, the mode logic in `nanowam/flow.py`.

## Outline
1. One model, many jobs: conditioning by masking.
2. The mode table: what each of policy/world/inverse/joint noises and supervises.
3. Per-mode conditioning subtleties (clean future for inverse, zeros for policy).
4. Drawing a mode per batch and the joint training objective.
5. A quick check that each mode produces a finite, differentiable loss.

## Summary & Exercises
- Summary: masks turn a single generative model into a multi-task WAM.
- Exercises: add a new mode (e.g., action-infilling); change `mode_probs` and
  reason about its effect on each capability.

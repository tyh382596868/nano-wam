# Chapter 11 — Long-Horizon Imagination: Causal Attention & KV Caching

> **This chapter covers**
> - Why full bidirectional attention blocks efficient long rollouts
> - Block-causal masking over interleaved (action_i, frame_i) time-steps
> - Keeping the context *cacheable* (the flow time must not modulate it)
> - The KV cache (`encode_prefix`/`forward_suffix`) and autoregressive dreaming

**Builds:** the causal mask + cached path in `NanoWAM`, `dream_rollout`.
**Maps to:** `nanowam/model.py` (causal), `flow.sample(use_cache=True)`,
`nanowam/rollout.py`, `scripts/sample.py --dream N`.

## Outline
1. The problem: recomputing everything every step is wasteful (and unbounded).
2. Block-causal attention: context visible to all, step i sees steps ≤ i.
3. Making the prefix constant across denoising steps (per-stream conditioning).
4. The cache: compute context K/V once, reuse across Euler steps; prove
   cached == uncached.
5. `dream_rollout`: slide the context window to imagine arbitrarily far.

## Summary & Exercises
- Summary: causality + a KV cache turn fixed-horizon prediction into open-ended
  imagination.
- Exercises: measure the cache speedup as `dim`/context grow; extend the cache to
  retain finalized future steps across a block.

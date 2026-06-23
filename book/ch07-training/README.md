# Chapter 7 — Training nano-wam

> **This chapter covers**
> - The training loop end to end
> - Training the tokenizer by reconstruction while training the WAM by flow
>   matching on **detached** latents (the latent-diffusion split)
> - Optimization details: AdamW, warmup, AMP, gradient clipping
> - Checkpointing and reading the loss curves

**Builds:** the loop in `train.py`; `set_seed`, `save/load_checkpoint`.
**Maps to:** `scripts/train.py`, `nanowam/utils.py`.

## Outline
1. Putting the pieces together: data → tokenize → mode → flow loss → step.
2. Why we `detach` the latents for the flow loss (and what breaks if you don't).
3. Recon loss + flow loss; one optimizer over tokenizer and model.
4. Warmup, AMP, clipping, checkpoints.
5. Sanity check: overfit a tiny fixed batch and watch the loss fall.

## Summary & Exercises
- Summary: a single loop trains the encoder (recon) and the WAM (flow).
- Exercises: remove the detach and observe latent collapse; plot per-mode loss
  components; resume from a checkpoint.

# nano-wam

A minimal, from-scratch, single-GPU **World Action Model (WAM)** — the
`nanoGPT` of world-action models.

A WAM jointly predicts the **future of the world** (next frames) and the
**action** that drives it. nano-wam distills the core ideas from recent
research systems — DreamZero, FastWAM, LingBot-VA, and Motus — into a tiny,
readable PyTorch codebase you can train on a toy task on one GPU (or CPU).

> **Status: M2.** Tokenizer, MoT DiT, rectified-flow loss/sampler, all four
> modes, the procedural data pipeline, and the training loop are implemented and
> tested (incl. a policy-mode overfit). Still to come: scaled multi-mode training
> and the eval/imagination-ablation harness (`scripts/sample.py`). See the
> roadmap in `DESIGN.md §9`.

## The idea in one diagram

```
history frames ─┐
goal / language ─┤──►  MoT Diffusion Transformer  ──►  future frames  (world)
(noised future) ─┘     shared attention, per-stream  ──►  action chunk   (policy)
(noised action) ─┘     FFN + flow-matching objective
```

One network, trained with a UniDiffuser-style mode mask, gives you several
inference modes from a single checkpoint: **policy**, **world model**,
**inverse dynamics**, and **joint** forecasting — plus a built-in toggle to
test FastWAM's question: *does test-time imagination actually help?*

## What it borrows from the references

- **DiT + flow matching** core — from all four.
- **Mixture-of-Transformers** (shared attention, per-stream weights) — LingBot-VA, Motus.
- **Imagination-free deployment switch** — FastWAM.
- **Multi-mode single checkpoint** — Motus / UniDiffuser.

What it deliberately omits: 5B–14B params, Wan/Qwen pretrained weights,
multi-GPU infra, real robot stacks. See `DESIGN.md §1`.

## Quickstart (target API — not yet runnable)

```bash
pip install -r requirements.txt

# 1. build toy windows from the self-contained procedural "reacher" task
python scripts/prepare_data.py --config configs/pusht_tiny.yaml --episodes 200

# 2. train the tiny WAM  (tokenizer recon + flow on detached latents)
python scripts/train.py --config configs/pusht_tiny.yaml

# 3. evaluate; toggle test-time imagination on/off   (M3/M4 — wip)
python scripts/sample.py --config configs/pusht_tiny.yaml --imagine none
python scripts/sample.py --config configs/pusht_tiny.yaml --imagine joint
```

## Layout

```
nanowam/    core library — config, tokenizer, model (MoT DiT), flow, modes, data
configs/    tiny run configs
scripts/    prepare_data / train / sample
tests/      shape-contract tests
DESIGN.md   the spec — read this first
```

## References

- DreamZero — https://github.com/dreamzero0/dreamzero
- FastWAM — https://github.com/yuantianyuan01/FastWAM
- LingBot-VA — https://github.com/Robbyant/lingbot-va
- Motus — https://github.com/thu-ml/Motus

## License

MIT (see `LICENSE`).

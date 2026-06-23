# nano-wam

[![CI](https://github.com/tyh382596868/nano-wam/actions/workflows/ci.yml/badge.svg)](https://github.com/tyh382596868/nano-wam/actions/workflows/ci.yml)

A minimal, from-scratch, single-GPU **World Action Model (WAM)** — the
`nanoGPT` of world-action models.

A WAM jointly predicts the **future of the world** (next frames) and the
**action** that drives it. nano-wam distills the core ideas from recent
research systems — DreamZero, FastWAM, LingBot-VA, and Motus — into a tiny,
readable PyTorch codebase you can train on a toy task on one GPU (or CPU).

> **Status: M5 (adapter).** Full nano stack: tokenizer, MoT DiT, rectified-flow
> loss/sampler, all four modes, procedural data + interactive `ReacherEnv`,
> training loop, open-loop rollout viz, the `--closed-loop` **imagination
> ablation** (FastWAM's question at nano scale), and a **LeRobot real-data
> adapter** (pushT / LIBERO slices via `sources.py`). Open stretch: KV-cache
> long-horizon rollout + a GPU-scale run for trustworthy numbers. See
> `DESIGN.md §9`.

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

# 3a. visualize predicted futures (open-loop): GT-vs-pred rollout gif + grid
python scripts/sample.py --config configs/pusht_tiny.yaml --mode world

# 3b. closed-loop imagination ablation: does test-time imagination help?
python scripts/sample.py --config configs/pusht_tiny.yaml --closed-loop --episodes 50
```

**Real data (LeRobot).** Any LeRobot-format dataset (pushT, LIBERO slices, …)
drops in behind the same interface — frames are resized, actions checked against
`action_dim`, episodes pooled into shards (with optional normalization):

```bash
pip install lerobot
python scripts/prepare_data.py --config configs/lerobot_pusht.yaml \
    --source lerobot --repo-id lerobot/pusht --normalize
python scripts/train.py --config configs/lerobot_pusht.yaml
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

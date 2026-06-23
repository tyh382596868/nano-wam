# nano-wam

[![CI](https://github.com/tyh382596868/nano-wam/actions/workflows/ci.yml/badge.svg)](https://github.com/tyh382596868/nano-wam/actions/workflows/ci.yml)

A minimal, from-scratch, single-GPU **World Action Model (WAM)** — the
`nanoGPT` of world-action models.

A WAM jointly predicts the **future of the world** (next frames) and the
**action** that drives it. nano-wam distills the core ideas from recent
research systems — DreamZero, FastWAM, LingBot-VA, and Motus — into a tiny,
readable PyTorch codebase you can train on a toy task on one GPU (or CPU).

> **Status: M6.** Full nano stack: tokenizer, MoT DiT, rectified-flow
> loss/sampler, all four modes, procedural data + interactive `ReacherEnv`,
> training loop, open-loop rollout viz, the `--closed-loop` **imagination
> ablation** (FastWAM's question at nano scale), a **LeRobot real-data adapter**
> (pushT / LIBERO slices), and **KV-cache autoregressive long-horizon rollout**
> (`causal` model + `--dream N`, LingBot-VA path). Open stretch: a GPU-scale run
> for trustworthy numbers. See `DESIGN.md §9`.

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

## Features (all implemented + tested)

| Capability | Where | From |
|---|---|---|
| MoT Diffusion Transformer (shared attn, per-stream FFN/AdaLN) | `model.py` | LingBot-VA, Motus |
| Rectified flow matching, joint video+action | `flow.py` | all four |
| 4 modes from one checkpoint (policy / world / inverse / joint) | `modes.py` | Motus / UniDiffuser |
| Imagination on/off ablation (`--closed-loop`) | `eval.py` | FastWAM |
| LeRobot real-data adapter (pushT / LIBERO) | `sources.py` | — |
| Causal KV-cache long-horizon dreaming (`--dream N`) | `rollout.py` | LingBot-VA |

> **Verified vs. not:** every *mechanism* is CPU-tested (20 tests, CI green).
> *Quality* numbers are not yet trustworthy — all runs so far are tiny/undertrained
> on CPU. Trustworthy quality + ablation numbers need a GPU run (below). See
> `PROJECT.md` for the honest breakdown.

## Quickstart

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

**Long-horizon dreaming (KV-cache, causal model).** Train the causal variant,
then autoregressively imagine far past the training horizon:

```bash
python scripts/prepare_data.py --config configs/causal_tiny.yaml --episodes 200
python scripts/train.py        --config configs/causal_tiny.yaml
python scripts/sample.py       --config configs/causal_tiny.yaml --dream 8   # 8 blocks → long gif
```

**GPU (one-click).** Scaled config + a single script that does deps → data →
train → visualize → ablation (and a long-horizon dream if the config is causal):

```bash
./scripts/run_gpu.sh                          # config: configs/reacher_gpu.yaml
EPISODES=4000 ./scripts/run_gpu.sh            # more data
./scripts/run_gpu.sh configs/causal_tiny.yaml # any config
```

## Documentation

- [`PROJECT.md`](PROJECT.md) — why this exists, how it's built, the milestone
  story, and what is / isn't verified. **Start here to understand the project.**
- [`DESIGN.md`](DESIGN.md) — the precise architecture spec (shapes, modes, block
  definition, roadmap).
- [`CLAUDE.md`](CLAUDE.md) — quick orientation + conventions for coding agents.

## Layout

```
nanowam/    core library
  model.py      MoT DiT (+ causal mask, KV cache)      flow.py    rectified flow
  tokenizer.py  conv AE frame tokenizer                modes.py   modes + masks
  data.py       windows + procedural reacher           sources.py EpisodeSource adapters
  envs.py       ReacherEnv                             eval.py    closed-loop + ablation
  rollout.py    autoregressive long-horizon dream      config.py  config dataclasses
scripts/    prepare_data / train / sample / run_gpu.sh
configs/    pusht_tiny (CPU) · reacher_gpu · lerobot_pusht · causal_tiny
tests/      executable shape + behavior contracts
```

## References

- DreamZero — https://github.com/dreamzero0/dreamzero
- FastWAM — https://github.com/yuantianyuan01/FastWAM
- LingBot-VA — https://github.com/Robbyant/lingbot-va
- Motus — https://github.com/thu-ml/Motus

## License

MIT (see `LICENSE`).

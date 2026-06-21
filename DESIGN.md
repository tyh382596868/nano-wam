# nano-wam — Design

A minimal, single-GPU, hackable **World Action Model (WAM)**. The goal is the
same as `nanoGPT`'s relationship to GPT: strip a research-grade system down to
the smallest thing that still embodies the core idea, so it can be read in an
afternoon and trained on a toy task in minutes.

This document is the spec. Code in this repo is currently **scaffold + stubs**;
each module's contract is defined here and mirrored in its docstring.

---

## 1. What is a World Action Model?

A WAM jointly models the **world** (future observations) and the **action** that
drives the transition. Given a short history of observations `o_{≤t}` and a goal
/ language instruction `g`, it predicts:

- the **future frames** `o_{t+1:t+H_v}` (the "world model" / imagination), and
- the **action chunk** `a_{t:t+H_a}` (the "policy").

The bet, shared by all the systems we surveyed, is that learning to *imagine the
future* is a strong auxiliary signal for *acting*, and that one network can do
both.

### Survey distilled (the 4 reference projects)

| Project | Backbone | Action mechanism | Headline idea |
|---|---|---|---|
| DreamZero (NVIDIA GEAR) | Wan2.1-I2V-14B DiT | Joint video+action diffusion | WAMs are zero-shot policies |
| FastWAM | ActionDiT (Wan2.2 DiT) | Diffusion action, *no* video rollout at deploy | Do WAMs need test-time imagination? |
| LingBot-VA | Dual-stream MoT + Wan2.2 VAE | Interleaved autoregressive video-action, KV cache | Causal video-action world modeling |
| Motus (THU-ML) | MoT: Wan VGM + Qwen-VL + experts | Latent (optical-flow) actions, UniDiffuser scheduler | Unified latent-action world model |

**Common DNA we keep in nano-wam:**

1. A **Diffusion Transformer (DiT)** core.
2. **Mixture-of-Transformers (MoT)**: separate per-modality weights, *shared*
   self-attention — so video and action tokens attend to each other but use
   their own FFN/normalization.
3. **Flow matching** (rectified flow) as the generative objective.
4. A **UniDiffuser-style** masking scheme that turns one trained network into
   several inference modes (world model / policy / inverse dynamics / joint).
5. The **FastWAM ablation built in**: an "imagination-free" inference switch so
   you can measure whether rolling out video at test time actually helps.

**What we drop (the "nano" budget):** billions of params, Wan/Qwen pretrained
weights, multi-GPU infra, real robot stacks, optical-flow latent-action
pretraining. nano-wam is from-scratch PyTorch on a toy dataset.

---

## 2. System overview

```
                  language/goal g ──► text embed (frozen tiny / one-hot)
                                              │
  history frames o_{t-k..t} ──► frame tokenizer ──► context tokens ─┐
                                                                    ▼
  future frames  (noised x_v) ──► patchify ──► video tokens ──►┌──────────┐
                                                               │  MoT DiT │──► v_video (velocity)
  action chunk   (noised x_a) ──► proj      ──► action tokens ─►│  (N blk) │──► v_action (velocity)
                                                               └──────────┘
                              flow time τ, mode mask ──► AdaLN-zero conditioning
```

- **Frame tokenizer** (`nanowam/tokenizer.py`): a tiny conv encoder/decoder that
  maps a low-res RGB frame to a small latent grid and back. This is nano-wam's
  stand-in for the Wan VAE. Default: 64×64 → 8×8×C latent.
- **MoT DiT** (`nanowam/model.py`): the core. Concatenates context + video +
  action tokens into one sequence, runs `N` MoT blocks (shared attention,
  per-stream FFN + AdaLN), and reads out per-stream velocity predictions.
- **Flow matching** (`nanowam/flow.py`): rectified-flow noising, loss, and an
  Euler ODE sampler.
- **Modes** (`nanowam/modes.py`): which streams are clean vs. noised.

---

## 3. Tensors & shapes (the contract)

Symbols: `B` batch, `Hv` video horizon (frames predicted), `Ha` action horizon,
`Da` action dim, `P` patches per frame, `C` latent channels, `d` model width.

| Name | Shape | Meaning |
|---|---|---|
| `ctx_frames` | `(B, K, 3, 64, 64)` | history observations (K frames) |
| `future_frames` | `(B, Hv, 3, 64, 64)` | targets for the world stream |
| `actions` | `(B, Ha, Da)` | target action chunk |
| `goal` | `(B, Lg)` int tokens / `(B, d)` embed | language / goal |
| latent `z_v` | `(B, Hv*P, C)` | tokenized future frames |
| latent `z_a` | `(B, Ha, d)` | projected actions |
| flow time `τ` | `(B,)` in `[0,1]` | shared or per-stream |
| velocity out | matches `z_v`, `z_a` | rectified-flow target `x1 - x0` |

Defaults (toy / single-GPU): `d=256`, `N=8` blocks, `heads=8`, `Hv=4`, `Ha=8`,
`Da=2` (pushT) , `C=4`, frame `64×64`, patch `8` → `P=64`. ~10–20M params.

---

## 4. MoT block

```
x = [ctx_tokens ; video_tokens ; action_tokens]      # shared sequence
for each block:
    # shared self-attention across ALL tokens (this is where world↔action talk)
    h = x + Attn( AdaLN_norm(x, τ, mode) )
    # per-stream FFN: video tokens use FFN_v, action tokens use FFN_a, ctx FFN_c
    h = h + StreamFFN( AdaLN_norm(h, τ, mode) )
    x = h
```

- **MoT = modality-routed weights.** A boolean stream-id per token selects which
  `{FFN, AdaLN, in/out proj}` set applies. Attention QKV can be shared or routed;
  nano-wam shares attention and routes FFN+AdaLN by default (smallest variant
  that still "is" an MoT).
- **AdaLN-zero** conditioning carries `(flow_time τ, mode embedding, goal embed)`.
  Initialized to zero so blocks start as identity (stable training).
- **Context tokens are clean** (never noised); they're the conditioning image.

---

## 5. Training objective — joint rectified flow

For a sample, draw `τ ~ U(0,1)`, Gaussian `ε`, and form the interpolant per
stream `x_τ = (1-τ)·ε + τ·x_1`, target velocity `u = x_1 - ε`. The network
predicts `v_θ`. Loss:

```
L = λ_v · ‖v_v − u_v‖²  +  λ_a · ‖v_a − u_a‖²
```

Streams selected by the **mode mask** are noised & supervised; unselected streams
are passed in clean as conditioning and not supervised.

### Modes (one network, many jobs)

| Mode | Clean (conditioning) | Noised (predicted) | Use |
|---|---|---|---|
| `joint` | ctx, goal | video + action | full WAM training |
| `policy` | ctx, goal, *(video clean or absent)* | action | act without imagining |
| `world` | ctx, goal, action | video | pure world model |
| `inverse` | ctx, future video | action | inverse dynamics |

Train by **sampling a mode per batch** (UniDiffuser-style) so all inference
modes are supported by the single checkpoint.

---

## 6. Inference

- **Sampler**: Euler ODE on the flow, integrate `τ: 0→1` over `S` steps
  (default `S=10`). `nanowam/flow.py:sample`.
- **Policy deployment**: run `policy` mode → action chunk only. Execute first
  `n` actions, replan (receding horizon).
- **Imagination toggle** (the FastWAM question): `--imagine {none,joint}`.
  - `none`: sample actions in `policy` mode, video stream never denoised.
  - `joint`: denoise video + action together, return both.
  The eval harness reports success rate for both so you can measure the cost/
  benefit of test-time imagination on the toy task.

---

## 7. Toy dataset

Default target: **pushT** (2D, `Da=2`) — tiny, renders to frames, has expert
demos, trains on one GPU/CPU. `nanowam/data.py` yields
`(ctx_frames, future_frames, actions, goal)` windows. A `prepare_data.py` stub
converts an episode source (pushT, or a small LeRobot/LIBERO slice) into sharded
`.npz`/`.pt` windows.

Stretch: a thin LeRobot adapter so a LIBERO subset drops in unchanged.

---

## 8. Repo layout

```
nano-wam/
├── DESIGN.md                 # this file (the spec)
├── README.md                 # quickstart + project pitch
├── requirements.txt
├── configs/
│   └── pusht_tiny.yaml       # the default tiny run
├── nanowam/
│   ├── __init__.py
│   ├── config.py             # dataclasses for model/train/data
│   ├── tokenizer.py          # tiny conv frame VAE (stub)
│   ├── model.py              # MoT DiT (stub)
│   ├── flow.py               # rectified flow: noise/loss/sample (stub)
│   ├── modes.py              # mode masks (stub)
│   ├── data.py               # toy windowed dataset (stub)
│   └── utils.py              # seed, ckpt, logging helpers (stub)
├── scripts/
│   ├── prepare_data.py       # episodes → windows (stub)
│   ├── train.py              # train loop (stub)
│   └── sample.py             # inference / eval, imagination toggle (stub)
└── tests/
    └── test_shapes.py        # forward-pass shape contracts (stub)
```

---

## 9. Roadmap

1. **M0 (this commit)** — design + scaffold + shape contracts.
2. **M1** — tokenizer + model forward pass green in `test_shapes.py`.
3. **M2** — flow loss + single-mode (`policy`) overfit on a tiny pushT slice.
4. **M3** — multi-mode training; `world` + `joint` sampling renders plausible
   futures.
5. **M4** — eval harness + the imagination on/off ablation (the FastWAM result,
   reproduced at nano scale).
6. **M5 (stretch)** — LeRobot/LIBERO adapter; KV-cache autoregressive rollout
   (the LingBot-VA path).

---

## 10. Open questions

- Share attention QKV across streams, or route them too (fuller MoT)?
- Per-stream flow time `τ` vs. shared (per-stream enables cleaner inverse
  dynamics / partial-noise modes).
- Tokenizer: train jointly, pretrain+freeze, or operate on raw downsampled
  pixels for ultra-nano?

These are deliberately left for M1+ once the scaffold is agreed.

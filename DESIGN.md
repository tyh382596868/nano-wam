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

- **Frame tokenizer** (`nanowam/tokenizer.py`): a tiny **deterministic conv
  autoencoder** (no KL/VAE sampling — that's overkill at nano scale) mapping a
  low-res RGB frame to a small latent grid and back. nano-wam's stand-in for the
  Wan VAE. Default: `64×64×3` → `8×8×C` latent. The **`patch_size` is exactly the
  tokenizer's downsample factor** (8), so each latent grid cell *is* one token
  (`P = (64/8)² = 64` tokens/frame, each `C`-dim) — there is no second patchify
  step. The tokenizer is trained **jointly** with the WAM (M2); a recon-only
  warmup is optional.
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
`Da=2` (pushT) , `C=4`, frame `64×64`, patch `8` → `P=64`. **≈24.5M params** for
the WAM (the per-stream FFN ×3 dominates) plus **≈0.36M** for the tokenizer.

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

- **MoT = modality-routed weights.** A stream-id per token (`0=ctx, 1=video,
  2=action`) selects which `{LayerNorm, AdaLN modulation, FFN}` set applies. The
  three streams share one self-attention (default `route_attention: false`); when
  `route_attention: true`, QKV is routed too (fuller MoT, more params). Streams
  are contiguous blocks in the sequence, so routing is a per-stream slice +
  scatter.
- **Conditioning vector** `c ∈ ℝ^d` (one per batch element) is
  `c = MLP_time(sinusoid(τ)) + Emb_mode(mode) + goal_embed(goal)`, with the goal
  term zero when `goal=None`. Each stream owns its **AdaLN-zero** modulation MLP
  `c → (shift₁,scale₁,gate₁,shift₂,scale₂,gate₂)`, the two gates **zero-init** so
  every block starts as identity (stable training). Pre-norm DiT layout:
  `x += gate₁·Attn(mod(LN₁(x),shift₁,scale₁))` then
  `x += gate₂·FFN(mod(LN₂(x),shift₂,scale₂))`.
- **Embeddings.** Per-stream input projections (`Linear(C→d)` for ctx/video,
  `Linear(Da→d)` for action) plus a learned positional table sized to that
  stream's token count; per-stream output heads (`Linear(d→C)`, `Linear(d→Da)`)
  read out velocities.
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

**Runnable default: a self-contained procedural "reacher"** (`nanowam/data.py`) —
a white agent square moves toward a gray goal; renders to 64×64 frames with a 2D
per-step displacement action (`Da=2`). It needs **no external sim**, so training
and CI run anywhere. `scripts/prepare_data.py` generates episodes, slides
`(ctx, future, action)` windows, and writes `.npz` shards; `WindowDataset` reads
them.

**Real targets (adapters, M5):** pushT and a LeRobot/LIBERO slice plug in behind
the same shard/window interface — same `(ctx_frames, future_frames, actions)`
contract, just a different generator.

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

1. **M0** ✅ — design + scaffold + shape contracts.
2. **M1** ✅ — tokenizer (conv AE) + MoT DiT forward; shape tests green, backward
   verified through tokenizer→model, zero-init heads give 0 velocity at init.
3. **M2** ✅ — rectified-flow loss + Euler sampler, all 4 modes, `sample_mode`,
   procedural reacher data pipeline, and the train loop (tokenizer recon + flow
   on detached latents). Tests: flow loss differentiable in every mode, sampler
   shapes, and a `policy`-mode overfit that drives loss down on a fixed batch.
4. **M3** ✅ — multi-mode training (all modes sampled per batch); `scripts/sample.py`
   denoises future-frame latents and decodes them to pixels, dumping a GT-vs-pred
   rollout gif + grid and reporting frame/action MSE against a copy-last-frame
   baseline.
5. **M4** ✅ — `ReacherEnv` + closed-loop receding-horizon rollout; the
   `--closed-loop` imagination on/off ablation (`imagine=none` POLICY vs
   `imagine=joint` JOINT) reports success rate + mean steps. Harness tested;
   real numbers need GPU-scale training (the nano-CPU checkpoint is undertrained).
6. **M5 (next, stretch)** — LeRobot/LIBERO adapter; KV-cache autoregressive
   rollout (the LingBot-VA path).

---

## 10. Resolved design decisions

The scaffold's open questions are now locked for the M1 implementation:

- **Attention sharing** → **shared QKV** across streams (route FFN + AdaLN only).
  Smallest variant that still "is" an MoT; `route_attention: true` is kept as an
  opt-in escape hatch for experiments.
- **Flow time `τ`** → **shared across streams** (`per_stream_time: false`).
  Per-stream `τ` (cleaner partial-noise / inverse-dynamics) is deferred; the API
  already accepts `τ` of shape `(B,)` or `(B, n_streams)` so enabling it later is
  non-breaking.
- **Tokenizer** → **tiny deterministic conv AE, trained jointly** (no KL, no
  freeze). `patch_size` = downsample factor, so latent cells are the tokens. An
  ultra-nano "pixel passthrough" mode (identity tokenizer on downsampled pixels)
  may be added behind a flag for debugging.

Still genuinely open, deferred past M1:

- Classifier-free guidance on the goal (the `goal_present` hook in `StreamMask`
  exists but is unused until we have a real multi-task goal).
- Whether `inverse` mode needs its own time schedule vs. reusing `policy`.

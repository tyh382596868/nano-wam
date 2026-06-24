# nano-wam — Project narrative

> Orientation for anyone (human or agent) picking this repo up cold. Read this
> for the *why* and the *story*; read [`DESIGN.md`](DESIGN.md) for the precise
> *spec*; read [`README.md`](README.md) for *how to run it*.

## Why this exists

**World Action Models (WAMs)** are the current frontier of robot learning: one
network that jointly predicts the *future of the world* (next video frames) and
the *action* that drives the transition. The bet is that learning to imagine the
future is a strong signal for learning to act, and that a single model can do
both — giving zero-shot-ish generalization across tasks.

The recent research systems that motivated this project:

| Project | One-line idea |
|---|---|
| **DreamZero** (NVIDIA GEAR) | WAMs are zero-shot policies — predict video + action jointly |
| **FastWAM** | Do WAMs actually need test-time future imagination? |
| **LingBot-VA** | Causal interleaved video-action world modeling with KV cache |
| **Motus** (THU-ML) | Unified latent-action world model, one net / many modes |

They are all impressive and all **huge**: 5B–14B parameters, Wan/Qwen pretrained
backbones, multi-GPU infra, real robot stacks. That makes the *ideas* hard to
see through the engineering.

**nano-wam is the `nanoGPT` of world-action models.** The goal is the smallest
thing that still genuinely embodies the core ideas, so it can be read in an
afternoon and trained on a toy task on a single GPU (or CPU). It is a teaching /
research-prototyping artifact, not a SOTA system.

## What the references have in common (and what we kept)

Distilling the four systems, the shared DNA is:

1. A **Diffusion Transformer (DiT)** core.
2. **Mixture-of-Transformers (MoT)**: separate per-modality weights, *shared*
   self-attention, so video and action tokens attend to each other but use their
   own FFN/normalization.
3. **Flow matching** (rectified flow) as the generative objective.
4. A **UniDiffuser-style** mode mask: one trained network → many inference modes
   (policy / world-model / inverse-dynamics / joint).
5. (LingBot-VA) **causal, KV-cached** autoregressive rollout for long horizons.

nano-wam keeps all five. It drops the billions of params, the pretrained
backbones, the multi-GPU infra, the real robot stack, and the optical-flow
latent-action pretraining.

## How it's built (the architecture in one breath)

```
history frames ─┐                          ┌─► future frames (world stream)
goal / mode     ┤─► MoT Diffusion Transformer ┤
(noised future) ┤   shared attention,         └─► action chunk  (policy stream)
(noised action) ┘   per-stream FFN + AdaLN-zero,
                    rectified-flow objective
```

- A tiny **conv autoencoder** tokenizes 64×64 frames to an 8×8×C latent grid
  (the Wan-VAE stand-in); each latent cell is one token.
- Three token streams — **context / video / action** — are concatenated and run
  through `N` **MoT blocks** (shared attention, per-stream FFN + AdaLN-zero).
- Training is **joint rectified flow**: noise the streams a randomly-sampled
  **mode** selects, predict the velocity, MSE. Sampling modes per batch makes one
  checkpoint serve policy / world / inverse / joint.
- Inference is an **Euler ODE** over the flow. A `--imagine {none,joint}` toggle
  reproduces FastWAM's question; a **causal** variant adds block-causal attention
  + a **KV cache** for arbitrarily long autoregressive "dreaming".

The win that ties it together: because the data, env, training, sampling, eval,
and real-data adapters all speak the same `(ctx, future, action)` window
contract, every piece composes without special-casing.

## The build, milestone by milestone (all ✅, CPU-verified)

- **M0** — design doc + repo scaffold + shape contracts.
- **M1** — tokenizer (conv AE) + MoT DiT forward; zero-init heads → 0 velocity at
  init; backward verified through tokenizer→model.
- **M2** — rectified-flow loss + Euler sampler, all 4 modes, `sample_mode`, the
  procedural reacher data pipeline, the train loop (tokenizer recon + flow on
  *detached* latents). A policy-mode overfit drives loss down.
- **M3** — multi-mode training + `sample.py`: decode predicted future latents to
  pixels, dump GT-vs-pred rollout gifs, report MSE vs a copy-last-frame baseline.
- **M4** — `ReacherEnv` + closed-loop receding-horizon rollout; the
  `--closed-loop` imagination on/off ablation (success rate + mean steps).
- **M5** — `EpisodeSource` abstraction + `LeRobotSource` (lazy dep): any
  LeRobot-format dataset (pushT, LIBERO slices) feeds the *same* shard/window
  pipeline; optional action normalization with saved stats.
- **M6** — KV-cache autoregressive long-horizon rollout (LingBot-VA path):
  block-causal attention, context-KV caching (the flow time stops modulating the
  context so its K/V are constant across Euler steps), `dream_rollout`. Cached ==
  uncached to atol 1e-5.

CI (GitHub Actions, CPU torch) runs the full suite on every push.

## What's verified vs. what isn't (read this before trusting numbers)

- **Verified (CPU):** every mechanism. Shapes, gradients, flow-loss in all modes,
  the policy overfit, the KV-cache numerical equivalence, causality, end-to-end
  train→sample→eval→dream. 20 tests, all green.
- **NOT yet verified:** *quality*. All training so far is a few hundred CPU steps
  on tiny models — deliberately undertrained. Concretely: M3 world prediction
  does **not** beat the copy-last-frame baseline yet, and the M4 ablation numbers
  are noisy. The reacher's small per-step motion also makes "copy last frame" a
  strong baseline. **Getting trustworthy quality/ablation numbers needs a
  GPU-scale run** (see `scripts/run_gpu.sh`). The sandbox this was built in had no
  GPU, so that step is intentionally left for a machine that has one.

## How to extend it

- **Run at scale:** `./scripts/run_gpu.sh` (config `configs/reacher_gpu.yaml`).
- **Real data:** `configs/lerobot_pusht.yaml` + `--source lerobot` (M5).
- **Long-horizon dreams:** any `causal: true` config + `sample.py --dream N` (M6).
- **New dataset:** implement an `EpisodeSource` (see `nanowam/sources.py`); the
  rest of the stack is unchanged.
- **Open stretch items:** real GPU-scale numbers; a richer goal/language encoder
  (the `goal_embed` hook + classifier-free guidance `goal_present` flag exist but
  are unused); per-stream flow time; a real robot deploy loop with action
  de-normalization.

## Map of the repo

`nanowam/` is the library, `scripts/` the entry points, `configs/` the runs,
`tests/` the contracts. The annotated layout lives in `DESIGN.md §8`. Start at
`nanowam/model.py` (the core) and `nanowam/flow.py` (the objective).

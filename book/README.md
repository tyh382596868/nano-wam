# Build a World Action Model (From Scratch)

> A hands-on book that builds **nano-wam** — a minimal World Action Model — one
> component at a time, in the spirit of Sebastian Raschka's *Build a Large
> Language Model (From Scratch)*. You write every line: the tokenizer, the
> Mixture-of-Transformers diffusion model, the flow-matching trainer, the
> closed-loop controller, and a causal KV-cached long-horizon imaginer.

## Who this is for

You know Python and basic PyTorch and have seen a transformer before. You do
**not** need a background in robotics, diffusion models, or world models — we
build the ideas from the ground up. By the end you will have, from scratch, a
working WAM that predicts future video frames and actions jointly, and you will
understand *why* each piece is there.

## How the book works

- **One chapter, one folder.** Each `chNN-*/` directory is a self-contained
  chapter: a `README.md` (the prose + code listings) and, where useful, a
  `code/` folder with the runnable snippets for that chapter.
- **Incremental.** Every chapter adds one component and ends with something you
  can run. The components accumulate into the full `nanowam/` library that lives
  at the repo root — the book is the narrated construction of that codebase.
- **Style (mirrors the reference book):** each chapter opens with a *"This
  chapter covers"* box, develops ideas with figures (described in text) and small
  code listings, and closes with a *Summary* and *Exercises*.
- **Honest.** Every claim is backed by code you can run. Where quality needs a
  GPU we say so (see Chapter 12).

## Table of contents

### Part I — Foundations
1. **Understanding World Action Models** — what world models and policies are,
   why unify them, a tour of four research WAMs, and the nano philosophy.
2. **The Window Contract: Data, Episodes, and the Reacher World** — the
   `(context, future, action)` window that the whole system speaks; a
   self-contained procedural task.
3. **From Frames to Latent Tokens: The Tokenizer** — a tiny conv autoencoder and
   why we model in latent space.

### Part II — The Model
4. **Rectified Flow Matching** — generative modeling as learning a velocity
   field; the interpolant, the loss, the ODE sampler.
5. **Coding the MoT Diffusion Transformer** — attention, AdaLN-zero, and the
   Mixture-of-Transformers block; assembling `NanoWAM`.
6. **Four Modes from One Network** — UniDiffuser-style masks turn one model into
   policy / world-model / inverse-dynamics / joint.

### Part III — Training & Behavior
7. **Training nano-wam** — the loop: tokenizer reconstruction + flow on detached
   latents, optimization, checkpoints.
8. **Sampling, Decoding, and Visualizing Rollouts** — generate, decode latents to
   pixels, and read honest baselines.
9. **Closed-Loop Control and the Imagination Ablation** — a step-able
   environment, receding-horizon control, and FastWAM's question.

### Part IV — Scaling Up
10. **Real Data: The LeRobot Adapter** — ingest real robot datasets behind the
    same window contract.
11. **Long-Horizon Imagination: Causal Attention & KV Caching** — block-causal
    attention and a KV cache for arbitrarily long autoregressive dreams.
12. **Scaling, Extensions, and the Frontier** — the GPU run, what's verified vs
    not, and how the big systems go further.

### Appendices
- **A — Setup & PyTorch Essentials**
- **B — The Math of Rectified Flow**

## Status

**Complete** — all 12 chapters and both appendices are written, each with a
runnable `01_main-chapter-code/` demo (for the code chapters) that was executed
end to end before commit. Track the chapter-to-code mapping in the table below.

| Ch | Title | Builds (repo file) | Status |
|----|-------|--------------------|--------|
| 1  | Understanding World Action Models | (concepts) | ✅ written |
| 2  | The Window Contract | `data.py`, `prepare_data.py` | ✅ written |
| 3  | The Tokenizer | `tokenizer.py` | ✅ written |
| 4  | Rectified Flow Matching | `flow.py` | ✅ written |
| 5  | The MoT Diffusion Transformer | `model.py` | ✅ written |
| 6  | Four Modes from One Network | `modes.py` | ✅ written |
| 7  | Training nano-wam | `train.py`, `utils.py` | ✅ written |
| 8  | Sampling & Visualizing | `sample.py` | ✅ written |
| 9  | Closed-Loop Control & Ablation | `envs.py`, `eval.py` | ✅ written |
| 10 | Real Data: LeRobot Adapter | `sources.py` | ✅ written |
| 11 | Causal Attention & KV Caching | `model.py`, `rollout.py` | ✅ written |
| 12 | Scaling & the Frontier | `run_gpu.sh` | ✅ written |
| A  | Setup & PyTorch | — | ✅ written |
| B  | The Math of Rectified Flow | — | ✅ written |

The finished code for every chapter already exists at the repo root (`nanowam/`),
so the book is reverse-engineering a working system into a teachable path. See
the repo's `PROJECT.md` and `DESIGN.md` for the spec the book narrates.

## Conventions (and how they relate to the reference book)

This book is modeled on **Sebastian Raschka's *Build a Large Language Model (From
Scratch)*** — https://github.com/rasbt/LLMs-from-scratch — and mirrors its
pedagogy: each chapter opens with a *"This chapter covers"* box, develops ideas
with figures and small code listings, and closes with a *Summary* and
*Exercises*. We adopt its conventions, with two deliberate adaptations:

| Reference repo (`rasbt/LLMs-from-scratch`) | This book | Why |
|---|---|---|
| `ch01`, `ch02`, … folders | `chNN-<slug>/` (slug for readability) | same idea, clearer names |
| `01_main-chapter-code/` with runnable code | `01_main-chapter-code/` runnable demo per code chapter | same convention |
| `ch02.ipynb` Jupyter notebooks | `README.md` prose + listings | git-reviewable; the *finished* code lives in `nanowam/` |
| `exercise-solutions.ipynb` | exercises at each chapter end; solutions appendix (planned) | same |
| `appendix-A … E` | `appendix-a-*`, `appendix-b-*` | same idea |
| `setup/` | repo `CLAUDE.md` + Appendix A | same idea |

The crucial difference from the reference: there, the chapters *are* the code.
Here, the finished, tested code already lives in `nanowam/` at the repo root, and
the book narrates its construction. Each code chapter's `01_main-chapter-code/`
holds a small **runnable script** that reproduces that chapter's hands-on result
using the library — the nano-wam analog of the reference's per-chapter notebooks.

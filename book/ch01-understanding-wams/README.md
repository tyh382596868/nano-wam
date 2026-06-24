# Chapter 1 — Understanding World Action Models

> **This chapter covers**
> - What a *world model* is, what a *policy* is, and the gap between them
> - The World Action Model (WAM) thesis: predict future frames and actions jointly
> - A tour of four research WAMs — DreamZero, FastWAM, LingBot-VA, Motus — and the
>   five-part DNA they share
> - Why those systems are enormous, and why a *nano* version is worth building
> - A map of everything you will build by the end of this book

This is the only chapter without code. Its job is to give you the mental model
that the next ten chapters fill in, one component at a time. By the end you should
be able to say, in one sentence, what a World Action Model is and why every piece
of nano-wam exists.

---

## 1.1 Two halves of acting in the world

Imagine a robot arm in front of a table, asked to push a small block to a target.
To do this well, two very different kinds of knowledge are useful.

The first is **knowing what will happen**. If the gripper moves left by two
centimeters, where does the block end up? If it presses down, does the object tip
over? A model that answers "given the current scene and an action, what does the
next scene look like?" is a **world model**. It is a *simulator learned from
data* — a function from (observation, action) to a future observation.

The second is **knowing what to do**. Given the current scene and the goal, what
action moves us closer to success? A model that answers "given an observation,
what action should I take?" is a **policy**. It is the controller.

Classical robot learning tends to build these separately: train a dynamics model
here, train a policy there, maybe use the model to plan. The two are often
different networks, different objectives, different data pipelines.

> **The key observation.** A world model and a policy are reading the *same*
> scene and reasoning about the *same* physics. The world model has to understand
> cause and effect to predict the future; a good policy needs exactly that
> understanding to choose actions. Splitting them throws that shared structure
> away.

## 1.2 The World Action Model thesis

A **World Action Model (WAM)** is the bet that you should learn both halves *in
one network, at the same time*. Concretely, given a short history of observations
and a goal, a WAM predicts, together:

- the **future frames** — what the world will look like over the next few steps
  (the world-model half), and
- the **action chunk** — the sequence of actions that drives that future (the
  policy half).

Why fuse them? Three reasons recur across the literature:

1. **Imagination is a training signal for control.** Forcing the network to
   predict pixels makes it learn the scene's dynamics — object positions, contact,
   motion — which is precisely the knowledge a policy needs. Predicting the future
   is a rich, dense auxiliary task that a sparse action-only objective lacks.
2. **One model generalizes more cheaply.** Shared representations mean a new task
   often needs less task-specific data; the world-modeling part transfers.
3. **It unlocks new inference modes.** Once a single network can reason over
   (observations, future, actions) jointly, you can run it several ways — as a
   pure predictor, a pure policy, an inverse-dynamics estimator — without training
   new models. We will make this concrete in Chapter 6.

This is the thesis nano-wam is built to demonstrate, in miniature.

## 1.3 Four systems, one idea

The WAM idea is not hypothetical — it powers several recent research systems.
They differ in scale and emphasis, but the family resemblance is strong. Four in
particular motivated this book.

**DreamZero** (NVIDIA GEAR Lab) builds a WAM on top of a 14-billion-parameter
video-diffusion backbone (Wan2.1) and shows that a model trained to jointly
imagine video and produce actions behaves as a *zero-shot policy*: it performs
new tasks without task-specific training. Its headline is the strongest form of
the thesis — *world action models are policies*.

**FastWAM** asks a sharp, almost contrarian question: *do WAMs actually need
test-time future imagination?* It is built around an action diffusion transformer
and investigates whether you can skip rolling out the (expensive) video at
deployment and still act well. This question — does imagining the future at
inference time earn its cost? — is one we will reproduce directly, with a toggle,
in Chapter 9.

**LingBot-VA** organizes video prediction and action inference into a single
interleaved, *causal* sequence, using a dual-stream Mixture-of-Transformers and a
**KV cache** for efficient, long-horizon autoregressive rollout. Its emphasis on
causal, cached generation is the blueprint for Chapter 11.

**Motus** (THU-ML) presents a unified latent-action world model: a
Mixture-of-Transformers combining a video generator, a vision-language model, and
dedicated action/understanding experts, with a UniDiffuser-style scheduler that
switches between *five* operational modes from a single model. Its multi-mode
design is the inspiration for Chapter 6.

### The shared DNA

Strip away the scale and the family shares five ingredients. nano-wam keeps all
five — they are the spine of the book:

| Ingredient | What it is | Built in |
|---|---|---|
| **Diffusion Transformer (DiT)** | a transformer that denoises, conditioned on a noise level | Ch 4–5 |
| **Mixture-of-Transformers (MoT)** | shared attention, but per-modality (video/action) weights | Ch 5 |
| **Flow matching** | the generative objective: learn a velocity field from noise to data | Ch 4 |
| **Multi-mode masking** | one trained net → policy / world / inverse / joint | Ch 6 |
| **Causal + KV-cached rollout** | generate the future block-by-block, efficiently, without bound | Ch 11 |

If you understand these five, you understand the shape of every system above.

## 1.4 Why build a *nano* one

Here is the catch: the four systems are **huge**. Parameter counts from 5 to 14
billion. Pretrained video backbones (Wan) and vision-language models (Qwen) as
dependencies. Multi-GPU training infrastructure. Real robot stacks and
simulators. All of that is necessary for state-of-the-art results — and all of it
*hides the ideas*. It is very hard to learn how a WAM works by reading a codebase
that assumes 64 GPUs and a 14B checkpoint.

This book takes the opposite stance, the one Andrej Karpathy's *nanoGPT* took for
language models:

> **The nano philosophy.** Build the *smallest* thing that still genuinely
> embodies the core ideas. Make it readable in an afternoon and trainable on a
> single GPU — or even a CPU, for everything except final quality. Favor clarity
> over performance. Back every claim with code you can run.

nano-wam therefore *keeps* the five DNA ingredients and *drops* the billions of
parameters, the pretrained backbones, the multi-GPU infrastructure, and the real
robot stack. In their place it uses a tiny from-scratch model and a
self-contained toy task (a 2-D "reacher" you will build in Chapter 2) that needs
no external data and runs anywhere.

A fair warning, stated up front because honesty is part of this book's method:
nano-wam is a vehicle for *understanding*, not a state-of-the-art system. We will
verify that every mechanism works; we will *not* pretend that a tiny model trained
on a CPU produces strong policies. Chapter 12 draws this line carefully.

## 1.5 The finish line

By the end of Chapter 11 you will have written, from scratch, this:

```
   history frames ─┐                                ┌─► future frames (world)
   goal / mode     ┤─►  MoT Diffusion Transformer  ─┤
   (noised future) ┤    shared attention,           └─► action chunk  (policy)
   (noised action) ┘    per-stream FFN + AdaLN,
                        trained by rectified flow matching
```

and you will be able to run it four ways from a single checkpoint (as a policy, a
world model, an inverse-dynamics model, and a joint forecaster), measure whether
test-time imagination helps, feed it real robot data, and let it dream arbitrarily
far into the future with a KV cache.

The path there:

- **Part I (Ch 2–3)** lays the foundation: the data contract every component
  speaks, and the tokenizer that turns frames into the latent tokens the model
  consumes.
- **Part II (Ch 4–6)** builds the model: flow matching, the MoT Diffusion
  Transformer, and the four modes.
- **Part III (Ch 7–9)** brings it to life: training, sampling and visualization,
  and closed-loop control with the imagination ablation.
- **Part IV (Ch 10–12)** scales it: real data via the LeRobot adapter,
  long-horizon causal KV-cached rollout, and a look back at the frontier.

One idea ties the whole construction together, and it is worth holding onto from
the very first line of code:

> **Everything speaks the `(context, future, action)` window contract.** The
> data, the model, the trainer, the environment, the evaluator, and the real-data
> adapter all exchange the same triple. That single agreement is what lets a
> few hundred lines compose into a complete World Action Model.

We build that contract next.

---

## Summary

- A **world model** predicts the future from (observation, action); a **policy**
  chooses actions from observations. They rely on the same understanding of the
  scene.
- A **World Action Model** learns both jointly in one network. Predicting the
  future is a dense training signal for control, shares representations, and
  unlocks multiple inference modes.
- Four research WAMs — **DreamZero, FastWAM, LingBot-VA, Motus** — share five
  ingredients: a **DiT** core, **Mixture-of-Transformers**, **flow matching**,
  **multi-mode masking**, and **causal KV-cached rollout**.
- Those systems are huge; **nano-wam** keeps the five ideas but shrinks everything
  else, in the spirit of nanoGPT, so the ideas are legible and runnable.
- The whole system is held together by one **window contract**,
  `(context, future, action)`, which we build in Chapter 2.

## Exercises

1. **Classify the references.** For each of the four systems, name which of the
   five DNA ingredients it most emphasizes, and why (one sentence each).
2. **Design an I/O.** Pick a task you care about (a game, a drone, a kitchen
   robot). Write down what the `context`, `future`, and `action` would be — their
   shapes and units. You will be surprised how much design this fixes.
3. **State the thesis.** In a single sentence and without jargon, explain to a
   friend why predicting video might make a robot act better.
4. **Predict the cost.** FastWAM asks whether test-time imagination is worth it.
   Before we measure it in Chapter 9, write down your hypothesis: when *should*
   imagining the future help a policy, and when should it be wasted effort?

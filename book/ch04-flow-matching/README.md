# Chapter 4 — Rectified Flow Matching

> **This chapter covers**
> - Generative modeling as learning a *velocity field* that flows noise to data
> - The rectified-flow interpolant and its target velocity
> - The flow-matching loss
> - Sampling by integrating an ODE (Euler steps)
> - Why we will *detach* the latents when training the full model (Chapter 7)

We now have data (Chapter 2) and a way to turn frames into latent tokens
(Chapter 3). The missing piece is the *generative engine*: how does the model
learn to produce plausible future tokens and actions? nano-wam, like all four
reference systems, uses **rectified flow matching**. This chapter builds the idea
from scratch on a 2-D toy, then connects it to the code in `nanowam/flow.py` that
the rest of the book uses.

The math is light here and expanded in Appendix B.

---

## 4.1 Generation as a flow from noise to data

Suppose we want to generate samples from some data distribution — 2-D points, or
latent tokens, it does not matter. The flow-matching idea is to learn a
**velocity field**: a function `v(x, τ)` that, if you start at pure noise and
*follow it*, carries you to a data sample.

Picture a time axis `τ` from 0 to 1. At `τ=0` you have noise `ε ~ N(0, I)`. At
`τ=1` you want a real data point `x₁`. A *flow* is a path between them, and the
velocity field tells you which way to move at every point along the way:

```
   τ=0                         τ=1
   ε  ───────►  ───────►  ───────►  x₁
        v(x,τ)    v(x,τ)    v(x,τ)
   (noise)                    (data)
```

If we can learn `v`, we can generate: start at noise and integrate the field
forward to `τ=1`.

## 4.2 The straight-line (rectified) path

There are many paths from `ε` to `x₁`. **Rectified flow** picks the simplest one:
a straight line. Define the point at time `τ` as a linear interpolation:

```
x_τ = (1 - τ) · ε + τ · x₁
```

At `τ=0` this is `ε` (noise); at `τ=1` it is `x₁` (data). The velocity along this
straight path is just its time derivative — a *constant*:

```
dx_τ/dτ = x₁ - ε
```

So the **target velocity** the network should predict at `(x_τ, τ)` is `u = x₁ - ε`.
That is the whole trick: train a network to output `x₁ - ε` given the interpolated
point and the time.

In code (`nanowam/flow.py`), this is exactly `interpolate`:

```python
def interpolate(x1, eps, tau):
    """Form x_tau = (1-tau)*eps + tau*x1 and target velocity u = x1 - eps."""
    view = (-1,) + (1,) * (x1.dim() - 1)   # broadcast tau over feature dims
    t = tau.view(view)
    x_tau = (1 - t) * eps + t * x1
    u = x1 - eps
    return x_tau, u
```

The `view` reshape lets a per-sample time `tau` of shape `(B,)` broadcast over
whatever feature dimensions `x1` has (a 2-D point, a token grid, an action chunk).

## 4.3 The loss

Training could not be simpler. For each data point `x₁`:

1. draw a random time `τ ~ U(0, 1)` and noise `ε ~ N(0, I)`,
2. form the interpolant `x_τ` and target velocity `u = x₁ - ε`,
3. predict `v_θ(x_τ, τ)` and regress it onto `u` with mean-squared error.

```
L = || v_θ(x_τ, τ) - (x₁ - ε) ||²
```

That is it — no adversarial game, no score-matching subtleties, just a regression.
nano-wam's `flow_loss` is this loss applied to the WAM's video and action streams
(weighted, and only on the streams a *mode* selects — Chapter 6). Stripped to its
essence it is the three steps above.

## 4.4 Sampling: integrate the field

To generate, start at noise `x ← ε` and walk the velocity field from `τ=0` to
`τ=1`. The simplest integrator is **Euler**: take `S` small steps of size
`Δτ = 1/S`, each time nudging `x` by the predicted velocity:

```
x ← ε
for i in 0 .. S-1:
    τ = i / S
    x ← x + Δτ · v_θ(x, τ)
return x        # ≈ a data sample
```

This is the heart of `nanowam/flow.py:sample` (which also handles which streams
to denoise per mode, and the KV cache from Chapter 11). With straight-line flows,
a handful of steps (`S=10`–`16`) is often enough — a practical advantage of
rectified flow over many-step diffusion.

> **Forward reference: the detach.** In the full model, the *data* `x₁` we flow
> toward are the tokenizer's latents. We train the tokenizer separately by
> reconstruction (Chapter 3) and **detach** those latents when computing the flow
> loss, so the generative objective cannot reach back and corrupt the encoder.
> This is the standard latent-diffusion split; Chapter 7 shows the one line that
> does it, and Appendix B explains why it is correct.

## 4.5 Seeing it work: a 2-D flow

Abstract velocity fields are best understood by watching one. The runnable demo in
this chapter's `01_main-chapter-code/` learns to flow a Gaussian into a target
2-D distribution (eight Gaussians on a circle) with a tiny MLP velocity network,
using *exactly* the three-step loss above — and `nanowam.flow.interpolate` itself,
so the mechanism is the library's, not a copy — plus the Euler sampler. After a few
thousand steps, samples drawn from noise and integrated to `τ=1` cluster on the
eight modes:

```
mean distance to nearest mode BEFORE training: 0.979
mean distance to nearest mode AFTER  training: 0.141
```

It also saves a noise → generated → target scatter so you can see the eight blobs
appear. The entire generative mechanism of nano-wam is in that little script;
everything else is making the velocity network a conditional Mixture-of-Transformers
(Chapter 5) and choosing which streams to flow (Chapter 6).

---

## Summary

- Flow matching learns a **velocity field** `v(x, τ)` that transports noise
  (`τ=0`) to data (`τ=1`).
- **Rectified flow** uses straight-line paths: `x_τ = (1-τ)ε + τx₁`, whose target
  velocity is the constant `u = x₁ - ε`.
- Training is a plain **MSE regression** of the predicted velocity onto `u` at a
  random time `τ` — no adversarial or score-matching machinery.
- Sampling **integrates** the field with Euler steps from `τ=0` to `τ=1`; straight
  paths make a few steps suffice.
- In the full model the data are *detached* tokenizer latents (Chapter 7),
  keeping the encoder training clean.

## Exercises

1. **Derive the target.** Starting from `x_τ = (1-τ)ε + τx₁`, differentiate with
   respect to `τ` to confirm the target velocity is `x₁ - ε`. Why is it constant
   along the path?
2. **Step count.** In the chapter demo, vary the number of Euler steps `S` (e.g.,
   2, 5, 20). How does sample quality change? Where are the diminishing returns?
3. **A different target.** Replace the eight-Gaussians target with two concentric
   rings or a spiral. Does the same velocity network and loss still learn it? Which
   targets are harder, and why (think about where the velocity field is ambiguous)?
4. **Why straight lines?** In one or two sentences, argue why straight-line paths
   should need fewer integration steps than curved ones. (Appendix B has the full
   story.)

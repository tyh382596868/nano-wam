# Appendix B — The Math of Rectified Flow

> **This appendix covers**
> - The probability-flow view behind Chapter 4
> - Why the straight-line interpolant gives target velocity `x₁ - ε`
> - The link between flow matching and diffusion
> - Why few Euler steps can suffice, and why we detach the latents

Chapter 4 gave the *recipe*. This appendix gives the *reasons*, at the level of
detail needed to trust the one-line loss — no measure theory required.

---

## B.1 Generative modeling as transporting a distribution

We want to turn an easy distribution `p₀ = N(0, I)` (noise) into a hard one `p₁`
(data). Imagine a continuous family of distributions `p_τ` for `τ ∈ [0, 1]` that
morphs from `p₀` to `p₁`. If we knew a velocity field `v(x, τ)` whose flow carries
mass according to `p_τ`, we could sample by starting at noise and solving the
ordinary differential equation

```
dx/dτ = v(x, τ),     x(0) = ε ~ N(0, I)
```

up to `τ = 1`; then `x(1) ∼ p₁`. The whole game is learning that `v`.

## B.2 The conditional trick

Learning `v` directly looks circular — it depends on the very distributions we are
trying to model. **Flow matching** sidesteps this with a conditional construction.
Pick, for each data point `x₁`, a simple *conditional* path from a noise sample `ε`
to `x₁`, and define the marginal path as the mixture over data and noise. The key
theorem (Lipman et al., 2023) is that **regressing onto the conditional velocity
yields the correct marginal velocity** — the conditioning averages out. So we never
need the marginal field in closed form; we only need a per-sample path.

## B.3 The straight-line path and its velocity

**Rectified flow** (Liu et al., 2022) chooses the simplest conditional path: a
straight line from `ε` to `x₁`,

```
x_τ = (1 - τ) · ε + τ · x₁ .
```

Check the endpoints: `x₀ = ε` (noise), `x₁ = x₁` (data). Differentiate with respect
to `τ`:

```
d/dτ [ (1 - τ) ε + τ x₁ ] = -ε + x₁ = x₁ - ε .
```

The conditional velocity is the **constant** `x₁ - ε`, independent of `τ`. That is
the target the network regresses onto — exactly `interpolate`'s return value in
`nanowam/flow.py`. The loss is the conditional flow-matching objective:

```
L(θ) = E_{x₁, ε, τ} ‖ v_θ(x_τ, τ) - (x₁ - ε) ‖² ,    τ ~ U(0,1), ε ~ N(0, I).
```

Minimizing it makes `v_θ` the conditional-expectation `E[x₁ - ε | x_τ, τ]`, which —
by the conditional trick — is the marginal velocity that transports `p₀` to `p₁`.

## B.4 Why straight paths sample in few steps

We generate by integrating `dx/dτ = v_θ` with Euler steps. Integration error comes
from the path's **curvature**: a perfectly straight trajectory is integrated
exactly by a single Euler step, while a curved one needs many small steps to track.

Rectified flow's conditional paths are straight by construction. The learned
*marginal* trajectory is not perfectly straight (it is an average of straight
paths), but it is far straighter than the trajectories implied by standard
diffusion noise schedules. This is why nano-wam samples acceptably with `S = 10–16`
Euler steps where score-based diffusion often needs hundreds. (Iterated
"reflow" can straighten the paths further; nano-wam does not need it.)

## B.5 Relationship to diffusion

Flow matching and denoising diffusion are two views of the same probability-flow
ODE. Diffusion defines a forward noising stochastic process and learns the *score*
`∇ log p_τ`; sampling reverses the process. Flow matching directly learns a velocity
field for a chosen interpolation. For Gaussian noise the two are related by a simple
change of variables — predicting velocity, predicting noise (`ε`), and predicting
the score are interconvertible. nano-wam uses the velocity (rectified-flow)
parameterization because the loss and the straight-line target are the simplest to
state and implement.

## B.6 Why we detach the latents

Chapter 7 trains the tokenizer by reconstruction and **detaches** the latents fed to
the flow loss. The reason is now precise. The flow loss makes `x₁` (the latents) the
*target distribution* `p₁`. If gradients from the flow loss reached the encoder, the
encoder could change `p₁` itself — it could move the target to wherever is easiest to
denoise, rather than where frames are faithfully represented. That is a degenerate
solution (collapse). Detaching freezes `p₁` *from the flow loss's perspective* at
each step: the encoder is shaped only by reconstruction, and the flow model learns to
transport noise to whatever latent distribution the encoder currently defines. This
is the standard latent-diffusion split, and it is why the detach is load-bearing, not
cosmetic.

---

## Summary

- Generation is **transporting** `N(0, I)` to the data distribution by integrating a
  learned velocity field's ODE.
- Flow matching learns that field by regressing onto a **conditional** velocity; the
  conditioning averages out to the correct marginal field.
- **Rectified flow** uses straight-line paths, whose conditional velocity is the
  constant `x₁ - ε` — the one-line target in the code.
- Straight paths integrate in **few Euler steps**; flow matching and diffusion are
  two parameterizations of the same probability-flow ODE.
- **Detaching** the latents fixes the flow loss's target distribution, preventing the
  encoder from collapsing it — the formal reason behind Chapter 7's critical line.

## Further reading

- Lipman et al., *Flow Matching for Generative Modeling* (2023).
- Liu, Gong, Liu, *Rectified Flow* (2022).
- Esser et al., *Scaling Rectified Flow Transformers* (SD3, 2024) — the DiT +
  rectified-flow combination at scale.

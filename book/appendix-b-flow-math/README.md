# Appendix B — The Math of Rectified Flow

> **This appendix covers**
> - The probability-flow view behind Chapter 4
> - Why the straight-line interpolant gives target velocity `x₁ - ε`
> - The link between flow matching and diffusion
> - Practical consequences (few-step sampling, the detach)

**Maps to:** `nanowam/flow.py`; expands Chapter 4.

## Outline
1. Continuous normalizing flows and the probability-flow ODE in brief.
2. Conditional flow matching and the rectified (straight-path) choice.
3. Deriving the target velocity for `x_τ = (1-τ)ε + τ x₁`.
4. Relationship to denoising diffusion (and why few Euler steps can suffice).
5. Why latents are detached: training the encoder by reconstruction only.

## Summary
The minimal math to trust the one-line loss in Chapter 4.

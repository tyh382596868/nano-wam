# Chapter 4 — Rectified Flow Matching

> **This chapter covers**
> - Generative modeling as learning a *velocity field* that flows noise to data
> - The rectified-flow interpolant and its target velocity
> - The flow-matching loss
> - Sampling by integrating an ODE (Euler steps)

**Builds:** `interpolate`, `flow_loss` (single-stream first), the Euler `sample`.
**Maps to:** `nanowam/flow.py`. The math is expanded in Appendix B.

## Outline
1. Diffusion/flow intuition: straight-line paths from noise to data.
2. The interpolant `x_τ = (1-τ)ε + τ x₁` and target velocity `u = x₁ - ε`.
3. The loss: regress the network's predicted velocity onto `u`.
4. Sampling: integrate `dx/dτ = v_θ` from τ=0→1 with Euler steps.
5. A 2-D toy demo: learn to flow a Gaussian into a target distribution.

## Summary & Exercises
- Summary: train a velocity field; sample by integrating it.
- Exercises: vary the number of Euler steps and watch sample quality; implement a
  midpoint integrator and compare.

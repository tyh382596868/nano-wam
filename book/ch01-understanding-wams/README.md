# Chapter 1 — Understanding World Action Models

> **This chapter covers**
> - What a *world model* is and what a *policy* is, and why we unify them
> - The World Action Model (WAM) thesis: predict future frames and actions jointly
> - A tour of four research WAMs (DreamZero, FastWAM, LingBot-VA, Motus) and the
>   DNA they share
> - The "nano" philosophy and a map of what you will build in this book

**Builds:** nothing yet — this is the conceptual foundation.
**Maps to:** repo `PROJECT.md` (motivation) and `DESIGN.md §1` (survey).

## Outline
1. From prediction to action: world models vs. policies.
2. Why jointly model the world *and* the action (the imagination-helps-acting bet).
3. Four systems, one idea — a compact comparison table and their shared five-part
   DNA (DiT, MoT, flow matching, multi-mode, causal/KV-cache rollout).
4. The cost of the real systems (billions of params) and why a *nano* version is
   worth building.
5. The finish line: a diagram of the full nano-wam you'll have built by Chapter 11.

## Summary & Exercises
- Summary of the WAM idea and the book's arc.
- Exercises: classify each reference system by which DNA elements it emphasizes;
  sketch the I/O of a WAM for a task you care about.

# Chapter 2 — The Window Contract: Data, Episodes, and the Reacher World

> **This chapter covers**
> - Representing experience as *episodes* of (frames, actions)
> - The `(context, future, action)` **window** — the single contract the entire
>   system speaks
> - A fully self-contained procedural task (the "reacher") so we need no external
>   datasets
> - Slicing episodes into windows and sharding them to disk

**Builds:** `render_frame`, `generate_episode`, `make_windows`, `WindowDataset`.
**Maps to:** `nanowam/data.py`, `scripts/prepare_data.py`.

## Outline
1. Why a single data contract matters (everything downstream is unchanged when
   the source changes).
2. The reacher task: an agent square moving toward a goal — rendering and dynamics.
3. From an episode to training windows: context history, future frames, aligned
   action chunk.
4. Sharding windows to `.npz`; the `WindowDataset` that reads them.
5. Running `prepare_data.py` and inspecting a window.

## Summary & Exercises
- Summary: the window is the spine of the project.
- Exercises: change the horizons and observe the window count; add noise to the
  agent dynamics; render a goal-conditioned variant.

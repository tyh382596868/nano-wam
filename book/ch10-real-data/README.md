# Chapter 10 — Real Data: The LeRobot Adapter

> **This chapter covers**
> - The `EpisodeSource` abstraction that decouples data from the pipeline
> - Ingesting real LeRobot datasets (pushT, LIBERO slices)
> - Resizing frames and aligning the action dimension
> - Action normalization and saving stats for deployment

**Builds:** `EpisodeSource`, `ReacherSource`, `LeRobotSource`, `to_chw_uint8`.
**Maps to:** `nanowam/sources.py`, `scripts/prepare_data.py`, `nanowam/data.py`.

## Outline
1. Why an abstraction: the rest of the stack must not change for real data.
2. The episode interface and a lazy, optional `lerobot` import.
3. Frame resizing and the `action_dim` contract.
4. Pooling, train/val split, and per-dimension action normalization (`stats.npz`).
5. Building shards from `lerobot/pusht` and pointing training at them.

## Summary & Exercises
- Summary: real datasets feed the exact same window contract.
- Exercises: write an `EpisodeSource` for a dataset you have; verify the
  normalization round-trips; handle multi-camera observations.

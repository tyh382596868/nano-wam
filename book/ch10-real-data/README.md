# Chapter 10 — Real Data: The LeRobot Adapter

> **This chapter covers**
> - The `EpisodeSource` abstraction that decouples data from the pipeline
> - Ingesting real LeRobot datasets (pushT, LIBERO slices)
> - Resizing frames and aligning the action dimension
> - Action normalization and saving stats for deployment

We built nano-wam on a toy task so it would run anywhere. But the whole point of
the window contract (Chapter 2) was that the data source is *pluggable*. This
chapter cashes that promise in: we add an adapter for real robot datasets in the
**LeRobot** format — and, true to the contract, **not one line** of the model,
trainer, sampler, or evaluator changes. The code is `nanowam/sources.py`.

---

## 10.1 One abstraction: the episode

Everything downstream consumes *episodes* — a pair `(frames, actions)`:

```
frames  : (L, 3, H, W) uint8     H = W = cfg.image_size
actions : (L, Da)       float32
```

`make_windows` (Chapter 2) slices episodes into windows; `prepare_data.py` writes
the shards; `WindowDataset` reads them. None of that cares *where* the episodes
came from. So we define a tiny interface and let different sources implement it:

```python
class EpisodeSource:
    def iter_episodes(self):        # yields (frames uint8, actions float32)
        raise NotImplementedError
```

The toy task becomes one implementation:

```python
class ReacherSource(EpisodeSource):
    def iter_episodes(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_episodes):
            yield generate_episode(rng, self.ep_len, self.cfg.image_size)
```

and real data becomes another.

## 10.2 The LeRobot adapter

[LeRobot](https://github.com/huggingface/lerobot) is a standard format and library
for robot datasets (pushT, LIBERO, and many real-robot collections). `LeRobotSource`
wraps any LeRobot dataset and yields it as our episodes.

Two engineering realities shape it:

**(1) `lerobot` is a heavy, optional dependency.** nano-wam must stay installable
and testable without it, so the import is **lazy** — it happens only when you
actually iterate a `LeRobotSource`, with a clear message if the package is missing:

```python
def _load(self):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        raise ImportError("LeRobotSource needs the `lerobot` package: "
                          "pip install lerobot") from e
    return LeRobotDataset(self.repo_id)
```

**(2) Real frames and actions do not match our shapes.** LeRobot images come in
various sizes and layouts (HWC or CHW, `uint8` or float); actions can be any
dimensionality. The adapter resizes every frame to `cfg.image_size` and asserts the
action width matches `cfg.action_dim` with a helpful message:

```python
for i in range(start, end):
    row = ds[i]
    frames.append(to_chw_uint8(row[img_key], size))     # resize -> (3, size, size) uint8
    a = np.asarray(row[self.action_key], dtype=np.float32).reshape(-1)
    assert a.shape[0] == self.cfg.action_dim, (
        f"dataset action dim {a.shape[0]} != config action_dim {self.cfg.action_dim}")
    actions.append(a)
```

`to_chw_uint8` is a small, robust coercion: it accepts NumPy or torch, HWC or CHW,
`uint8` or float `[0,1]`, and always returns `(3, size, size) uint8` via bilinear
resize. Episodes are delimited using the dataset's episode index (with a
version-tolerant fallback), so windows never cross episode boundaries.

## 10.3 Action normalization

The reacher's actions were already near `[-1, 1]` by construction (Chapter 2). Real
robot actions are not — joint angles, end-effector deltas, and gripper commands
live on wildly different scales. Feeding those raw to a model whose velocity heads
are zero-initialized (and expect unit-ish targets) trains poorly.

So `prepare_data.py` can **normalize** actions: compute per-dimension mean and std
over the training windows, standardize, and **save the stats** for later
de-normalization at deployment:

```python
acts = np.concatenate([w[2].reshape(-1, cfg.data.action_dim) for w in train])
mean = acts.mean(0, keepdims=True)
std  = acts.std(0, keepdims=True) + 1e-6
np.savez(os.path.join(cfg.data.root, "stats.npz"), action_mean=mean, action_std=std)
# ... then (action - mean) / std is written into the shards
```

`WindowDataset` loads `stats.npz` if present and exposes `action_mean` /
`action_std`, so a deployed policy can map the model's normalized actions back to
real units. The stored shards already contain normalized actions, so training is
unchanged.

> **Why compute stats at prepare time?** It mirrors what the big systems do (they
> compute normalization statistics over the dataset), keeps training simple, and
> makes the stats an explicit, inspectable artifact next to the data.

## 10.4 Using it

The whole pipeline from Chapter 2 onward is reused; only the source flag changes:

```bash
pip install lerobot
python scripts/prepare_data.py --config configs/lerobot_pusht.yaml \
    --source lerobot --repo-id lerobot/pusht --normalize
python scripts/train.py --config configs/lerobot_pusht.yaml
```

`configs/lerobot_pusht.yaml` is the toy config with `action_dim: 2` (pushT's
action) and a larger model. After `prepare_data`, the shards look exactly like the
reacher's — same `ctx`/`future`/`action` arrays — so `train.py`, `sample.py`, and
the eval harness work without modification.

The chapter demo proves the adapter mechanics *without* downloading gigabytes: it
implements a tiny mock `EpisodeSource`, runs it through `make_windows`, normalizes,
shards, and reads it back with `WindowDataset` — confirming the round-trip and the
saved stats. It also exercises `to_chw_uint8` on odd-sized inputs and shows the
lazy-import error path.

---

## Summary

- The **`EpisodeSource`** interface decouples *where data comes from* from the rest
  of the pipeline; the toy task and real datasets are both just sources.
- **`LeRobotSource`** ingests any LeRobot-format dataset, with a **lazy** optional
  import, frame **resizing** (`to_chw_uint8`), and an **action-dim assertion**.
- **Action normalization** (per-dim mean/std, saved to `stats.npz`) is essential
  for real actions whose scales vary; `WindowDataset` exposes the stats for
  deployment de-normalization.
- Because real data writes the **same shard format**, training, sampling, and
  evaluation are completely unchanged — the window contract delivers.

## Exercises

1. **Write a source.** Implement an `EpisodeSource` for any `(frames, actions)`
   data you have (even a screen recording with logged inputs). Confirm the rest of
   the pipeline needs no changes.
2. **Normalize and invert.** Verify that `(action - mean) / std` then
   `x * std + mean` round-trips, and explain why deployment needs the inverse.
3. **Action-dim mismatch.** Point `LeRobotSource` at a dataset whose action width
   differs from the config and read the assertion. Why catch this at ingest time
   rather than at training time?
4. **Multi-camera.** Real datasets often have several camera views. Sketch how
   `to_chw_uint8` and the image-key selection would extend to stack or choose among
   them.

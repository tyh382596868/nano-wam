# Chapter 2 — The Window Contract: Data, Episodes, and the Reacher World

> **This chapter covers**
> - Representing experience as *episodes* of (frames, actions)
> - The `(context, future, action)` **window** — the single contract the whole
>   system speaks
> - Building a self-contained procedural task (the "reacher") so we need no
>   external datasets
> - Slicing episodes into windows and sharding them to disk
> - Loading windows with a `WindowDataset`

In Chapter 1 we ended on a promise: *everything in nano-wam speaks one data
contract*. This chapter defines that contract and builds the first concrete thing
that produces it. By the end you can generate data, slice it into training
windows, write it to disk, and load it back — with no dependency heavier than
NumPy and PyTorch.

All the code in this chapter lives in `nanowam/data.py` and
`scripts/prepare_data.py`.

---

## 2.1 Experience is a stream; training needs windows

A robot (or our toy agent) produces an **episode**: a stream of observations and
the actions taken between them.

```
frame_0  --a_0-->  frame_1  --a_1-->  frame_2  --a_2-->  ...  frame_{L-1}
```

A model cannot train on a whole variable-length episode at once. Instead we slide
a fixed-size **window** across the stream. Each window has three parts, and these
three parts are the contract:

- **context** — the last `K` frames *before* now. This is what the model
  conditions on (the recent past it can see).
- **future** — the next `Hv` frames *from* now. This is the world the model must
  predict (the "video horizon").
- **action** — the next `Ha` actions *from* now. This is the action chunk the
  model must predict (the "action horizon").

```
            |<--- K --->|<------ Hv frames ------>|
   ... f f f f f f f f  | F F F F                 |   future frames
                     ↑  t
            context  |  |<------ Ha actions ----->|
                        | a a a a a a a a         |   action chunk
```

> **Why a fixed contract matters.** Once every component agrees on the triple
> `(ctx_frames, future_frames, actions)` with fixed shapes, the data source
> becomes *pluggable*. In Chapter 10 we swap the toy generator below for real
> robot data, and **not a single line** of the model, trainer, or evaluator
> changes. The window is the spine of the project.

The shapes, with `K` context frames, `Hv` future frames, `Ha` actions, image size
`H×W`, and action dimension `Da`:

| Field | Shape | Meaning |
|---|---|---|
| `ctx_frames` | `(K, 3, H, W)` | recent history |
| `future_frames` | `(Hv, 3, H, W)` | world to predict |
| `actions` | `(Ha, Da)` | action chunk to predict |

## 2.2 A world we can build in twenty lines: the reacher

To learn the *mechanisms* of a WAM we do not need a photorealistic robot — we need
a task that (a) renders to frames, (b) has genuine actions, and (c) has real
dynamics linking the two, so there is something non-trivial to predict. And it
must run anywhere, with no external simulator, so tests and CI are portable.

The **reacher** satisfies all three: a white agent square moves toward a gray goal
square on a black background. The action is the agent's per-step displacement.

Rendering one frame is just stamping two squares onto a black canvas:

```python
def render_frame(agent: np.ndarray, goal: np.ndarray, size: int) -> np.ndarray:
    """Render one (3, H, W) uint8 frame: white agent, gray goal, black bg."""
    img = np.zeros((3, size, size), dtype=np.uint8)
    r = max(2, size // 12)

    def stamp(pos, value):
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        x0, x1 = max(0, cx - r), min(size, cx + r)
        y0, y1 = max(0, cy - r), min(size, cy + r)
        img[:, y0:y1, x0:x1] = value

    stamp(goal, 110)    # gray goal (draw first so agent overlays it)
    stamp(agent, 255)   # white agent
    return img
```

Note two small but deliberate choices. Frames are channel-first `(3, H, W)` —
PyTorch's convention — and `uint8` in `[0, 255]`, the natural format for storing
images compactly on disk (we convert to floats only at load time, §2.5).

## 2.3 Generating an episode

An episode picks a random agent and goal, then steps the agent toward the goal a
little each frame, recording the displacement as the action:

```python
def generate_episode(rng: np.random.Generator, length: int, size: int):
    """One episode. Returns frames (L,3,H,W) uint8, actions (L,Da=2) float32."""
    margin = size // 6
    agent = rng.uniform(margin, size - margin, size=2)
    goal = rng.uniform(margin, size - margin, size=2)
    speed = size / 16.0

    frames, actions = [], []
    for _ in range(length):
        frames.append(render_frame(agent, goal, size))
        direction = goal - agent
        dist = np.linalg.norm(direction) + 1e-6
        step = direction / dist * min(speed, dist)
        step = step + rng.normal(0, speed * 0.05, size=2)  # small noise
        actions.append((step / speed).astype(np.float32))   # ~[-1, 1]
        agent = np.clip(agent + step, 0, size - 1)
    return np.stack(frames), np.stack(actions)
```

Three details worth pausing on, because they come back later:

1. **The action is the displacement, normalized by `speed` to roughly `[-1, 1]`.**
   Keeping action targets near unit scale matters for the flow-matching model in
   Chapter 4 — its zero-initialized output heads expect to learn small-magnitude
   velocities. (For *real* data, whose actions have arbitrary scale, we add
   normalization in Chapter 10.)
2. **The action at step `t` is exactly the displacement that turns `frame_t` into
   `frame_{t+1}`.** That means the data contains a genuine forward- and
   inverse-dynamics relationship — there is something real for the world model and
   the inverse-dynamics mode (Chapter 6) to learn.
3. **A little noise** is added to the step so the mapping is not perfectly
   deterministic, which keeps the task honest.

## 2.4 From episode to windows, and onto disk

Sliding the window across one episode is a short loop. For each valid start `t`,
take the `K` frames before it as context, the next `Hv` frames as the future, and
the next `Ha` actions as the chunk:

```python
def make_windows(frames, actions, K, Hv, Ha):
    """Slide windows over one episode. Yields (ctx, future, action) arrays."""
    L = len(frames)
    horizon = max(Hv, Ha)
    out = []
    for t in range(K, L - horizon + 1):
        ctx = frames[t - K:t]
        future = frames[t:t + Hv]
        action = actions[t:t + Ha]
        out.append((ctx, future, action))
    return out
```

The range `range(K, L - horizon + 1)` guarantees every window has a full context
*and* a full future/action horizon — we never run off either end of the episode.

`scripts/prepare_data.py` calls `generate_episode` many times, pools all the
windows, shuffles them, splits off a validation fraction, and writes them as
compressed NumPy shards. The on-disk format is exactly the contract:

```
<root>/<split>/shard_*.npz   with arrays
    ctx     (n, K,  3, H, W) uint8
    future  (n, Hv, 3, H, W) uint8
    action  (n, Ha, Da)      float32
```

Running it is one command (the config fixes `K`, `Hv`, `Ha`, image size, etc.):

```bash
python scripts/prepare_data.py --config configs/pusht_tiny.yaml --episodes 200
```

```
[prepare_data] source='reacher' image=64 window=ctx2+future4, actions 8x2
[prepare_data] 200 episode(s) -> 3000 windows
[prepare_data]   wrote 1 shard(s) to data/.../train
[prepare_data]   wrote 1 shard(s) to data/.../val
```

> **Why shard to disk at all?** Two reasons. It separates the (sometimes slow)
> data-generation step from training, so you generate once and train many times.
> And it is the seam where real datasets enter: in Chapter 10 a different source
> writes the *same* shard format, and everything downstream is unchanged.

## 2.5 Reading windows back: `WindowDataset`

The training side is a standard PyTorch `Dataset` that loads the shards into
memory and serves one window at a time, converting `uint8` frames to floats in
`[0, 1]`:

```python
class WindowDataset(Dataset):
    """Reads sharded windows from `<cfg.root>/<split>` into memory."""

    def __init__(self, cfg: DataConfig, split: str = "train"):
        shard_dir = os.path.join(cfg.root, split)
        shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not shards:
            raise FileNotFoundError(f"No shards in {shard_dir}. Run prepare_data first.")
        ctx, future, action = [], [], []
        for s in shards:
            d = np.load(s)
            ctx.append(d["ctx"]); future.append(d["future"]); action.append(d["action"])
        self.ctx = np.concatenate(ctx)
        self.future = np.concatenate(future)
        self.action = np.concatenate(action)
        # (optional action-normalization stats are loaded here too — see Ch 10)

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, idx):
        return {
            "ctx_frames": torch.from_numpy(self.ctx[idx]).float() / 255.0,
            "future_frames": torch.from_numpy(self.future[idx]).float() / 255.0,
            "actions": torch.from_numpy(self.action[idx]).float(),
        }
```

That `__getitem__` return value — a dict with `ctx_frames`, `future_frames`,
`actions` — *is* the window contract in code. Every later chapter consumes exactly
this.

> **A note on simplicity.** Loading all shards into memory is fine because the toy
> dataset is small. For a large real dataset you would memory-map or stream; the
> interface (`__len__` / `__getitem__`) would not change. We keep the nano version.

## 2.6 Looking at the data

It is always worth *seeing* your data before modeling it. A few lines produce a
window and confirm the shapes:

```python
from nanowam.config import load_config
from nanowam.data import WindowDataset

cfg = load_config("configs/pusht_tiny.yaml")
ds = WindowDataset(cfg.data, "train")
item = ds[0]
print(item["ctx_frames"].shape)     # torch.Size([2, 3, 64, 64])
print(item["future_frames"].shape)  # torch.Size([4, 3, 64, 64])
print(item["actions"].shape)        # torch.Size([8, 2])
```

If you render `ctx_frames` followed by `future_frames` as an image strip, you see
the white square drifting steadily toward the gray one — exactly the dynamics the
WAM will have to reproduce.

---

## Summary

- Experience is a stream of frames and actions; training consumes fixed-size
  **windows** sliced from it.
- The window is the triple `(context, future, action)` — the recent past to
  condition on, the future frames to predict, and the action chunk to predict.
  This **contract** is what makes the data source pluggable.
- The **reacher** is a self-contained procedural task: render two squares, step
  the agent toward the goal, record the displacement as the action. It needs no
  external simulator, yet has genuine dynamics to learn.
- `make_windows` slides the window over an episode; `prepare_data.py` writes
  windows as `.npz` shards; `WindowDataset` reads them back as floats in `[0, 1]`.
- The action is normalized to roughly `[-1, 1]` now (it matters for training in
  Chapter 4); real-data normalization comes in Chapter 10.

## Exercises

1. **Horizon arithmetic.** With episode length `L`, context `K`, and horizon
   `max(Hv, Ha)`, how many windows does one episode yield? Change `Hv` in the
   config, regenerate, and confirm your formula against the printed window count.
2. **Make it harder.** Increase the action noise in `generate_episode` and
   regenerate. How might a noisier world change what the model can predict?
3. **A second task.** Sketch (or write) a new generator — e.g., two agents, or a
   goal that moves — that still returns `(frames, actions)` with the same shapes.
   Notice that `make_windows`, the shards, and `WindowDataset` need no changes.
4. **Inspect, don't trust.** Save the first window of an episode as an image strip
   and verify by eye that the action signs match the agent's motion direction.

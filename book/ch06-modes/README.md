# Chapter 6 — Four Modes from One Network

> **This chapter covers**
> - The UniDiffuser idea: which streams are *clean* vs *noised* defines the task
> - The four modes — policy, world, inverse, joint — and their stream masks
> - Per-mode conditioning subtleties (clean future, or zeros, for the video stream)
> - Sampling a mode per batch so one checkpoint serves them all
> - Wiring modes into the flow loss

We have a generative engine (Chapter 4) and a conditional model to drive it
(Chapter 5). This chapter delivers on a promise from Chapter 1: that *one trained
network* can act as a policy, a world model, an inverse-dynamics estimator, and a
joint forecaster. The trick is beautifully simple — it is all about which streams
you noise. The code lives in `nanowam/modes.py` and the mode logic in
`nanowam/flow.py`.

---

## 6.1 Conditioning by masking

In Chapter 4, generation meant denoising: start a stream at noise and flow it to
data. But our model has *two* generable streams — video and action — plus a clean
context. What if, for a given training example, we **noise only some streams and
leave others clean**?

A stream that is left clean becomes **conditioning**: the model sees it as given.
A stream that is noised becomes a **target**: the model must generate it. So the
choice of *which streams to noise* turns the same network into different tasks.
This is the **UniDiffuser** idea (and Motus's multi-mode scheduler):

| If you noise… | …and condition on… | the model is a… |
|---|---|---|
| action | context | **policy** (act from observations) |
| video | context + action | **world model** (predict frames from actions) |
| action | context + future video | **inverse-dynamics** model (what action caused this?) |
| video + action | context | **joint** forecaster (imagine both) |

Four tasks, one network, zero extra parameters.

## 6.2 The mode table

nano-wam names the four modes and describes each by a small `StreamMask` —
booleans for which streams get noised (`nanowam/modes.py`):

```python
class Mode(IntEnum):
    JOINT = 0
    POLICY = 1
    WORLD = 2
    INVERSE = 3

@dataclass(frozen=True)
class StreamMask:
    noise_video: bool
    noise_action: bool
    goal_present: bool = True

MODE_TABLE = {
    Mode.JOINT:   StreamMask(noise_video=True,  noise_action=True),
    Mode.POLICY:  StreamMask(noise_video=False, noise_action=True),
    Mode.WORLD:   StreamMask(noise_video=True,  noise_action=False),
    Mode.INVERSE: StreamMask(noise_video=False, noise_action=True),
}
```

The context stream is never in the table because it is *always* clean — it is the
conditioning image, by definition.

## 6.3 A subtlety: what fills an un-noised stream?

`POLICY` and `INVERSE` have the same mask (`noise_action=True`,
`noise_video=False`) — both predict actions without denoising video. But they
differ in *what the clean video stream contains*:

- In **inverse dynamics**, the future frames are *given* — the question is "what
  action took us from the context to this observed future?" So the video stream is
  fed the **real future**.
- In **policy**, there is no future available at deployment time. So the video
  stream is fed **zeros** — the model must act from context alone, without
  imagining.

`build_stream_inputs` (in `nanowam/flow.py`) encodes exactly this. For an
un-noised video stream it feeds the clean future for `INVERSE` and zeros
otherwise:

```python
if mask.noise_video:
    eps = torch.randn_like(batch["video"])
    x_v, u_v = interpolate(batch["video"], eps, tau)      # noise it; supervise
else:
    cond_v = batch["video"] if mode == Mode.INVERSE else torch.zeros_like(batch["video"])
    x_v, u_v = cond_v, None                               # condition; don't supervise
```

The same function noises the action stream when the mask says so. Streams that are
not noised are passed through clean and contribute no loss.

> **Why policy feeds zeros.** This is the deployment-consistent choice and the
> basis of FastWAM's question (Chapter 9): a policy that never denoises video is
> "imagination-free." Training it with a zero video stream means it behaves the
> same at deployment, where the future is genuinely unavailable.

## 6.4 The joint training objective

Training draws a mode per example and applies the flow loss only to the streams
that mode noises. `flow_loss` (`nanowam/flow.py`) is the Chapter 4 loss, per
stream, gated by supervision flags:

```python
def flow_loss(model, batch, mode, cfg, generator=None):
    tau = torch.rand(B, device=device, generator=generator)
    s = build_stream_inputs(batch, mode, tau, generator)     # noise per the mode
    v_video, v_action = model(batch["ctx"], s["video_in"], s["action_in"],
                              tau, mode_id, batch.get("goal"))
    loss = 0
    if s["sup_video"]:
        loss = loss + cfg.loss_video_weight  * F.mse_loss(v_video,  s["u_video"])
    if s["sup_action"]:
        loss = loss + cfg.loss_action_weight * F.mse_loss(v_action, s["u_action"])
    return loss, metrics
```

It is the rectified-flow MSE from Chapter 4, summed over whichever streams are
live, with per-stream weights.

## 6.5 Sampling a mode per batch

To make a *single* checkpoint good at all four jobs, each training batch draws a
mode from a configurable distribution (the UniDiffuser-style schedule):

```python
def sample_mode(mode_probs, generator=None):
    names = list(mode_probs.keys())
    weights = torch.tensor([mode_probs[n] for n in names])
    idx = int(torch.multinomial(weights, 1, generator=generator).item())
    return Mode[names[idx].upper()]
```

with probabilities set in the config, e.g.:

```yaml
train:
  mode_probs: { joint: 0.5, policy: 0.25, world: 0.15, inverse: 0.10 }
```

Tilt this distribution toward the capability you care about. Heavy `joint` teaches
the full world-action coupling; more `policy` sharpens control; more `world`
sharpens prediction.

## 6.6 Checking the modes

A fresh (untrained) model cannot do anything well yet, but every mode should
produce a **finite, differentiable** scalar loss — confirming the masks and the
loss wiring are correct. The chapter demo runs all four:

```python
from nanowam.flow import flow_loss
from nanowam.modes import Mode, MODE_TABLE

for mode in Mode:
    loss, _ = flow_loss(model, batch, mode, cfg.flow)
    loss.backward()                       # must be differentiable
    print(mode.name, MODE_TABLE[mode], float(loss))
```

It prints, for each mode, which streams it noises and a finite loss you can
backprop — the contract every later training step relies on.

---

## Summary

- **Conditioning by masking**: a clean stream is given (conditioning); a noised
  stream is generated (a target). The choice of which streams to noise defines the
  task.
- nano-wam's four modes — **policy, world, inverse, joint** — are described by a
  tiny `StreamMask` each in `MODE_TABLE`. Context is always clean.
- `POLICY` and `INVERSE` share a mask but differ in the clean video stream's
  contents: real future for inverse, **zeros** for policy (deployment-consistent,
  imagination-free).
- `flow_loss` applies the Chapter 4 rectified-flow MSE to exactly the streams a
  mode noises; `sample_mode` draws a mode per batch so one checkpoint serves all.

## Exercises

1. **Read the masks.** For each mode, write down what is clean and what is noised,
   and name the real-world question it answers.
2. **Policy vs. inverse.** They share a `StreamMask`. Explain precisely what makes
   them different tasks, and where that difference lives in the code.
3. **Tilt the schedule.** Change `mode_probs` to be policy-heavy, then world-heavy.
   Before training (Chapter 7), predict how each change should affect the model's
   eventual control vs. prediction quality.
4. **Add a mode.** Design a fifth mode — say, *action infilling* (some actions
   given, some predicted). What `StreamMask` (or extension to it) would you need,
   and what would `build_stream_inputs` have to do?

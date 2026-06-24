# Chapter 9 — Closed-Loop Control and the Imagination Ablation

> **This chapter covers**
> - Turning the data generator into a step-able environment
> - Using the WAM as a receding-horizon policy
> - FastWAM's question: does *test-time imagination* help?
> - Measuring success rate with imagination on vs. off

So far the model has only *predicted*. This chapter closes the loop: we put the
WAM in control of an environment, let it act, and measure whether it reaches the
goal. Then we run the experiment this whole project was partly built to ask — does
imagining the future at test time actually make the policy better? The code is
`nanowam/envs.py` and `nanowam/eval.py`.

---

## 9.1 Open loop vs. closed loop

Chapter 8 was **open loop**: feed a fixed context, predict, look. **Closed loop**
is different and harder — the model's actions change the world, which changes the
next observation, which changes the next action. Errors compound. A model can have
a low one-step prediction error and still fail in closed loop because small
mistakes snowball.

To run closed loop we need an **environment** we can `reset` and `step`.

## 9.2 A step-able world that matches the data

Crucially, the environment must use the *same action convention* as the offline
data (Chapter 2), or a policy trained on that data will be speaking the wrong
language. `ReacherEnv` (`nanowam/envs.py`) is the step-able twin of
`generate_episode`: same agent, same goal, same `action = displacement / speed`:

```python
class ReacherEnv:
    def __init__(self, cfg, seed=0):
        self.size = cfg.image_size
        self.speed = self.size / 16.0           # identical to generate_episode
        self.success_radius = self.size / 10.0
        ...

    def reset(self):
        self.agent = self.rng.uniform(margin, self.size - margin, size=2)
        self.goal  = self.rng.uniform(margin, self.size - margin, size=2)
        return self._frame()

    def step(self, action):
        self.agent = np.clip(self.agent + action * self.speed, 0, self.size - 1)
        dist = np.linalg.norm(self.agent - self.goal)
        return self._frame(), dist < self.success_radius, dist
```

`step` returns the new frame, a `done` flag (reached the goal), and the distance.
That is all a controller needs.

## 9.3 The WAM as a receding-horizon policy

The model predicts an *action chunk* of length `Ha`, but the world drifts after we
act, so executing all `Ha` actions blindly is unwise. The standard fix is
**receding horizon** (a.k.a. model-predictive control): predict a chunk, execute
only the first few actions, then **replan** from the new observation.

`rollout_episode` (`nanowam/eval.py`) does exactly this:

```python
buf = [frame] * K                       # rolling window of the last K frames
while steps < max_steps:
    ctx_tok = to_tokens(tokenizer.encode(stack(buf[-K:])))
    out = sample(model, {"ctx": ctx_tok}, mode, cfg.flow)   # predict an action chunk
    actions = out["action"][0]
    for j in range(replan_every):       # execute a few, then loop to replan
        frame, done, _ = env.step(actions[j])
        buf.append(frame)
        if done:
            return True, steps
        steps += 1
```

Note what conditions the policy: only the last `K` frames. The model must infer
where the goal is and which way to move from the image alone.

## 9.4 The experiment: does imagination help?

Now the payoff. The same `rollout_episode` can run in two modes, and the
difference between them *is* FastWAM's question:

- **`imagine=none`** → **`POLICY`** mode: denoise the action stream only. The model
  acts without imagining the future (the video stream is zeros, as in Ch 6).
- **`imagine=joint`** → **`JOINT`** mode: denoise the future frames *and* the
  actions together, then act on the actions. The model imagines, then acts.

If imagining the future genuinely helps control, `joint` should reach the goal
more often. `run_ablation` runs both over a set of **seeded** episodes (the same
episodes for each setting, for a fair comparison) and reports success rate and mean
steps:

```python
def run_ablation(model, tokenizer, cfg, device, episodes, settings=("none", "joint")):
    results = {}
    for setting in settings:
        mode = {"none": Mode.POLICY, "joint": Mode.JOINT}[setting]
        succ, steps = [], []
        for ep in range(episodes):
            env = ReacherEnv(cfg.data, seed=1000 + ep)      # same seeds across settings
            ok, n = rollout_episode(model, tokenizer, env, mode, cfg, device, max_steps)
            succ.append(ok); steps.append(n) if ok else None
        results[setting] = {"success_rate": np.mean(succ), "mean_steps": ...}
    return results
```

Running it:

```bash
python scripts/sample.py --config configs/pusht_tiny.yaml --closed-loop --episodes 50
```

```
[eval] closed-loop imagination ablation (50 episodes)
setting        mode       success   mean_steps
imagine=none   policy        ...
imagine=joint  joint         ...
```

## 9.5 Reading the result honestly

This is where the book's commitment to honesty matters most. On a brief CPU run,
the model is undertrained, and the numbers are *noisy* — with only tens of
episodes and a weak policy, a few-point gap between `none` and `joint` is not
strong evidence either way. The **harness** is correct and tested; the **scientific
conclusion** requires a GPU-scale run with hundreds of episodes (Chapter 12).

> **What we can and cannot claim.** We *can* claim: the closed-loop machinery
> works, the ablation is set up fairly (shared seeds, identical everything but the
> mode), and it produces the numbers. We *cannot* yet claim a trustworthy answer to
> "does imagination help?" at this scale. Saying so plainly is the point — a result
> you cannot reproduce at scale is not a result.

This is, in miniature, exactly the kind of experiment the reference systems run at
scale. You have now built the apparatus to ask the question; Chapter 12 is about
turning the crank hard enough to trust the answer.

---

## Summary

- **Closed loop** is harder than open loop: the policy's actions change future
  observations, so errors compound.
- `ReacherEnv` is the step-able twin of the data generator, using the **same action
  convention**, so an offline-trained policy transfers.
- The WAM acts by **receding horizon**: predict an action chunk, execute a few
  actions, replan.
- The **imagination ablation** compares `imagine=none` (POLICY) vs `imagine=joint`
  (JOINT) over **seeded** episodes — FastWAM's question, made measurable.
- At nano-CPU scale the numbers are **noisy**; the harness is the deliverable, and a
  trustworthy answer needs scale (Ch 12). Saying this honestly is the method.

## Exercises

1. **Replan rate.** Vary `replan_every` from 1 to `Ha`. How does replanning more
   often change success rate and compute cost?
2. **Fair comparisons.** Why does `run_ablation` seed each episode the same way
   across settings? What would happen to the comparison if it did not?
3. **Compounding error.** Construct an example where a model with low one-step
   prediction error still fails in closed loop. What property does closed-loop
   success require that open-loop MSE does not measure?
4. **Your hypothesis, revisited.** In Chapter 1 you predicted when imagination
   should help. Given a GPU, how many episodes and what task changes would you need
   to test it convincingly?

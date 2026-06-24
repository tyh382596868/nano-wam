# Chapter 12 — Scaling, Extensions, and the Frontier

> **This chapter covers**
> - Running nano-wam at scale with one command
> - What is verified (mechanisms) vs. what needs a GPU (quality)
> - Natural extensions: language/goal conditioning, CFG, per-stream time, deployment
> - How the big research systems go beyond nano-wam

You have built the whole thing: a tokenizer, a Mixture-of-Transformers diffusion
model, a flow-matching trainer, four inference modes, a closed-loop controller with
an imagination ablation, a real-data adapter, and causal KV-cached dreaming. This
final chapter connects what you built to the frontier — how to scale it, what is
and isn't proven, and where to go next. There is no new mechanism here; it is about
*perspective*.

---

## 12.1 Scaling with one command

Everything so far ran on a CPU because the model was tiny. The same code scales by
changing the config. `configs/reacher_gpu.yaml` widens and deepens the model
(`dim: 512, depth: 12`), enlarges the batch, turns on mixed precision, and trains
for many more steps. `scripts/run_gpu.sh` runs the full pipeline end to end:

```bash
./scripts/run_gpu.sh                          # data -> train -> visualize -> ablation
EPISODES=4000 ./scripts/run_gpu.sh            # more data
./scripts/run_gpu.sh configs/causal_tiny.yaml # any config (causal -> also dreams)
```

Nothing about the *architecture* changes — only `dim`, `depth`, `batch_size`,
`max_steps`, and `amp`. That is the point of building from scratch: scaling is a
config edit, not a rewrite. The chapter demo prints the parameter count for the CPU
and GPU configs side by side so you can feel the difference.

## 12.2 The honesty ledger

This book has insisted, chapter after chapter, on separating *mechanism* from
*quality*. Here is the full ledger.

**Verified (on CPU, by tests and the chapter demos):**

- every shape and gradient path;
- flow loss in all four modes; the policy-mode overfit;
- the generation → decode → visualize pipeline, with an honest baseline;
- the closed-loop control loop and the imagination-ablation harness;
- the real-data adapter (mock-source round-trip);
- block-causal attention, the **exact** KV cache, and unbounded dreaming.

**Not verified (needs a GPU-scale run):**

- whether predicted frames *beat* the copy-last-frame baseline convincingly;
- a *trustworthy* answer to FastWAM's question (does imagination help control?);
- whether dreamed rollouts stay coherent over many blocks.

> **Why say this so loudly?** Because a result you cannot reproduce is not a result.
> The value of nano-wam is that every claim in its docs is backed by a test or a run
> — and the claims it cannot yet back, it labels plainly. When you turn the crank on
> a GPU, you will know exactly which numbers were mechanism and which were quality.

## 12.3 Extensions already hooked in the code

Several natural extensions have hooks in place, left as on-ramps:

- **Goal / language conditioning.** The model takes a `goal` vector and has a
  `goal_embed` config; a real task would feed a language or goal-image embedding
  here to make the WAM multi-task (as DreamZero and Motus do).
- **Classifier-free guidance.** `StreamMask.goal_present` exists so you can train
  with the goal sometimes dropped, then guide sampling toward the goal at inference.
- **Per-stream flow time.** The API already accepts a `τ` of shape `(B, n_streams)`;
  enabling truly independent per-stream noise levels (cleaner partial-noise modes)
  is non-breaking.
- **Real-robot deployment.** Closed-loop control plus the saved action-normalization
  stats (Chapter 10) are most of a deployment loop; the missing piece is mapping the
  model's normalized actions back to real units and into a controller.

## 12.4 Back to the frontier

Re-read the four systems from Chapter 1 now that you have built the core. Each adds
something *on top of* what you wrote:

- **DreamZero** swaps the tiny tokenizer for a powerful pretrained video VAE (Wan)
  and scales to 14B parameters, reaching zero-shot policy behavior.
- **FastWAM** studies, at scale and across real benchmarks, the exact ablation you
  built — turning "does imagination help?" into a measured, defensible answer.
- **LingBot-VA** takes the causal, KV-cached path you implemented and pushes it to
  long-horizon, interleaved video-action rollout on real robots.
- **Motus** generalizes the four modes into a richer multi-expert
  Mixture-of-Transformers (adding a vision-language model and latent actions) with a
  five-mode scheduler.

None of these is magic now. They are *your* nano-wam — the same five ingredients
(DiT, MoT, flow matching, multi-mode masking, causal KV-cache rollout) — scaled up,
fed real data, and engineered hard. That is the whole point: once you have built the
smallest thing that embodies the ideas, the frontier is a matter of degree, not of
kind.

## 12.5 Where to go next

Concrete next steps, roughly in order of value:

1. **Run it on a GPU.** Scale `reacher_gpu.yaml` until world prediction clearly
   beats the baseline; then run the imagination ablation over hundreds of episodes
   for a trustworthy number.
2. **Feed it real data.** Point the LeRobot adapter at pushT or a LIBERO slice and
   train.
3. **Add language.** Wire a goal/instruction embedding into `goal` and make the WAM
   multi-task.
4. **Pick one reference idea** — e.g., a pretrained VAE, or latent actions — and
   graft it onto nano-wam. You now have the codebase to do it cleanly.

---

## Summary

- Scaling nano-wam is a **config change** (`reacher_gpu.yaml`) plus
  `scripts/run_gpu.sh` — the architecture is unchanged.
- The **honesty ledger** cleanly separates verified *mechanisms* (everything you
  built and tested) from unverified *quality* (which needs a GPU run).
- Extensions are pre-hooked: **goal/language conditioning**, **classifier-free
  guidance**, **per-stream flow time**, and **deployment** with action
  de-normalization.
- The four frontier systems are nano-wam's five ingredients **scaled and
  engineered** — not different in kind. Building the small version is what makes the
  big ones legible.

## Exercises

1. **Beat the baseline.** On a GPU, scale up until world-prediction MSE drops below
   copy-last-frame. Record what it took (params, steps, data).
2. **A trustworthy ablation.** Run the imagination ablation over many episodes and
   seeds. Is the `joint`-vs-`none` gap real? Report a confidence interval.
3. **Add a goal.** Implement goal conditioning for a two-goal reacher and show the
   policy can be directed. What changes in data, model, and training?
4. **Graft a frontier idea.** Choose one element from DreamZero / FastWAM /
   LingBot-VA / Motus and implement a nano version on top of this codebase.

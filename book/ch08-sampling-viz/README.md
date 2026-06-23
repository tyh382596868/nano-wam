# Chapter 8 — Sampling, Decoding, and Visualizing Rollouts

> **This chapter covers**
> - Generating future frames and actions with the trained model
> - Decoding predicted latents back into pixels
> - Building GT-vs-prediction grids and animated rollouts
> - Reading honest baselines (copy-last-frame) before trusting a number

A trained model (Ch 7) is only useful if you can *run* it and *see* what it
produces. This chapter turns the model into pictures: sample in any of the four
modes, decode the predicted latents back to frames, and lay ground truth beside
prediction. It also introduces the discipline that keeps this project honest —
always compare against a trivial baseline. The code is `scripts/sample.py`.

---

## 8.1 Sampling is the Chapter 4 sampler, gated by mode

Generation reuses the Euler ODE sampler from Chapter 4 (`nanowam/flow.py:sample`),
with the mode (Ch 6) deciding which streams are denoised and which are clean
conditioning:

- **`--mode world`**: condition on context **and the ground-truth actions**,
  denoise the **future frames**. Tests "given what the agent does, can the model
  imagine the consequences?"
- **`--mode joint`**: condition on context only, denoise **both** future frames
  and actions. The full WAM forecast.
- **`--mode policy`**: denoise the **action chunk** only — no imagination.

In code the only difference between modes is which clean conditioning you pass and
which mode id you hand to `sample`:

```python
ctx_tok = to_tokens(tokenizer.encode(ctx))     # tokenize the context frames
cond = {"ctx": ctx_tok}
if mode == Mode.WORLD:
    cond["action"] = actions                   # world mode conditions on actions
out = sample(model, cond, mode, cfg.flow)      # -> {"video": ...} and/or {"action": ...}
```

## 8.2 From latents back to pixels

The model predicts latent *tokens*. To look at them we reverse the tokenizer
pipeline from Chapter 3: reshape tokens to a latent grid, then decode to frames:

```python
def decode_video_tokens(tokenizer, tokens, T, h, w):
    """(B, T*h*w, C) latent tokens -> (B, T, 3, H, W) frames in [0, 1]."""
    return tokenizer.decode(from_tokens(tokens, T, h, w))
```

That is the whole bridge: `from_tokens` undoes the flatten, `decode` undoes the
encode. Now `out["video"]` — a stack of predicted latent tokens — becomes a stack
of RGB frames you can write to disk.

## 8.3 Seeing ground truth beside prediction

To judge a world model you look at its predictions *next to* reality. `sample.py`
builds two artifacts (`save_comparison`):

- a **PNG grid**: each row is one sample, laid out as `[ground-truth frames | gap |
  predicted frames]`, and
- an **animated GIF**: per future timestep, ground truth stacked above prediction,
  so you can watch them evolve together.

The conversion from float `(.., 3, H, W)` tensors to the `uint8` HWC images that
image libraries expect is a one-liner:

```python
def to_uint8(frames):
    x = frames.clamp(0, 1).mul(255).byte().cpu().numpy()
    return np.moveaxis(x, -3, -1)            # CHW -> HWC
```

## 8.4 The discipline of a baseline

Here is a trap. You train a world model, generate frames, compute the
mean-squared error against the ground-truth future, and get a small number like
`0.03`. Is that good? **You cannot know without a baseline.**

For the reacher, the agent moves only a little each frame, so a *trivial* predictor
— "the future looks exactly like the last context frame" — is already quite
accurate. `sample.py` always reports this baseline next to the model:

```python
v_mse = torch.mean((pred - fut) ** 2).item()
base  = ctx[:, -1:].expand_as(fut)                  # copy the last context frame
b_mse = torch.mean((base - fut) ** 2).item()
print(f"future-frame MSE: model {v_mse:.4f}  vs  copy-last-frame {b_mse:.4f}")
```

> **A number without a baseline is not a result.** On a brief CPU run the model's
> frame MSE will likely *not* beat copy-last-frame — and that is the honest truth:
> the model is undertrained, and slow motion makes the baseline strong. The point
> of this chapter is the *pipeline* (generate → decode → compare), not a quality
> claim. Beating the baseline is a GPU-scale question we return to in Chapter 12.

There is one comparison the model *does* tend to win even at nano scale, and it is
instructive: the **action** MSE in `joint` mode is often lower than in `policy`
mode. Denoising the future alongside the action seems to help the action — a tiny
preview of FastWAM's question, which Chapter 9 measures properly in closed loop.

## 8.5 Running it

```bash
python scripts/sample.py --config configs/pusht_tiny.yaml --mode world
```

```
[sample] mode=world steps=10 n=4
[sample] future-frame MSE: model 0.0278  vs  copy-last-frame 0.0106
[sample] wrote runs/.../world_grid.png and runs/.../world_rollout.gif  (top=GT, bottom=pred)
```

The chapter demo trains a tiny model briefly, samples in world mode, decodes,
prints both MSEs, and saves a ground-truth-vs-prediction strip — the full
visualization pipeline end to end, honest baseline included.

---

## Summary

- Generation reuses the Chapter 4 Euler sampler; the **mode** selects which streams
  are denoised vs. clean conditioning (`world`, `joint`, `policy`).
- Predicted **latent tokens** become pixels by reversing the tokenizer:
  `from_tokens` then `decode`.
- `sample.py` writes a **GT-vs-prediction grid and GIF** so you can judge the model
  by eye.
- **Always report a baseline.** Copy-last-frame is a stiff bar for slow motion; a
  raw MSE means nothing without it. Beating it is a scaling question (Ch 12).
- A hint of the WAM thesis already shows up: `joint` action MSE often beats
  `policy` — imagining seems to help acting.

## Exercises

1. **Steps vs. sharpness.** Sweep `sample_steps` (2, 10, 30) and compare the
   predicted frames. Where do extra Euler steps stop helping?
2. **Beat the baseline (try).** Train longer / larger and see whether model frame
   MSE drops below copy-last-frame. What would it take?
3. **Imagining helps acting?** Print action MSE in `joint` vs `policy` mode. Form a
   hypothesis you will test in closed loop in Chapter 9.
4. **A harder world.** Modify the reacher so the agent moves faster (bigger steps).
   Does copy-last-frame get weaker? Does the model's relative advantage grow?

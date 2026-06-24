# Chapter 7 — Training nano-wam

> **This chapter covers**
> - The training loop end to end
> - Training the tokenizer by reconstruction while training the WAM by flow
>   matching on **detached** latents (the latent-diffusion split)
> - Optimization details: AdamW, warmup, AMP, gradient clipping
> - Checkpointing and a tiny-batch overfit as a sanity check

Everything is in place: data (Ch 2), a tokenizer (Ch 3), a generative objective
(Ch 4), the model (Ch 5), and the modes (Ch 6). This chapter wires them into a
training loop — the thing you actually run. The code is `scripts/train.py`, with
helpers in `nanowam/utils.py`.

---

## 7.1 The shape of one training step

A single step does six things:

1. pull a batch of windows (`ctx_frames`, `future_frames`, `actions`),
2. **tokenize** the frames into latents (Ch 3),
3. **reconstruct** them to get the tokenizer's loss,
4. draw a **mode** (Ch 6),
5. compute the **flow loss** on the detached latents (Ch 4 + 6),
6. backprop the sum and step the optimizer.

There are two losses being optimized at once, by one optimizer, over two modules
(the tokenizer and the WAM). Understanding *why it is split this way* is the heart
of the chapter.

## 7.2 Two jobs, two losses

The tokenizer and the WAM are trained by **different objectives**:

- The **tokenizer** is trained only to **reconstruct** frames — encode then decode,
  MSE against the input. It learns a good latent space; it knows nothing about the
  future or actions.
- The **WAM** is trained only by **flow matching** — predict the velocity that
  flows noise to the latent targets.

The helper that tokenizes also returns the reconstruction for the tokenizer's loss
(`scripts/train.py`):

```python
def encode_to_tokens(tokenizer, frames):
    z = tokenizer.encode(frames)       # (B, T, C, h, w)
    recon = tokenizer.decode(z)        # (B, T, 3, H, W)  -> for recon loss
    return recon, to_tokens(z)         # (B, T*P, C)      -> for the WAM
```

## 7.3 The detach: the one line that matters most

Here is the subtle, important part. The flow loss flows toward the tokenizer's
latents as its *targets*. If we let gradients from the flow loss reach back into
the encoder, the encoder could "cheat" — collapsing the latent space to whatever
is easiest to denoise rather than what faithfully represents frames. That is the
classic failure of jointly training an encoder with a latent generative model.

The fix is the **latent-diffusion split**: train the encoder by reconstruction
only, and **detach** the latents before the flow loss so no flow gradient touches
the encoder:

```python
ctx_recon, ctx_tok = encode_to_tokens(tokenizer, ctx_frames)
fut_recon, fut_tok = encode_to_tokens(tokenizer, future_frames)
recon_loss = F.mse_loss(ctx_recon, ctx_frames) + F.mse_loss(fut_recon, future_frames)

lat = {"ctx": ctx_tok.detach(), "video": fut_tok.detach(), "action": actions}
mode = sample_mode(cfg.train.mode_probs)
f_loss, metrics = flow_loss(model, lat, mode, cfg.flow)

loss = f_loss + recon_loss          # tokenizer learns from recon; WAM from flow
```

> **Do not "optimize away" the detach.** It looks redundant — both modules are in
> one optimizer — but removing it lets the flow objective corrupt the encoder and
> the whole thing degrades. This is the single most important line in the trainer.
> (Appendix B explains the latent-diffusion split formally.)

## 7.4 Optimization details

The rest of the loop is standard, deliberately so:

```python
opt = torch.optim.AdamW(list(model.parameters()) + list(tokenizer.parameters()),
                        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
scaler = torch.amp.GradScaler(enabled=use_amp)

# ... inside the loop, after computing `loss` under autocast ...
scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
scaler.step(opt); scaler.update(); sched.step()
```

- **AdamW** over *both* modules' parameters.
- **Linear warmup** of the learning rate (the velocity field is unstable early;
  warmup tames it).
- **AMP** (mixed precision) on GPU for speed; a no-op on CPU.
- **Gradient clipping** for stability.

Checkpoints (model + tokenizer + optimizer) are saved periodically via
`save_checkpoint` / `load_checkpoint` in `nanowam/utils.py`, so training can be
resumed and sampling (Ch 8) can load weights.

## 7.5 Running it

With data prepared (Ch 2), training is one command:

```bash
python scripts/train.py --config configs/pusht_tiny.yaml
```

```
[train] device=cpu ... dim=256 depth=8 max_steps=20000
[train] 3000 windows
[train] params: tokenizer=0.36M wam=24.50M
[train] step      0 loss 2.50 flow 2.03 recon 0.47 mode joint
[train] step    500 loss 1.10 flow 1.06 recon 0.03 mode world
...
```

Two things to watch in the logs. The **recon** loss should fall fast — the reacher
is easy to reconstruct — and then the **flow** loss is the one that matters,
varying by mode (policy is typically lower than joint because predicting an action
chunk is easier than imagining four frames).

## 7.6 The sanity check that never lies: overfit a tiny batch

Before trusting a training run, prove the machinery can learn *anything*: take a
single small batch and overfit it. If the loss does not fall on data the model has
seen hundreds of times, something is wrong — and you have isolated it away from
data-pipeline or generalization issues.

This is exactly what nano-wam's `test_policy_overfit_decreases` test does, and what
the chapter demo reproduces: a tiny model, a fixed batch, policy mode, ~120 steps,
and an assertion that the loss drops well below its starting value. The demo prints
the first and last loss so you can see the drop.

> **Why this is the most useful test you will write.** Overfitting a tiny batch
> separates "my model/loss/optimizer are wired correctly" from "my model
> generalizes." When something breaks, run this first.

---

## Summary

- One loop trains **two modules with two losses**: the tokenizer by
  **reconstruction**, the WAM by **flow matching**, summed and optimized together.
- The **detach** on the latents fed to the flow loss is essential — it keeps the
  flow objective from corrupting the encoder (the latent-diffusion split).
- The optimizer is plain **AdamW** with **warmup**, **AMP**, **gradient clipping**,
  and periodic **checkpoints**.
- In the logs, recon falls fast; the **flow loss is the meaningful signal**, and it
  varies by mode.
- Always **overfit a tiny batch** first: it proves the mechanism works before you
  ask about generalization.

## Exercises

1. **Break the detach.** Remove `.detach()` and train briefly. What happens to the
   reconstructions and the latents, and why?
2. **Read the curves.** Run a short training and plot the flow loss split by mode.
   Which mode has the lowest loss, and does that match your intuition from Ch 6?
3. **Warmup matters.** Set `warmup_steps` to 0 and train. Do you see instability in
   the first steps? Why is the velocity field fragile early in training?
4. **Resume.** Save a checkpoint, then resume from it with `--resume`. Confirm the
   step counter and loss continue smoothly.

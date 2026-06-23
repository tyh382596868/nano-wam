# Chapter 11 — Long-Horizon Imagination: Causal Attention & KV Caching

> **This chapter covers**
> - Why full bidirectional attention blocks efficient long rollouts
> - Block-causal masking over interleaved (action_i, frame_i) time-steps
> - Keeping the context *cacheable* (the flow time must not modulate it)
> - The KV cache (`encode_prefix`/`forward_suffix`) and autoregressive dreaming

Our model so far predicts a *fixed* horizon: `Hv` future frames in one shot. But a
world model is most compelling when it can imagine *far* — roll the future forward
indefinitely. This chapter adds the machinery that makes that efficient, following
the LingBot-VA path: **block-causal attention** plus a **KV cache**. It is the most
intricate chapter, and it pays off in `sample.py --dream N`. The code is the causal
path in `nanowam/model.py`, `flow.sample(use_cache=True)`, and `nanowam/rollout.py`.

---

## 11.1 The problem with predicting everything at once

To imagine a long future, we could just predict a longer block. But two problems
arise. First, attention is `O(N²)`: doubling the horizon quadruples the cost.
Second — and more fundamentally — to generate *autoregressively* (predict step 1,
then step 2 conditioned on step 1, …) we want to **reuse computation**: once a step
is finalized, its contribution to attention should not be recomputed every time we
extend the future.

That reuse is what a **KV cache** provides. But a KV cache is only valid if earlier
tokens do not depend on later ones — i.e., attention must be **causal**.

## 11.2 Block-causal attention over interleaved steps

nano-wam's causal mode (enabled by `model.causal`, requiring `action_horizon ==
video_horizon`) imposes a **block-causal** structure over time-steps. Conceptually
the sequence is interleaved as `(action_0, frame_0), (action_1, frame_1), …`, and:

- **context** is visible to everything and attends only to itself;
- **step `i`** (its action token and its frame's tokens) attends to the context and
  to all steps `≤ i`, but **not** to future steps.

The mask is built from a per-token *time index* — context at time `-1`, frame `i`
and action `i` at time `i` — with the rule "query attends key iff `time[key] ≤
time[query]`":

```python
def _build_causal_mask(self):
    t = self._time_index()                  # ctx=-1; frame i and action i at time i
    return (t[None, :] <= t[:, None])       # (T, T) bool: attend iff key_time <= query_time
```

The attention itself just receives this mask (Chapter 5 already plumbed an optional
`attn_mask` into `scaled_dot_product_attention`). Crucially, the mask is applied in
**training too**, so the model learns under exactly the structure used at inference.

> **Why this lets us cache.** Under block-causal attention, the context tokens
> attend only to themselves — their representation does not depend on the noised
> future at all. So the context's keys and values are the *same* no matter what the
> future looks like. That is precisely what we will cache.

## 11.3 The catch: the flow time must not touch the context

There is a subtle trap. In Chapter 5 we conditioned every stream with the flow time
`τ`. But during sampling, `τ` changes at every Euler step. If `τ` modulates the
context, then the context's representation — and thus its keys/values — would change
every step, and caching them would be wrong.

This is why, back in Chapter 5, nano-wam made the context's conditioning **exclude
the flow time**:

```python
def _cond_per_stream(self, static, time):
    # ctx gets no flow-time; video/action do
    return torch.stack([static, static + time, static + time], dim=1)
```

With the context conditioned only on `mode`/`goal` (constant across a denoising
trajectory) *and* causally isolated from the noised streams, its K/V are truly
constant across all Euler steps. Now caching them is exact.

## 11.4 The cache: `encode_prefix` and `forward_suffix`

The model splits the forward pass into two phases:

**`encode_prefix`** runs the clean context through all blocks once and returns each
layer's context keys and values:

```python
def encode_prefix(self, ctx_latents, mode, goal=None):
    ...
    cache = []
    for blk in self.blocks:
        x, kv = blk(x, sids, cond_ps, attn_mask=None, return_kv=True)
        cache.append(kv)                     # this layer's context (K, V)
    return cache
```

**`forward_suffix`** runs only the noised video/action tokens, and at each layer the
attention prepends the cached context K/V instead of recomputing them:

```python
def forward_suffix(self, video_latents, action_latents, tau, mode, prefix_cache, goal=None):
    ...
    mask = self.full_mask[self.n_ctx:, :]    # suffix queries, all keys [ctx; suffix]
    for blk, kv in zip(self.blocks, prefix_cache):
        x = blk(x, sids, cond_ps, attn_mask=mask, prefix_kv=kv)
    return self._readout(x)
```

The sampler uses this automatically when you pass `use_cache=True`: compute the
prefix once, then reuse it across all `S` Euler steps.

```python
prefix = model.encode_prefix(cond["ctx"], mode_id, goal) if cached else None
for i in range(steps):
    if cached:
        v_video, v_action = model.forward_suffix(x_v, x_a, tau, mode_id, prefix, goal)
    else:
        v_video, v_action = model(cond["ctx"], x_v, x_a, tau, mode_id, goal)
    ...
```

> **This is the single most important correctness claim in the chapter:** the cached
> path is *numerically identical* to the uncached one, not an approximation. The
> chapter demo (and the repo's `test_kv_cache_forward_equivalence`) asserts
> `cached == uncached` to a tolerance of `1e-5`.

## 11.5 Dreaming: rolling the future forward

With efficient block prediction in hand, long-horizon imagination is a loop:
predict a block of `Hv` frames, decode them, slide the context window onto the last
`K` predicted frames, and predict the next block — for as many blocks as you like
(`nanowam/rollout.py`):

```python
ctx = init_ctx_frames.unsqueeze(0)
for _ in range(n_blocks):
    ctx_tok = to_tokens(tokenizer.encode(ctx))
    out = sample(model, {"ctx": ctx_tok}, mode, cfg.flow, use_cache=True)
    block = tokenizer.decode(from_tokens(out["video"], Hv, h, h))[0]   # (Hv, 3, H, W)
    frames_out.append(block)
    ctx = block[-K:].unsqueeze(0)             # slide the window onto our own prediction
```

In `JOINT` mode the model also imagines the actions, so the dream is autonomous —
no external action sequence needed. The model is now feeding on its own predictions,
the essence of a world model "dreaming."

```bash
python scripts/sample.py --config configs/causal_tiny.yaml --dream 8   # 8 blocks -> long gif
```

## 11.6 What the cache buys, honestly

The KV cache avoids recomputing the context's K/V on every Euler step. On a small
CPU model the speedup is modest (~15% in our measurements) because the context is
only a third of the tokens and the model is tiny; the win grows with model size,
context length, and sampling steps. The more important property is **correctness**:
the cache is exact, so you can enable it for free.

> **A note on what's verified.** The causal mechanism, the exact cache, and the
> dream loop are all CPU-tested. Whether the *dreamed frames stay coherent* over
> many blocks is, again, a quality question that depends on a well-trained model —
> a GPU-scale matter (Chapter 12). The apparatus is correct; the quality is a
> turn-of-the-crank away.

---

## Summary

- Efficient, unbounded autoregressive rollout needs **causal** attention so earlier
  tokens do not depend on later ones — the precondition for a **KV cache**.
- nano-wam's **block-causal** mask lets context see itself, and step `i` see context
  + steps `≤ i`; it is applied in training and inference alike.
- For the cache to be valid, the **flow time must not modulate the context** (a
  Chapter 5 design choice), so context K/V are constant across Euler steps.
- **`encode_prefix`** caches the context's per-layer K/V once; **`forward_suffix`**
  reuses them across every step. The cached path is **exactly** equal to the
  uncached one (`atol 1e-5`).
- **`dream_rollout`** slides the context window onto the model's own predictions to
  imagine arbitrarily far. The cache speedup is modest at nano scale but grows; its
  real value is exactness.

## Exercises

1. **Read the mask.** For `Hv = Ha = 4`, write out the time index of every token and
   sketch the `(T, T)` causal mask. Which entries are blocked, and why?
2. **Break the isolation.** What goes wrong with the cache if you let the flow time
   `τ` modulate the context stream? Why does the equivalence test catch it?
3. **Measure the speedup.** Time `sample(..., use_cache=True)` vs `False` as you grow
   `dim`, context length, and `sample_steps`. When does the cache start to matter?
4. **Coherent dreams.** Run `--dream` with more blocks. Where does the dream drift,
   and what would you expect a well-trained, larger model to do differently?

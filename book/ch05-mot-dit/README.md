# Chapter 5 — Coding the MoT Diffusion Transformer

> **This chapter covers**
> - Embedding three token streams: context, video, and action
> - Self-attention and AdaLN-zero conditioning on the flow time
> - The **Mixture-of-Transformers** block: shared attention, per-stream FFN/AdaLN
> - Assembling `NanoWAM` and why the velocity heads are zero-initialized

Chapter 4 gave us the generative engine — a velocity field trained by flow
matching. But that engine was a tiny MLP on 2-D points. To model *future frames
and actions conditioned on context*, we need a real velocity network: a
conditional transformer. This chapter builds it, the core of nano-wam, in
`nanowam/model.py`.

The architecture has a name in the literature: a **Diffusion Transformer (DiT)**,
specialized into a **Mixture-of-Transformers (MoT)** so that different modalities
(video, action) keep their own weights while still attending to one another.

---

## 5.1 Three streams, one sequence

Recall the window contract: `(context, future, action)`. After tokenizing
(Chapter 3), each becomes a sequence of vectors. The model concatenates all three
into one long sequence and processes them together:

```
x = [ ctx tokens | video tokens | action tokens ]
        clean         noised          noised
     (K·P vectors)  (Hv·P vectors)  (Ha vectors)
```

- **context** tokens are the latents of the history frames — always *clean*
  conditioning (never noised).
- **video** tokens are the latents of the future frames — *noised* during
  training, the world stream the model learns to denoise.
- **action** tokens are the action chunk projected to model width — *noised*, the
  policy stream.

Each stream gets its own input projection and a learned positional embedding, then
they are concatenated. From `nanowam/model.py`:

```python
self.ctx_proj    = nn.Linear(C, d)     # latent channels -> model dim
self.video_proj  = nn.Linear(C, d)
self.action_proj = nn.Linear(Da, d)    # action dim -> model dim

self.ctx_pos    = nn.Parameter(torch.zeros(1, self.n_ctx, d))
self.video_pos  = nn.Parameter(torch.zeros(1, self.n_video, d))
self.action_pos = nn.Parameter(torch.zeros(1, self.n_action, d))
```

A small bookkeeping detail that matters later: the model records *which stream
each token belongs to* with a `stream_ids` vector (`0` = context, `1` = video,
`2` = action). That id is what routes the per-stream weights below.

## 5.2 Conditioning: time, mode, and goal

A flow-matching model must know the **noise level** `τ` it is operating at — the
same `τ` from Chapter 4. We also condition on which **mode** we are running
(Chapter 6) and optionally a **goal**. These are combined into a conditioning
vector.

The flow time is turned into a vector with a sinusoidal embedding (the same trick
the 2-D demo used), then an MLP:

```python
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

nano-wam makes one deliberate refinement here that pays off in Chapter 11:

> **The flow time conditions only the *noised* streams.** Context is clean — it is
> not being denoised — so its conditioning excludes `τ`. The conditioning is built
> per stream: context gets `mode (+ goal)`; video and action get `mode (+ goal) +
> time`. This is why, later, the context's representation is constant across
> denoising steps and can be cached.

## 5.3 AdaLN-zero: conditioning that starts as identity

How does the conditioning vector actually influence the transformer? Through
**adaptive layer normalization (AdaLN)**: the conditioning produces per-channel
`shift` and `scale` applied after normalization, plus a `gate` on each residual
branch.

```python
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
```

The "**-zero**" part is crucial: the network that produces these modulation
parameters is **initialized to zero**, so at the start of training every block is
an identity map and the residual branches are gated off. The model begins as a
no-op and *learns* how much conditioning to apply. This is a standard DiT trick
for stable training, and nano-wam keeps it.

## 5.4 The Mixture-of-Transformers block

Here is the heart of the architecture, and the idea nano-wam borrows from
LingBot-VA and Motus. A normal transformer block applies the *same* attention and
the *same* feed-forward network to every token. An **MoT** block instead:

- **shares the self-attention** across all streams — so video tokens and action
  tokens attend to each other and to the context (this is where world and action
  *talk*), but
- **routes the feed-forward and AdaLN per stream** — context, video, and action
  each have their own FFN and their own modulation weights.

Intuitively: one shared "communication channel" (attention), but each modality
keeps its own "private processing" (FFN). The block, lightly abridged:

```python
class MoTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn  = Attention(cfg)                       # shared across streams
        self.ffn   = nn.ModuleList(MLP(d) for _ in range(3))   # one per stream
        self.adaln = nn.ModuleList(nn.Linear(d, 6 * d) for _ in range(3))
        for lin in self.adaln:                            # zero-init -> identity start
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, x, stream_ids, cond_ps, attn_mask=None):
        sh1, sc1, g1, sh2, sc2, g2 = self._stream_modulation(stream_ids, cond_ps, ...)
        x = x + g1 * self.attn(modulate(self.norm1(x), sh1, sc1), stream_ids, attn_mask)
        # per-stream FFN: route each token to its stream's MLP
        h = modulate(self.norm2(x), sh2, sc2)
        ffn_out = h.new_zeros_like(h)
        for s in range(3):
            m = stream_ids == s
            ffn_out[:, m] = self.ffn[s](h[:, m])
        x = x + g2 * ffn_out
        return x
```

`_stream_modulation` gathers each token's six AdaLN parameters from *its* stream's
modulation network. The per-stream FFN loop sends each token through *its* MLP.
The attention, by contrast, sees the whole sequence at once.

The attention itself is ordinary scaled-dot-product self-attention — with two
hooks we will not need until Chapter 11 (an optional causal `attn_mask`, and an
optional cached prefix):

```python
class Attention(nn.Module):
    def forward(self, x, stream_ids, attn_mask=None, prefix_kv=None, return_kv=False):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # ... reshape to heads ...
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out)
```

## 5.5 Assembling `NanoWAM`

The full model embeds the three streams, builds the conditioning, runs `N` MoT
blocks, and reads out a velocity per noised stream:

```python
class NanoWAM(nn.Module):
    def forward(self, ctx_latents, video_latents, action_latents, tau, mode, goal=None):
        ctx    = self.ctx_proj(ctx_latents)     + self.ctx_pos
        suffix = self._embed_suffix(video_latents, action_latents)  # video+action
        x = torch.cat([ctx, suffix], dim=1)
        stream_ids = ...                          # 0=ctx, 1=video, 2=action
        cond_ps = self._cond_per_stream(*self._cond_parts(tau, mode, goal))
        for blk in self.blocks:
            x = blk(x, stream_ids, cond_ps)
        return self._readout(x[:, self.n_ctx:])   # -> (v_video, v_action)
```

The readout heads turn the final hidden states of the video and action tokens into
**velocities** — the same `v_θ` from Chapter 4, now produced by a conditional
transformer:

```python
self.video_head  = nn.Linear(d, C)
self.action_head = nn.Linear(d, Da)
for head in (self.video_head, self.action_head):
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)   # zero-init
```

> **Why zero-init the heads?** At step 0 the model should predict *zero velocity*
> — i.e., do nothing — which is the safest possible starting point for the ODE
> integrator. Combined with AdaLN-zero, the entire network begins as a clean
> identity/no-op and learns away from there. The chapter demo verifies that a
> freshly built model outputs exactly zero velocity.

## 5.6 A forward pass

Putting it together, a forward pass maps (clean context, noised video, noised
action, time, mode) to per-stream velocities:

```python
import torch
from nanowam.config import load_config
from nanowam.model import NanoWAM

cfg = load_config("configs/pusht_tiny.yaml")
model = NanoWAM(cfg.model, cfg.data)
print(f"{model.num_params()/1e6:.1f}M parameters")     # ~24.5M at defaults

B, P, C = 2, model.P, cfg.model.latent_channels
ctx = torch.randn(B, model.n_ctx, C)
vid = torch.randn(B, model.n_video, C)
act = torch.randn(B, model.n_action, cfg.data.action_dim)
tau = torch.rand(B)
mode = torch.zeros(B, dtype=torch.long)

v_vid, v_act = model(ctx, vid, act, tau, mode)
print(v_vid.shape, v_act.shape)                          # match vid, act
print("init velocity max:", v_vid.abs().max().item())    # exactly 0.0
```

The runnable demo in `01_main-chapter-code/` runs exactly this, confirms the
output shapes, and asserts the zero-init property — the model truly starts by
predicting nothing.

---

## Summary

- nano-wam's velocity network is a **conditional Diffusion Transformer** over a
  single sequence of **context + video + action** tokens.
- Each stream has its own input projection and positional embedding; a
  `stream_ids` vector tags every token's modality.
- Conditioning (time, mode, goal) enters through **AdaLN-zero**, which starts as
  identity. The flow time conditions only the *noised* streams, leaving context
  cacheable (Chapter 11).
- The **Mixture-of-Transformers** block **shares attention** (so streams talk) but
  **routes the FFN and AdaLN per stream** (so each modality keeps its own weights).
- The velocity **heads are zero-initialized**, so the model predicts zero velocity
  at step 0 — the safest start for the flow ODE.

## Exercises

1. **Count the parameters.** Where do nano-wam's ~24.5M parameters go? Note that
   the per-stream FFN triples the feed-forward weights. Print `num_params()` as
   you scale `dim` and `depth`.
2. **Verify the zero start.** Build a fresh `NanoWAM` and confirm `v_video` and
   `v_action` are exactly zero. Which two design choices guarantee this?
3. **Shared vs. routed attention.** The config has `route_attention`. Read what it
   does in `Attention`, toggle it on, and reason about the parameter cost and what
   it might buy.
4. **Why share attention but split FFN?** In your own words, explain the MoT bet:
   what is gained by letting streams attend to each other while keeping their
   feed-forward weights separate?

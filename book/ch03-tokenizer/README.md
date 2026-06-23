# Chapter 3 — From Frames to Latent Tokens: The Tokenizer

> **This chapter covers**
> - Why we model in a compressed *latent* space instead of raw pixels
> - Building a tiny convolutional autoencoder (the Wan-VAE stand-in)
> - Encoding 64×64 frames to a small latent grid and decoding back
> - Turning a latent grid into transformer tokens (`to_tokens` / `from_tokens`)

Chapter 2 gave us windows of raw frames. But a transformer does not consume
`64×64×3` images directly — it consumes a *sequence of vectors* (tokens). This
chapter builds the bridge: a small autoencoder that compresses each frame into a
handful of latent vectors, and the reshaping that turns those vectors into a token
sequence. It is the exact analog of the *tokenizer* chapter in a language-model
book — except our "text" is pixels.

All code here lives in `nanowam/tokenizer.py`.

---

## 3.1 Why not just use pixels?

A `64×64×3` frame has 12,288 numbers. If every pixel were a token, a 4-frame
future would be ~49k tokens, and self-attention — which costs `O(N²)` — would be
hopeless even at this toy scale. Worse, raw pixels are a *redundant* and *noisy*
representation: neighboring pixels are highly correlated, and tiny pixel-level
differences carry little meaning.

The standard fix, inherited from **latent diffusion**, is to model in a
compressed latent space. We learn an **autoencoder**:

- an **encoder** that maps a frame to a small grid of latent vectors, and
- a **decoder** that maps that grid back to a frame.

The big WAMs use a powerful pretrained video VAE (e.g. Wan's) for this. nano-wam
uses a tiny conv autoencoder we train from scratch — the smallest thing that
demonstrates the idea.

> **Deterministic, not variational.** A "VAE" adds a sampled latent with a KL
> penalty. At nano scale that machinery buys us nothing, so our tokenizer is a
> plain **deterministic** autoencoder (encode → decode, MSE reconstruction). We
> keep the name "tokenizer" because its *job* is to produce tokens.

## 3.2 A tiny conv autoencoder

The encoder is a stack of stride-2 convolutions that halve the spatial size each
step, then a final conv that projects to `C` latent channels. With image size 64
and a downsample factor of 8 (= three stride-2 stages), a frame becomes an `8×8`
grid of `C`-dimensional vectors.

The key design decision is to tie the downsample factor to the model's
`patch_size`:

> **`patch_size` == the tokenizer's downsample factor.** Because of this, each
> latent grid cell *is* exactly one token for the model — there is no separate
> "patchify" step to keep consistent. The number of tokens per frame is
> `P = (image_size / patch_size)²` — here `(64/8)² = 64`.

Here is the constructor that builds the symmetric encoder/decoder
(`nanowam/tokenizer.py`):

```python
class FrameTokenizer(nn.Module):
    def __init__(self, model_cfg, data_cfg):
        super().__init__()
        C = model_cfg.latent_channels
        f = model_cfg.patch_size              # downsample factor, e.g. 8
        n_down = int(round(math.log2(f)))     # -> three stride-2 stages
        assert 2 ** n_down == f, "patch_size must be a power of 2"
        self.latent_hw = data_cfg.image_size // f

        chans = [3, 32, 64, 128][: n_down + 1]

        # Encoder: n_down stride-2 convs, then project to C channels.
        enc = []
        for i in range(n_down):
            enc += [nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1),
                    nn.GroupNorm(8, chans[i + 1]), nn.SiLU()]
        enc += [nn.Conv2d(chans[n_down], C, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        # Decoder: project from C, then n_down transposed convs back to RGB.
        dec = [nn.Conv2d(C, chans[n_down], 3, padding=1), nn.SiLU()]
        for i in range(n_down, 0, -1):
            out_ch = chans[i - 1] if i - 1 > 0 else 32
            dec += [nn.ConvTranspose2d(chans[i], out_ch, 4, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch), nn.SiLU()]
        dec += [nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()]  # frames in [0, 1]
        self.decoder = nn.Sequential(*dec)
```

Two small things worth noticing. The decoder ends in a **`Sigmoid`**, so
reconstructions land in `[0, 1]` — matching the float frames from Chapter 2.
And we use **`GroupNorm` + `SiLU`** rather than BatchNorm, because we will run
this on tiny batches and even single frames; GroupNorm does not depend on batch
statistics.

## 3.3 Handling the time dimension

Our windows are not single frames — they are stacks like `(B, Hv, 3, H, W)`. The
encoder is a 2-D conv, so we fold any leading dimensions into the batch, run the
conv, and unfold:

```python
def _fold(self, x):
    *lead, c, h, w = x.shape
    return x.reshape(-1, c, h, w), tuple(lead)

def encode(self, frames):
    """frames (..., 3, H, W) -> latents (..., C, H/f, W/f)."""
    x, lead = self._fold(frames)
    z = self.encoder(x)
    return z.reshape(*lead, *z.shape[1:])

def decode(self, latents):
    """latents (..., C, H/f, W/f) -> frames (..., 3, H, W)."""
    z, lead = self._fold(latents)
    x = self.decoder(z)
    return x.reshape(*lead, *x.shape[1:])
```

This little trick — fold, apply, unfold — means `encode`/`decode` work
transparently on `(B, 3, H, W)`, `(B, T, 3, H, W)`, or any leading shape. The
latent mirrors whatever leading dims you gave it.

## 3.4 Latent grids *are* token sequences

A transformer wants a sequence `(B, N, C)`: `N` tokens, each a `C`-vector. Our
latent is a grid `(B, T, C, h, w)`. Flattening the spatial dimensions turns each
cell into a token; we keep the channel dimension as the token's feature vector:

```python
def to_tokens(grids):
    """(B, T, C, h, w) -> (B, T*h*w, C). Spatial cells become tokens."""
    B, T, C, h, w = grids.shape
    return grids.permute(0, 1, 3, 4, 2).reshape(B, T * h * w, C)

def from_tokens(tokens, T, h, w):
    """(B, T*h*w, C) -> (B, T, C, h, w). Inverse of to_tokens."""
    B, N, C = tokens.shape
    assert N == T * h * w, f"token count {N} != T*h*w {T*h*w}"
    return tokens.reshape(B, T, h, w, C).permute(0, 1, 4, 2, 3).contiguous()
```

So a `Hv=4`-frame future with an `8×8×C` latent per frame becomes `4·64 = 256`
tokens. That is the sequence the model in Chapter 5 will denoise.

> **Where this fits.** The training loop (Chapter 7) calls `encode` to turn frames
> into latents, then `to_tokens` to feed the model. Sampling (Chapter 8) does the
> reverse: the model predicts latent tokens, `from_tokens` reshapes them to a
> grid, and `decode` turns them back into pixels you can look at.

## 3.5 A sanity check

A fresh, untrained tokenizer cannot reconstruct anything meaningful — but it
should at least preserve shapes and output valid pixels in `[0, 1]`:

```python
import torch
from nanowam.config import load_config
from nanowam.tokenizer import FrameTokenizer, to_tokens

cfg = load_config("configs/pusht_tiny.yaml")
tok = FrameTokenizer(cfg.model, cfg.data)

frames = torch.rand(2, cfg.data.video_horizon, 3, 64, 64)  # (B, Hv, 3, H, W)
z = tok.encode(frames)
print(z.shape)              # (2, 4, 4, 8, 8)   -> C=4, 8x8 grid
recon = tok.decode(z)
print(recon.shape)          # (2, 4, 3, 64, 64)
print(recon.min().item(), recon.max().item())  # within [0, 1]

tokens = to_tokens(z)
print(tokens.shape)         # (2, 256, 4)   = (B, Hv*64, C)
```

After joint training in Chapter 7, `decode(encode(frame))` will actually
reconstruct the reacher scene. The runnable demo in this chapter's
`01_main-chapter-code/` trains the tokenizer for a few hundred steps on the
reconstruction loss alone and shows the input-vs-reconstruction strip going from
noise to recognizable squares.

---

## Summary

- Modeling raw pixels is wasteful and ill-conditioned; like latent diffusion, we
  compress frames with an **autoencoder** and model in latent space.
- nano-wam's tokenizer is a **deterministic** conv autoencoder (encode → decode,
  MSE), the smallest stand-in for a pretrained video VAE.
- Setting **`patch_size` = the downsample factor** makes each latent grid cell a
  token directly; a frame yields `P = (image_size / patch_size)²` tokens.
- A fold/apply/unfold trick lets `encode`/`decode` handle arbitrary leading
  (batch/time) dimensions.
- `to_tokens` / `from_tokens` convert between the latent grid `(B, T, C, h, w)`
  and the token sequence `(B, T·h·w, C)` the transformer consumes.

## Exercises

1. **Compression ratio.** A `64×64×3` frame is 12,288 numbers; an `8×8×C=4` latent
   is 256. What is the compression ratio? How many tokens does a `Hv=4` future
   become, and how does that compare to one-token-per-pixel?
2. **Capacity vs. fidelity.** Change `latent_channels` to 2 and to 8, train the
   tokenizer briefly (see the chapter demo), and compare reconstruction MSE.
3. **Why the sigmoid?** Remove the final `Sigmoid` and retrain the tokenizer.
   What goes wrong with the reconstructions, and why does `[0, 1]` output matter
   given the float frames from Chapter 2?
4. **The round-trip.** Show, in code, that `from_tokens(to_tokens(z)) == z` for a
   random latent `z`. Why is `.contiguous()` used in `from_tokens`?

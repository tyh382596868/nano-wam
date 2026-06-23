# Chapter 3 — From Frames to Latent Tokens: The Tokenizer

> **This chapter covers**
> - Why we model in a compressed *latent* space instead of raw pixels
> - Building a tiny convolutional autoencoder (the Wan-VAE stand-in)
> - Encoding 64×64 frames to an 8×8×C latent grid and decoding back
> - Turning a latent grid into transformer tokens (`to_tokens` / `from_tokens`)

**Builds:** `FrameTokenizer` (conv AE) + the token/grid converters.
**Maps to:** `nanowam/tokenizer.py`.

## Outline
1. Latent diffusion in one paragraph: why compress first.
2. A symmetric conv encoder/decoder; choosing the downsample factor = `patch_size`.
3. Reconstruction as the tokenizer's only job (foreshadowing the detached-latent
   training split in Chapter 7).
4. Latent cells *are* tokens: flattening the grid; the round-trip helpers.
5. A sanity check: encode → decode a reacher frame and eyeball the reconstruction.

## Summary & Exercises
- Summary: frames become a short sequence of latent tokens.
- Exercises: change `latent_channels`; measure reconstruction MSE vs. capacity;
  swap the AE for a pixel-passthrough and discuss the trade-off.

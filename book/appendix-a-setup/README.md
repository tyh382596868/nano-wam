# Appendix A — Setup & PyTorch Essentials

> **This appendix covers**
> - Installing the environment (CPU is fine for everything but scale)
> - Running the tests and the scripts from the repo root
> - The few PyTorch idioms used throughout the book

**Maps to:** `requirements.txt`, `CLAUDE.md` (commands & conventions).

## Outline
1. Python/venv, `pip install -r requirements.txt`, CPU vs CUDA torch.
2. Running from the repo root; `python -m pytest tests/ -q`.
3. PyTorch idioms used: `nn.Module`, broadcasting, `einops`-free reshapes,
   `scaled_dot_product_attention`, autocast/AMP, `no_grad`.
4. Reproducibility: seeding and determinism caveats.

## Summary
A 10-minute setup so every code listing in the book runs.

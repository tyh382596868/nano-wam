# Appendix A — Setup & PyTorch Essentials

> **This appendix covers**
> - Installing the environment (CPU is fine for everything but scale)
> - Running the tests and the scripts from the repo root
> - The few PyTorch idioms used throughout the book

This appendix is a ten-minute on-ramp so every code listing and chapter demo runs.

---

## A.1 Installing

nano-wam does not vendor PyTorch. Create a virtual environment and install the
requirements:

```bash
python -m venv .venv && source .venv/bin/activate   # or conda, uv, etc.
pip install -r requirements.txt
```

The dependencies are deliberately light: `torch`, `numpy`, `pyyaml`, `tqdm`,
`imageio`, and `pytest`. A **CPU build of PyTorch is enough** for every test and
every chapter demo in this book — they are tiny by design. You only need a GPU for
the *scale* run in Chapter 12 (`scripts/run_gpu.sh`). Optional extras: `matplotlib`
(for the Chapter 4 scatter) and `lerobot` (only for the real-data adapter in
Chapter 10).

## A.2 Running things from the repo root

Two conventions matter, and skipping them causes the most common beginner errors:

1. **Run everything from the repository root.** The scripts insert the root onto
   `sys.path` themselves, and the chapter demos do the same, so `import nanowam`
   resolves. The configs also use paths relative to the root.

2. **Invoke pytest as a module:**

   ```bash
   python -m pytest tests/ -q
   ```

   `python -m pytest` puts the current directory (the repo root) on `sys.path`, so
   `import nanowam` and the relative config paths in the tests both resolve. Bare
   `pytest` may not.

A quick smoke test that everything is wired up:

```bash
python -m pytest tests/ -q              # the full contract suite (should be all green)
python scripts/prepare_data.py --config configs/pusht_tiny.yaml --episodes 50
python scripts/train.py --config configs/pusht_tiny.yaml          # Ctrl-C after a few steps
```

Each chapter's `01_main-chapter-code/` script is standalone and also run from the
root, e.g. `python book/ch05-mot-dit/01_main-chapter-code/model_demo.py`.

## A.3 The PyTorch you need

The book uses a small, stable subset of PyTorch. If these are familiar, you are
ready:

- **`nn.Module`**: subclass it, register layers in `__init__`, define `forward`.
  Parameters are tracked automatically.
- **Tensor shapes and broadcasting**: most of the code is shape bookkeeping. The
  habit of writing the shape in a comment after every nontrivial line pays off.
- **Reshapes without `einops`**: `reshape`, `permute`, `transpose`, and
  `.contiguous()`. The tokenizer's `to_tokens`/`from_tokens` (Chapter 3) are pure
  reshapes.
- **`F.scaled_dot_product_attention`**: PyTorch's fused attention, used in
  Chapter 5. It optionally takes a boolean `attn_mask` (Chapter 11).
- **Autograd**: `loss.backward()` then `optimizer.step()`; `.detach()` to stop
  gradient flow (the critical line in Chapter 7); `@torch.no_grad()` for inference.
- **Mixed precision**: `torch.amp.autocast` and `GradScaler` (Chapter 7) — a no-op
  on CPU, a speedup on GPU.

## A.4 Reproducibility

`nanowam/utils.py:set_seed` seeds Python, NumPy, and PyTorch (and CUDA) so toy runs
are repeatable:

```python
from nanowam.utils import set_seed
set_seed(1337)
```

Exact bit-for-bit determinism across machines and backends is not guaranteed (CPU
vs GPU kernels differ), but seeding makes a single machine's runs reproducible —
enough to debug and to compare changes.

---

## Summary

Install the light requirements (CPU torch is fine), run everything **from the repo
root**, invoke tests as **`python -m pytest`**, and you have what you need. The
PyTorch subset is small: modules, shapes/broadcasting, reshapes, fused attention,
autograd with `.detach()`, and AMP. Seed with `set_seed` for reproducible runs.

# CLAUDE.md — orientation for AI coding sessions

nano-wam is a minimal, from-scratch, single-GPU **World Action Model** — the
`nanoGPT` of WAMs. It jointly predicts future video frames + actions with an MoT
Diffusion Transformer trained by rectified flow matching.

**Read order:** `PROJECT.md` (why/how/story) → `DESIGN.md` (the spec) →
`README.md` (run it). This file is the quick agent briefing.

## Commands

```bash
python -m pytest tests/ -q                 # full suite (run from repo ROOT)
python scripts/prepare_data.py --config configs/pusht_tiny.yaml --episodes 200
python scripts/train.py        --config configs/pusht_tiny.yaml
python scripts/sample.py       --config configs/pusht_tiny.yaml --mode world
python scripts/sample.py       --config configs/pusht_tiny.yaml --closed-loop --episodes 50
./scripts/run_gpu.sh                        # one-click GPU: data→train→viz→ablation
```

torch is not vendored; `pip install -r requirements.txt` (CPU torch is fine for
everything except scale). CI = `.github/workflows/ci.yml` runs pytest on push.

## Repo map (start here)

- `nanowam/model.py` — the core: `NanoWAM` (MoT DiT), `MoTBlock`, `Attention`
  (+ causal mask, KV-cache `encode_prefix`/`forward_suffix`).
- `nanowam/flow.py` — rectified-flow `interpolate` / `flow_loss` / `sample`
  (Euler ODE; `use_cache=True` for causal models).
- `nanowam/modes.py` — the 4 modes + per-mode stream masks + `sample_mode`.
- `nanowam/tokenizer.py` — conv AE + `to_tokens`/`from_tokens`.
- `nanowam/data.py` / `sources.py` — windows + `EpisodeSource` (Reacher, LeRobot).
- `nanowam/envs.py` / `eval.py` — closed-loop env + imagination ablation.
- `nanowam/rollout.py` — causal KV-cache long-horizon `dream_rollout`.
- `configs/` — `pusht_tiny` (CPU default), `reacher_gpu`, `lerobot_pusht`,
  `causal_tiny`. `tests/test_shapes.py` — the executable contracts.

## Conventions & gotchas (don't relearn these the hard way)

- **Run scripts from the repo root.** They self-insert the root on `sys.path`;
  pytest must be invoked as `python -m pytest` so `import nanowam` resolves.
- **Latents are detached for the flow loss** (tokenizer trained by recon only) —
  standard latent-diffusion split; don't "fix" it by removing the detach.
- **Velocity heads are zero-init** → 0 velocity at step 0 (stable start). Keep it.
- **Causal mode requires `action_horizon == video_horizon`** (interleaved steps)
  and the flow time must NOT modulate the context stream (so context K/V are
  cacheable). The KV cache is exact — `tests/` asserts cached == uncached.
- **Everything speaks the `(ctx, future, action)` window contract.** New data
  sources = new `EpisodeSource`; nothing downstream changes.
- The default **reacher** task is procedural/self-contained (no external deps) so
  tests and CI run anywhere; pushT/LIBERO come via the LeRobot adapter.

## State (as of M6)

All milestones M0–M6 complete; 20 tests green; CI green. **Mechanisms are
verified on CPU; quality/ablation numbers are NOT** — they need a GPU-scale run
(`scripts/run_gpu.sh`). See `PROJECT.md` "What's verified vs what isn't".

## Working agreements

- Develop on the feature branch; don't push to `main` without explicit ask.
- Commit messages end with the Co-Authored-By / Claude-Session trailers.
- Keep new code matching the surrounding style; add a test for new mechanisms.
- Be honest about verified vs. assumed — this project's value is that every claim
  in the docs is backed by a test or a real run.

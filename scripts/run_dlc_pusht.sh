#!/usr/bin/env bash
# nano-wam — DLC entrypoint for the scaled pushT run (M7).
# Submitted via the dlc helper; submit auto-runs dlc_init.sh first to align the
# container paths to /x2robot_v2/niko/... (the shared cpfs), so this repo, its
# prepared data, and runs/ are all visible and persist across preemption.
set -uo pipefail

REPO=/x2robot_v2/niko/code/nano-wam
cd "$REPO"

export PYTHONUNBUFFERED=1
# headless rendering for gym-pusht (pygame) inside the container
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
# gym-pusht + opencv were prebuilt onto cpfs from zbl (numpy/torch pinned), so the
# container needs no internet; torch/numpy come from the image, not from here.
export PYTHONPATH="$REPO/.dlc_deps:${PYTHONPATH:-}"

CFG=configs/pusht_dlc.yaml
OUT=runs/pusht_dlc

echo "==> GPU"; nvidia-smi -L || true
echo "==> python: $(which python)"; python --version
echo "==> torch/cuda + eval deps"
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import gym_pusht, cv2, gymnasium  # noqa: F401
    print("[run] gym_pusht + cv2 + gymnasium import OK")
except Exception as e:
    print(f"[run] WARNING: eval deps import failed ({type(e).__name__}: {e}); "
          f"training will run, in-loop eval will be skipped")
PY

# preemption-safe: resume from the newest numbered checkpoint if one exists
RESUME=""
LATEST=$(ls -1 ${OUT}/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "${LATEST:-}" ]; then
  RESUME="--resume ${LATEST}"
  echo "==> resuming from ${LATEST}"
else
  echo "==> fresh start (no checkpoint in ${OUT})"
fi

echo "==> training: $CFG"
python scripts/train.py --config "$CFG" ${RESUME}

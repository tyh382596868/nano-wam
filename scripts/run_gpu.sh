#!/usr/bin/env bash
# nano-wam — one-click GPU run: deps -> data -> train -> visualize -> ablation.
#
# Usage:
#   ./scripts/run_gpu.sh                         # default: configs/reacher_gpu.yaml
#   ./scripts/run_gpu.sh configs/causal_tiny.yaml
#   EPISODES=4000 ./scripts/run_gpu.sh
#   SKIP_DEPS=1 ./scripts/run_gpu.sh             # skip pip install
#
# Everything is driven by the YAML config; artifacts land in its train.out_dir.
set -euo pipefail

CONFIG="${1:-configs/reacher_gpu.yaml}"
EPISODES="${EPISODES:-2000}"
EP_LEN="${EP_LEN:-32}"

cd "$(dirname "$0")/.."   # repo root (scripts run from here, no install needed)

echo "==> config: $CONFIG   episodes: $EPISODES"

echo "==> GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "WARNING: no nvidia-smi found — training will fall back to CPU (slow)."
fi

if [ "${SKIP_DEPS:-0}" != "1" ]; then
  echo "==> installing dependencies"
  python -m pip install -q -r requirements.txt
fi

echo "==> 1/4 preparing data"
python scripts/prepare_data.py --config "$CONFIG" --episodes "$EPISODES" --ep-len "$EP_LEN"

echo "==> 2/4 training"
python scripts/train.py --config "$CONFIG"

echo "==> 3/4 open-loop world rollout (GT vs prediction gif/grid)"
python scripts/sample.py --config "$CONFIG" --mode world

echo "==> 4/4 closed-loop imagination ablation (none vs joint)"
python scripts/sample.py --config "$CONFIG" --closed-loop --episodes 50

# If the config is causal (action_horizon == video_horizon), also dream long-horizon.
if python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
sys.exit(0 if cfg.get("model", {}).get("causal") else 1)
PY
then
  echo "==> bonus: causal model — autoregressive long-horizon dream"
  python scripts/sample.py --config "$CONFIG" --dream 8
fi

OUT=$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["train"]["out_dir"])
PY
)
echo "==> done. checkpoints + gifs in: $OUT"

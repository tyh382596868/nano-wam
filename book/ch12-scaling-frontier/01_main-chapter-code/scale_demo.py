"""Chapter 12 — runnable demo: scaling is a config change.

Builds NanoWAM at the CPU-tiny and GPU configs and prints the parameter counts
side by side, showing that the architecture is identical -- only dim/depth/batch
change. No training.

Run from the repo root:
    python book/ch12-scaling-frontier/01_main-chapter-code/scale_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanowam.config import load_config  # noqa: E402
from nanowam.model import NanoWAM  # noqa: E402
from nanowam.tokenizer import FrameTokenizer  # noqa: E402

configs = [
    ("CPU tiny (pusht_tiny)", "configs/pusht_tiny.yaml"),
    ("GPU scaled (reacher_gpu)", "configs/reacher_gpu.yaml"),
]

print(f"{'config':<28}{'dim':>5}{'depth':>7}{'batch':>7}{'amp':>5}{'WAM params':>14}")
print("-" * 72)
for name, path in configs:
    cfg = load_config(path)
    model = NanoWAM(cfg.model, cfg.data)
    tok = FrameTokenizer(cfg.model, cfg.data)
    wam_m = model.num_params() / 1e6
    tok_m = sum(p.numel() for p in tok.parameters()) / 1e6
    print(f"{name:<28}{cfg.model.dim:>5}{cfg.model.depth:>7}{cfg.train.batch_size:>7}"
          f"{str(cfg.train.amp):>5}{wam_m:>11.1f}M  (+{tok_m:.2f}M tok)")

print("\nSame architecture, different config -- scaling is an edit, not a rewrite.")
print("Run the full GPU pipeline with:  ./scripts/run_gpu.sh")

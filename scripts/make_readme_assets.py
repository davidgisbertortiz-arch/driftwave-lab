#!/usr/bin/env python
"""Generate all README visual assets headlessly.

Usage:
    python scripts/make_readme_assets.py
    python scripts/make_readme_assets.py --config configs/assets.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from driftwave_lab.utils.config import load_yaml  # noqa: E402
from driftwave_lab.viz.readme_assets import generate_all_assets  # noqa: E402

_DEFAULT_CONFIG = {
    "output_dir": "assets",
    "solver": {"nx": 64, "n_steps": 2000, "save_every": 50, "seed": 0},
    "rollout_steps": 30,
    "fps": 8,
    "fno_checkpoint": "checkpoints/fno2d_best.pt",
    "unet_checkpoint": "checkpoints/unet_best.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README visual assets")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to assets YAML config (default: built-in defaults)",
    )
    args = parser.parse_args()

    if args.config is not None:
        cfg = load_yaml(args.config)
    else:
        cfg = _DEFAULT_CONFIG

    assets = generate_all_assets(cfg)

    if not assets:
        print(
            "\n⚠  No assets were generated.  To produce ML comparison assets, first:\n"
            "   1. Generate a dataset:  python scripts/generate_dataset.py --config configs/dataset_tiny.yaml\n"
            "   2. Train a model:       python scripts/train.py --config configs/train_fno_tiny.yaml\n"
            "   3. Re-run this script:  python scripts/make_readme_assets.py\n"
        )
        sys.exit(0)

    print(f"\n✓  {len(assets)} assets saved.  See assets/ directory.")


if __name__ == "__main__":
    main()

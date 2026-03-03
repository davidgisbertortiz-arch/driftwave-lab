#!/usr/bin/env python
"""Train an ML surrogate model.

Usage:
    python scripts/train.py --config configs/train_fno.yaml
    python scripts/train.py --config configs/train_unet.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from driftwave_lab.training.train_fno import train  # noqa: E402
from driftwave_lab.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ML surrogate model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_fno.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda")
    parser.add_argument("--quiet", action="store_true", help="Suppress epoch logs")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    result = train(cfg, device=args.device, verbose=not args.quiet)

    # Save history as JSON next to checkpoint
    history_path = Path(result["checkpoint_path"]).parent / "history.json"
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)

    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"History:    {history_path}")


if __name__ == "__main__":
    main()

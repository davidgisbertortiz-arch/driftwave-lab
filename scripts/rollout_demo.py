#!/usr/bin/env python
"""Run an autoregressive rollout demo with a trained surrogate.

Usage:
    python scripts/rollout_demo.py --checkpoint checkpoints/fno2d_best.pt \
        --data data/raw/manifest.json --steps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from driftwave_lab.data.dataset import HWNextStepDataset  # noqa: E402
from driftwave_lab.evaluation.rollout import evaluate_rollout  # noqa: E402
from driftwave_lab.models import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoregressive rollout demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to manifest.json (uses test split)")
    parser.add_argument("--steps", type=int, default=20, help="Number of rollout steps")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default=None, help="If set, save results to this NPZ path")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    # Build ground-truth trajectory from test split
    data_path = args.data
    if data_path is None:
        data_path = str(Path(cfg["data"]["dataset_dir"]) / "manifest.json")

    ds = HWNextStepDataset(data_path, split="test", preload=True)

    # Find first test trajectory and build truth sequence of length (steps+1)
    n_steps = min(args.steps, len(ds) - 1)
    truth = []
    inp0, _ = ds[0]
    truth.append(inp0)
    for i in range(n_steps):
        _, tgt = ds[i]
        truth.append(tgt)

    truth_tensor = torch.stack(truth)  # (T, C, H, W)

    # Evaluate rollout
    result = evaluate_rollout(model, truth_tensor, device=device)

    print(f"Rollout {n_steps} steps on {device}")
    print(f"  Final MSE:    {result['mse'][-1]:.4e}")
    print(f"  Final relL2:  {result['rel_l2'][-1]:.4f}")
    print(f"  Mean MSE:     {np.mean(result['mse']):.4e}")
    print(f"  Mean relL2:   {np.mean(result['rel_l2']):.4f}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        preds = torch.stack(result["preds"]).numpy()
        truth_np = truth_tensor.numpy()
        np.savez(
            str(out),
            preds=preds,
            truth=truth_np,
            mse=np.array(result["mse"]),
            rel_l2=np.array(result["rel_l2"]),
        )
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()

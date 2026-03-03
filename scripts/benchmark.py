#!/usr/bin/env python
"""Run runtime benchmarks (solver vs surrogates).

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/fno2d_best.pt
    python scripts/benchmark.py --checkpoint checkpoints/fno2d_best.pt \
        --unet-checkpoint checkpoints/unet_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from driftwave_lab.evaluation.benchmark import benchmark_model, system_info  # noqa: E402
from driftwave_lab.models import build_model  # noqa: E402


def _load_model(ckpt_path: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Primary model checkpoint")
    parser.add_argument("--unet-checkpoint", type=str, default=None, help="U-Net checkpoint")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution (H=W)")
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default=None, help="Save JSON results to path")
    args = parser.parse_args()

    device = torch.device(args.device)
    sample = torch.randn(1, 2, args.resolution, args.resolution)

    results: list[dict] = []

    for ckpt_path in filter(None, [args.checkpoint, args.unet_checkpoint]):
        model, _cfg = _load_model(ckpt_path)
        res = benchmark_model(
            model,
            sample,
            rollout_steps=args.rollout_steps,
            device=device,
        )
        results.append(res.to_dict())
        print(
            f"{res.name:12s}  1-step: {res.one_step_ms:8.2f} ms  "
            f"rollout({res.rollout_steps}): {res.rollout_ms:8.2f} ms  "
            f"params: {res.n_params:,}"
        )

    info = system_info()
    output = {"system": info, "results": results}

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {out}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

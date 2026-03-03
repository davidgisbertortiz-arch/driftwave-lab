#!/usr/bin/env python
"""Generate a parameterised multi-trajectory dataset from the HW solver.

Usage::

    python scripts/generate_dataset.py                              # default config
    python scripts/generate_dataset.py --config configs/dataset.yaml
    python scripts/generate_dataset.py --config configs/dataset_tiny.yaml
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate HW dataset (multiple trajectories with varied parameters)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset YAML config (default: configs/dataset.yaml)",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    from driftwave_lab.data.generator import DatasetConfig, generate_dataset
    from driftwave_lab.utils.config import load_yaml

    raw = load_yaml(args.config)
    cfg = DatasetConfig.from_dict(raw)

    n_total = cfg.n_train + cfg.n_val + cfg.n_test + cfg.n_ood
    print(
        f"▶ Generating dataset  "
        f"nx={cfg.nx}  n_steps={cfg.n_steps}  dt={cfg.dt}\n"
        f"  trajectories: {n_total}  "
        f"(train={cfg.n_train}  val={cfg.n_val}  test={cfg.n_test}  ood={cfg.n_ood})\n"
        f"  alpha ∈ {list(cfg.alpha_range)}  "
        f"ood_alpha ∈ {list(cfg.ood_alpha_range)}\n"
        f"  output → {cfg.output_dir}"
    )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    result = generate_dataset(cfg)

    print(
        f"\n✓ Dataset complete: {result.n_trajectories} trajectories "
        f"in {result.total_elapsed_s:.1f} s\n"
        f"  manifest → {result.manifest_path}"
    )

    # ------------------------------------------------------------------
    # Summary by split
    # ------------------------------------------------------------------
    from collections import Counter

    split_counts = Counter(e["split"] for e in result.manifest)
    for split in ("train", "val", "test", "ood"):
        print(f"  {split:>5s}: {split_counts.get(split, 0)} trajectories")


if __name__ == "__main__":
    main()

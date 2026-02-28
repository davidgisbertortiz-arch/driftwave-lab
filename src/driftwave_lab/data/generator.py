"""Parameterised dataset generation from the Hasegawa–Wakatani solver.

This module provides utilities to:

1. **Sample physics parameters** from configurable uniform ranges using a
   deterministic master seed.
2. **Run a batch of solver trajectories** with varied parameters and seeds.
3. **Split the resulting trajectories** into train / val / test / OOD sets
   at the *trajectory* level (no frame leakage).
4. **Write a manifest** (``manifest.json``) that records every trajectory's
   parameters, file path and split assignment for full reproducibility.

See ``PROJECT_SPEC.md`` §6 for the scientific rationale.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from driftwave_lab.data.io import save_trajectory
from driftwave_lab.solver.hw import HWParams, HWTrajectory, solve
from driftwave_lab.solver.initial_conditions import random_perturbation
from driftwave_lab.solver.spectral import SpectralGrid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Full specification for a dataset generation run.

    Created from a YAML config dict via :func:`DatasetConfig.from_dict`.
    """

    # Grid
    nx: int = 64
    ny: int = 64
    lx: float = 40.0
    ly: float = 40.0

    # Time integration
    dt: float = 0.025
    n_steps: int = 2000
    save_every: int = 50

    # Initial-condition amplitude
    ic_amplitude: float = 1e-4

    # Number of in-distribution + OOD trajectories
    n_train: int = 30
    n_val: int = 8
    n_test: int = 8
    n_ood: int = 4

    # Parameter ranges – in-distribution
    alpha_range: tuple[float, float] = (0.5, 1.5)
    kappa_range: tuple[float, float] = (0.8, 1.2)
    D_range: tuple[float, float] = (0.005, 0.02)
    nu_range: tuple[float, float] = (0.005, 0.02)

    # OOD parameter override (only alpha shifted; others same as ID)
    ood_alpha_range: tuple[float, float] = (1.5, 2.5)

    # Master seed (controls parameter sampling + per-traj IC seeds)
    seed: int = 123

    # Output
    output_dir: str = "data/raw"

    # Misc
    verbose: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetConfig":
        """Build from a flat / nested YAML-style dict."""
        grid = d.get("grid", {})
        time_cfg = d.get("time", {})
        ic_cfg = d.get("initial_condition", {})
        split = d.get("split", {})
        ranges = d.get("parameter_ranges", {})
        ood = d.get("ood", {})

        def _range(key: str, default: tuple[float, float]) -> tuple[float, float]:
            v = ranges.get(key, default)
            return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else default

        # Support both absolute counts and fractional splits
        n_total = d.get("n_trajectories", 50)
        if all(isinstance(split.get(k), float) for k in ("train", "val", "test")):
            n_train = int(round(n_total * split["train"]))
            n_val = int(round(n_total * split["val"]))
            n_test = n_total - n_train - n_val
        else:
            n_train = split.get("train", 30)
            n_val = split.get("val", 8)
            n_test = split.get("test", 8)

        n_ood = d.get("n_ood", split.get("ood", 4))

        ood_alpha = ood.get("alpha", [1.5, 2.5])

        return cls(
            nx=grid.get("nx", cls.nx),
            ny=grid.get("ny", cls.ny),
            lx=grid.get("lx", cls.lx),
            ly=grid.get("ly", cls.ly),
            dt=time_cfg.get("dt", cls.dt),
            n_steps=time_cfg.get("n_steps", cls.n_steps),
            save_every=time_cfg.get("save_every", cls.save_every),
            ic_amplitude=ic_cfg.get("amplitude", cls.ic_amplitude),
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_ood=int(n_ood),
            alpha_range=_range("alpha", cls.alpha_range),
            kappa_range=_range("kappa", cls.kappa_range),
            D_range=_range("D", cls.D_range),
            nu_range=_range("nu", cls.nu_range),
            ood_alpha_range=(float(ood_alpha[0]), float(ood_alpha[1])),
            seed=d.get("seed", cls.seed),
            output_dir=d.get("output_dir", cls.output_dir),
            verbose=d.get("verbose", cls.verbose),
        )


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

@dataclass
class TrajectorySpec:
    """Specification for a single trajectory to be generated."""

    index: int
    split: str  # "train", "val", "test", or "ood"
    alpha: float
    kappa: float
    D: float
    nu: float
    ic_seed: int  # per-trajectory IC seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "D": self.D,
            "nu": self.nu,
            "ic_seed": self.ic_seed,
        }


def sample_trajectory_specs(cfg: DatasetConfig) -> list[TrajectorySpec]:
    """Deterministically sample physics parameters for every trajectory.

    The master RNG is seeded once with ``cfg.seed`` and used to:
    1. Draw parameter values (uniform) for each in-distribution trajectory.
    2. Assign trajectories to train / val / test splits via a shuffle.
    3. Draw OOD trajectory parameters (alpha from ``ood_alpha_range``).
    4. Assign unique IC seeds to every trajectory.

    Returns a list ordered by global trajectory index.
    """
    rng = np.random.default_rng(cfg.seed)

    n_id = cfg.n_train + cfg.n_val + cfg.n_test  # in-distribution total

    # --- In-distribution parameters ---
    alphas = rng.uniform(*cfg.alpha_range, size=n_id)
    kappas = rng.uniform(*cfg.kappa_range, size=n_id)
    Ds = rng.uniform(*cfg.D_range, size=n_id)
    nus = rng.uniform(*cfg.nu_range, size=n_id)

    # Shuffle indices for split assignment
    indices = np.arange(n_id)
    rng.shuffle(indices)
    split_labels = (
        ["train"] * cfg.n_train
        + ["val"] * cfg.n_val
        + ["test"] * cfg.n_test
    )

    id_specs: list[TrajectorySpec] = []
    for rank, idx in enumerate(indices):
        id_specs.append(
            TrajectorySpec(
                index=int(idx),
                split=split_labels[rank],
                alpha=float(alphas[idx]),
                kappa=float(kappas[idx]),
                D=float(Ds[idx]),
                nu=float(nus[idx]),
                ic_seed=int(rng.integers(0, 2**31)),
            )
        )

    # --- OOD parameters (shifted alpha range) ---
    ood_alphas = rng.uniform(*cfg.ood_alpha_range, size=cfg.n_ood)
    ood_kappas = rng.uniform(*cfg.kappa_range, size=cfg.n_ood)
    ood_Ds = rng.uniform(*cfg.D_range, size=cfg.n_ood)
    ood_nus = rng.uniform(*cfg.nu_range, size=cfg.n_ood)

    ood_specs: list[TrajectorySpec] = []
    for i in range(cfg.n_ood):
        ood_specs.append(
            TrajectorySpec(
                index=n_id + i,
                split="ood",
                alpha=float(ood_alphas[i]),
                kappa=float(ood_kappas[i]),
                D=float(ood_Ds[i]),
                nu=float(ood_nus[i]),
                ic_seed=int(rng.integers(0, 2**31)),
            )
        )

    # Sort by global index for deterministic file naming
    all_specs = sorted(id_specs + ood_specs, key=lambda s: s.index)
    return all_specs


# ---------------------------------------------------------------------------
# Single-trajectory runner
# ---------------------------------------------------------------------------

def run_single_trajectory(
    spec: TrajectorySpec,
    cfg: DatasetConfig,
) -> tuple[dict[str, NDArray], dict[str, Any], float]:
    """Run one HW simulation and return (arrays, metadata, elapsed_s).

    Does NOT write files — the caller handles I/O and manifest bookkeeping.
    """
    grid = SpectralGrid(nx=cfg.nx, ny=cfg.ny, lx=cfg.lx, ly=cfg.ly)
    params = HWParams(
        alpha=spec.alpha,
        kappa=spec.kappa,
        D=spec.D,
        nu=spec.nu,
        dt=cfg.dt,
        n_steps=cfg.n_steps,
        save_every=cfg.save_every,
    )
    n0, omega0 = random_perturbation(grid, amplitude=cfg.ic_amplitude, seed=spec.ic_seed)

    t0 = time.perf_counter()
    traj: HWTrajectory = solve(grid, params, n0, omega0, verbose=cfg.verbose)
    elapsed = time.perf_counter() - t0

    arrays = traj.to_arrays()
    metadata = {
        **spec.to_dict(),
        "nx": cfg.nx,
        "ny": cfg.ny,
        "lx": cfg.lx,
        "ly": cfg.ly,
        "dt": cfg.dt,
        "n_steps": cfg.n_steps,
        "save_every": cfg.save_every,
        "ic_amplitude": cfg.ic_amplitude,
        "elapsed_s": round(elapsed, 3),
    }
    return arrays, metadata, elapsed


# ---------------------------------------------------------------------------
# Full dataset generation driver
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Summary of a completed dataset generation run."""

    output_dir: Path
    manifest_path: Path
    n_trajectories: int
    total_elapsed_s: float
    manifest: list[dict[str, Any]] = field(default_factory=list)


def generate_dataset(cfg: DatasetConfig) -> GenerationResult:
    """Generate a full multi-trajectory dataset.

    1. Sample parameters using :func:`sample_trajectory_specs`.
    2. Run each trajectory with the solver.
    3. Save NPZ files with embedded metadata.
    4. Write a ``manifest.json`` recording parameters, splits and paths.

    Parameters
    ----------
    cfg : DatasetConfig
        Full generation configuration.

    Returns
    -------
    GenerationResult
    """
    specs = sample_trajectory_specs(cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()

    for i, spec in enumerate(specs):
        fname = f"traj_{spec.index:04d}.npz"
        fpath = out_dir / fname

        logger.info(
            "Trajectory %d/%d  (split=%s, alpha=%.3f, kappa=%.3f)  → %s",
            i + 1, len(specs), spec.split, spec.alpha, spec.kappa, fname,
        )
        if cfg.verbose:
            print(
                f"  [{i + 1}/{len(specs)}] split={spec.split}  "
                f"alpha={spec.alpha:.3f}  kappa={spec.kappa:.3f}  → {fname}"
            )

        arrays, meta, elapsed = run_single_trajectory(spec, cfg)

        save_trajectory(
            fpath,
            times=arrays["times"],
            n=arrays["n"],
            omega=arrays["omega"],
            phi=arrays["phi"],
            metadata=meta,
        )

        manifest_entry = {
            **meta,
            "file": fname,
        }
        manifest.append(manifest_entry)

    total_elapsed = time.perf_counter() - total_t0

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "config": {
                    "seed": cfg.seed,
                    "nx": cfg.nx,
                    "ny": cfg.ny,
                    "n_steps": cfg.n_steps,
                    "dt": cfg.dt,
                    "save_every": cfg.save_every,
                    "alpha_range": list(cfg.alpha_range),
                    "kappa_range": list(cfg.kappa_range),
                    "D_range": list(cfg.D_range),
                    "nu_range": list(cfg.nu_range),
                    "ood_alpha_range": list(cfg.ood_alpha_range),
                },
                "splits": {
                    "train": cfg.n_train,
                    "val": cfg.n_val,
                    "test": cfg.n_test,
                    "ood": cfg.n_ood,
                },
                "total_elapsed_s": round(total_elapsed, 2),
                "trajectories": manifest,
            },
            f,
            indent=2,
        )

    logger.info(
        "Dataset complete: %d trajectories in %.1f s  → %s",
        len(specs), total_elapsed, out_dir,
    )

    return GenerationResult(
        output_dir=out_dir,
        manifest_path=manifest_path,
        n_trajectories=len(specs),
        total_elapsed_s=total_elapsed,
        manifest=manifest,
    )

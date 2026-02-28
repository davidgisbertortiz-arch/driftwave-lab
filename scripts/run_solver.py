#!/usr/bin/env python
"""Run the Hasegawa–Wakatani solver to produce a single trajectory.

Usage:
    python scripts/run_solver.py                          # uses configs/solver.yaml
    python scripts/run_solver.py --config configs/solver_tiny.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run HW solver")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/solver.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    from driftwave_lab.utils.config import load_yaml

    cfg = load_yaml(args.config)
    grid_cfg = cfg["grid"]
    phys = cfg["physics"]
    time_cfg = cfg["time"]
    ic_cfg = cfg.get("initial_condition", {})
    out_cfg = cfg.get("output", {"dir": "outputs", "prefix": "hw_run"})
    seed = cfg.get("seed", 42)
    verbose = cfg.get("verbose", False)

    # ------------------------------------------------------------------
    # Build grid + params
    # ------------------------------------------------------------------
    from driftwave_lab.solver.spectral import SpectralGrid

    grid = SpectralGrid(
        nx=grid_cfg["nx"],
        ny=grid_cfg["ny"],
        lx=grid_cfg["lx"],
        ly=grid_cfg["ly"],
    )

    from driftwave_lab.solver.hw import HWParams

    params = HWParams(
        alpha=phys["alpha"],
        kappa=phys["kappa"],
        D=phys["D"],
        nu=phys["nu"],
        dt=time_cfg["dt"],
        n_steps=time_cfg["n_steps"],
        save_every=time_cfg["save_every"],
    )

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    from driftwave_lab.solver.initial_conditions import random_perturbation

    amplitude = ic_cfg.get("amplitude", 1e-4)
    n0, omega0 = random_perturbation(grid, amplitude=amplitude, seed=seed)

    # ------------------------------------------------------------------
    # Run solver
    # ------------------------------------------------------------------
    from driftwave_lab.solver.hw import solve

    print(f"▶ Running HW solver  nx={grid.nx}  n_steps={params.n_steps}  dt={params.dt}")
    t0 = time.perf_counter()
    traj = solve(grid, params, n0, omega0, verbose=verbose)
    elapsed = time.perf_counter() - t0
    print(f"✓ Solver finished in {elapsed:.2f} s  ({len(traj.times)} snapshots)")

    # ------------------------------------------------------------------
    # Save trajectory
    # ------------------------------------------------------------------
    from driftwave_lab.data.io import save_trajectory

    arrays = traj.to_arrays()
    metadata = {
        "alpha": phys["alpha"],
        "kappa": phys["kappa"],
        "D": phys["D"],
        "nu": phys["nu"],
        "dt": time_cfg["dt"],
        "n_steps": time_cfg["n_steps"],
        "save_every": time_cfg["save_every"],
        "nx": grid_cfg["nx"],
        "ny": grid_cfg["ny"],
        "lx": grid_cfg["lx"],
        "ly": grid_cfg["ly"],
        "seed": seed,
        "elapsed_s": elapsed,
    }
    out_dir = Path(out_cfg["dir"])
    out_path = out_dir / f"{out_cfg['prefix']}.npz"
    save_trajectory(
        out_path,
        times=arrays["times"],
        n=arrays["n"],
        omega=arrays["omega"],
        phi=arrays["phi"],
        metadata=metadata,
    )
    print(f"✓ Trajectory saved → {out_path}")

    # ------------------------------------------------------------------
    # Quick diagnostic snapshot plot
    # ------------------------------------------------------------------
    _save_snapshot_plot(arrays, grid, out_dir, out_cfg["prefix"])

    # ------------------------------------------------------------------
    # Diagnostics summary
    # ------------------------------------------------------------------
    from driftwave_lab.solver.diagnostics import trajectory_diagnostics

    diag = trajectory_diagnostics(
        arrays["n"], arrays["omega"], arrays["phi"], grid
    )
    print(
        f"  energy  : {diag['energy'][0]:.3e} → {diag['energy'][-1]:.3e}\n"
        f"  enstrophy: {diag['enstrophy'][0]:.3e} → {diag['enstrophy'][-1]:.3e}\n"
        f"  n_rms   : {diag['n_rms'][0]:.3e} → {diag['n_rms'][-1]:.3e}"
    )


# ------------------------------------------------------------------
# Simple snapshot figure
# ------------------------------------------------------------------

def _save_snapshot_plot(
    arrays: dict[str, np.ndarray],
    grid: Any,
    out_dir: Path,
    prefix: str,
) -> None:
    """Save a 3-panel PNG of the final snapshot (n, ω, φ)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fields = {"n": arrays["n"][-1], "ω": arrays["omega"][-1], "φ": arrays["phi"][-1]}
    for ax, (label, fld) in zip(axes, fields.items()):
        vmax = max(abs(fld.min()), abs(fld.max())) or 1.0
        im = ax.imshow(
            fld.T,
            origin="lower",
            extent=[0, grid.lx, 0, grid.ly],
            cmap="RdBu_s" if "RdBu_s" in plt.colormaps() else "RdBu",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    t_final = arrays["times"][-1]
    fig.suptitle(f"HW snapshot  t = {t_final:.2f}", fontsize=13)
    fig.tight_layout()

    fig_path = out_dir / f"{prefix}_snapshot.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"✓ Snapshot plot  → {fig_path}")


if __name__ == "__main__":
    main()

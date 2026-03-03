"""Automated generation of all README visual assets.

Orchestrates solver runs, model loading, and visualisation to produce
``hero.gif``, ``error.gif``, ``spectra.png``, ``benchmark.png``, and
optional supplementary figures.

This module is designed to run headlessly via::

    python scripts/make_readme_assets.py [--config configs/assets.yaml]

It can operate in two modes:

1. **With checkpoints** — loads trained FNO/UNet models and compares
   them against solver ground truth.
2. **Solver-only** — generates solver visualisations even when no
   ML checkpoints are available (hero.gif shows two different solver
   realisations; spectra shows solver only).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from driftwave_lab.solver.diagnostics import isotropic_spectrum
from driftwave_lab.solver.hw import HWParams, solve
from driftwave_lab.solver.initial_conditions import random_perturbation
from driftwave_lab.solver.spectral import SpectralGrid
from driftwave_lab.viz.gifs import make_error_gif, make_hero_gif
from driftwave_lab.viz.plots import (
    plot_benchmark,
    plot_comparison_panel,
    plot_rollout_error,
    plot_spectra,
)

# Force non-interactive backend for headless generation
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Solver-based data generation
# ---------------------------------------------------------------------------


def _run_solver(
    nx: int = 64,
    n_steps: int = 2000,
    save_every: int = 50,
    seed: int = 0,
    **hw_kwargs: Any,
) -> dict[str, NDArray]:
    """Run a quick HW solver trajectory and return field arrays."""
    grid = SpectralGrid(nx, nx, 2 * np.pi, 2 * np.pi)
    params = HWParams(
        n_steps=n_steps,
        save_every=save_every,
        **{k: v for k, v in hw_kwargs.items() if k in HWParams.__dataclass_fields__},
    )
    n0, omega0 = random_perturbation(grid, seed=seed)
    traj = solve(grid, params, n0, omega0, verbose=False)
    arrays = traj.to_arrays()
    return {"n": arrays["n"], "phi": arrays["phi"], "omega": arrays["omega"], "grid": grid}


def _load_model_rollout(
    checkpoint_path: str | Path,
    truth_n: NDArray,
    truth_phi: NDArray,
    n_steps: int,
) -> NDArray:
    """Load a trained model and produce an autoregressive rollout.

    Returns predicted density fields of shape (n_steps+1, H, W).
    """
    import torch

    from driftwave_lab.models import build_model

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Initial state: stack [n, phi] as channels
    x = torch.from_numpy(np.stack([truth_n[0], truth_phi[0]], axis=0)[None].astype(np.float32))

    preds_n = [truth_n[0]]  # initial = ground truth

    with torch.no_grad():
        for _ in range(n_steps):
            y = model(x)
            preds_n.append(y[0, 0].numpy())  # channel 0 = n
            x = y

    return np.stack(preds_n)


# ---------------------------------------------------------------------------
# Top-level asset generators
# ---------------------------------------------------------------------------


def generate_hero_gif(
    out_dir: Path,
    *,
    truth_n: NDArray,
    pred_n: NDArray | None = None,
    truth_n_alt: NDArray | None = None,
    fps: int = 8,
) -> Path:
    """Generate ``hero.gif`` — truth vs prediction (or two solver runs)."""
    if pred_n is not None:
        right = [pred_n[i] for i in range(len(pred_n))]
        right_label = "FNO rollout"
    elif truth_n_alt is not None:
        right = [truth_n_alt[i] for i in range(len(truth_n_alt))]
        right_label = "Solver (seed 2)"
    else:
        raise ValueError("Need either pred_n or truth_n_alt for hero.gif")

    n_frames = min(len(truth_n), len(right))
    truth_list = [truth_n[i] for i in range(n_frames)]
    right = right[:n_frames]

    return make_hero_gif(truth_list, right, out_dir / "hero.gif", field_name="n", fps=fps)


def generate_error_gif(
    out_dir: Path,
    *,
    truth_n: NDArray,
    pred_n: NDArray,
    fps: int = 8,
) -> Path:
    """Generate ``error.gif`` — animated error field."""
    n_frames = min(len(truth_n), len(pred_n))
    truth_list = [truth_n[i] for i in range(n_frames)]
    pred_list = [pred_n[i] for i in range(n_frames)]
    return make_error_gif(truth_list, pred_list, out_dir / "error.gif", field_name="n", fps=fps)


def generate_spectra_png(
    out_dir: Path,
    *,
    fields_dict: dict[str, NDArray],
    grid: SpectralGrid,
) -> Path:
    """Generate ``spectra.png`` — log-log spectral comparison."""
    spectra = {}
    for label, field in fields_dict.items():
        k, s = isotropic_spectrum(field, grid)
        spectra[label] = (k, s)

    fig = plot_spectra(spectra, title="Isotropic energy spectrum comparison")
    path = out_dir / "spectra.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def generate_benchmark_png(
    out_dir: Path,
    *,
    names: list[str],
    one_step_ms: list[float],
    n_params: list[int] | None = None,
) -> Path:
    """Generate ``benchmark.png`` — inference latency bar chart."""
    fig = plot_benchmark(names, one_step_ms, n_params=n_params)
    path = out_dir / "benchmark.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def generate_rollout_error_png(
    out_dir: Path,
    *,
    mse_values: NDArray | list[float],
    rel_l2_values: NDArray | list[float] | None = None,
) -> Path:
    """Generate ``rollout_error.png`` — error vs horizon."""
    fig = plot_rollout_error(mse_values, rel_l2_values, title="FNO rollout error vs horizon")
    path = out_dir / "rollout_error.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def generate_comparison_png(
    out_dir: Path,
    *,
    truth: NDArray,
    pred: NDArray,
    step: int = 0,
) -> Path:
    """Generate ``rollout_comparison.png`` — snapshot comparison panel."""
    fig = plot_comparison_panel(truth, pred, field_name="n", step=step)
    path = out_dir / "rollout_comparison.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def generate_all_assets(cfg: dict[str, Any]) -> dict[str, Path]:
    """Generate all README assets according to *cfg*.

    Parameters
    ----------
    cfg : dict
        Parsed assets YAML config.

    Returns
    -------
    dict mapping asset name to file path.
    """
    out_dir = Path(cfg.get("output_dir", "assets"))
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_cfg = cfg.get("solver", {})
    nx = solver_cfg.get("nx", 64)
    n_steps = solver_cfg.get("n_steps", 2000)
    save_every = solver_cfg.get("save_every", 50)
    seed = solver_cfg.get("seed", 0)
    fps = cfg.get("fps", 8)

    print(f"[assets] Running solver ({nx}×{nx}, {n_steps} steps, seed={seed})...")
    t0 = time.perf_counter()
    data = _run_solver(nx=nx, n_steps=n_steps, save_every=save_every, seed=seed)
    truth_n = data["n"]
    truth_phi = data["phi"]
    grid = data["grid"]
    print(f"[assets] Solver done in {time.perf_counter() - t0:.1f}s  ({len(truth_n)} snapshots)")

    # Try loading FNO checkpoint
    fno_ckpt = cfg.get("fno_checkpoint")
    unet_ckpt = cfg.get("unet_checkpoint")

    has_fno = fno_ckpt is not None and Path(fno_ckpt).exists()
    has_unet = unet_ckpt is not None and Path(unet_ckpt).exists()

    assets: dict[str, Path] = {}

    # Determine rollout length for GIFs
    n_rollout = min(cfg.get("rollout_steps", 30), len(truth_n) - 1)

    if has_fno:
        print(f"[assets] Loading FNO from {fno_ckpt}...")
        fno_pred = _load_model_rollout(fno_ckpt, truth_n, truth_phi, n_rollout)

        # hero.gif: solver vs FNO
        print("[assets] Generating hero.gif...")
        assets["hero.gif"] = generate_hero_gif(
            out_dir, truth_n=truth_n[: n_rollout + 1], pred_n=fno_pred, fps=fps
        )

        # error.gif
        print("[assets] Generating error.gif...")
        assets["error.gif"] = generate_error_gif(
            out_dir, truth_n=truth_n[: n_rollout + 1], pred_n=fno_pred, fps=fps
        )

        # rollout error metrics
        from driftwave_lab.evaluation.metrics import mse as compute_mse
        from driftwave_lab.evaluation.metrics import relative_l2 as compute_rel_l2

        import torch

        mse_vals = []
        rel_vals = []
        for i in range(1, len(fno_pred)):
            p = torch.from_numpy(fno_pred[i][None])
            t = torch.from_numpy(truth_n[i][None])
            mse_vals.append(compute_mse(p, t).item())
            rel_vals.append(compute_rel_l2(p, t).item())

        print("[assets] Generating rollout_error.png...")
        assets["rollout_error.png"] = generate_rollout_error_png(
            out_dir, mse_values=mse_vals, rel_l2_values=rel_vals
        )

        # Snapshot comparison at mid-rollout
        mid = n_rollout // 2
        print("[assets] Generating rollout_comparison.png...")
        assets["rollout_comparison.png"] = generate_comparison_png(
            out_dir, truth=truth_n[mid], pred=fno_pred[mid], step=mid
        )

    else:
        print("[assets] No FNO checkpoint found — generating solver-only hero.gif...")
        data2 = _run_solver(nx=nx, n_steps=n_steps, save_every=save_every, seed=seed + 42)
        assets["hero.gif"] = generate_hero_gif(
            out_dir,
            truth_n=truth_n[: n_rollout + 1],
            truth_n_alt=data2["n"][: n_rollout + 1],
            fps=fps,
        )

    # spectra.png — always possible
    print("[assets] Generating spectra.png...")
    spec_fields: dict[str, NDArray] = {"Solver (n)": truth_n[-1]}

    if has_fno:
        spec_fields["FNO rollout (n)"] = fno_pred[-1]
    if has_unet:
        print(f"[assets] Loading UNet from {unet_ckpt} for spectrum...")
        unet_pred = _load_model_rollout(unet_ckpt, truth_n, truth_phi, n_rollout)
        spec_fields["U-Net rollout (n)"] = unet_pred[-1]

    assets["spectra.png"] = generate_spectra_png(out_dir, fields_dict=spec_fields, grid=grid)

    # benchmark.png — only with checkpoints
    if has_fno or has_unet:
        print("[assets] Generating benchmark.png...")
        import torch

        from driftwave_lab.evaluation.benchmark import benchmark_model
        from driftwave_lab.models import build_model

        bench_names: list[str] = []
        bench_times: list[float] = []
        bench_params: list[int] = []
        sample = torch.randn(1, 2, nx, nx)

        for label, ckpt_path in [("FNO", fno_ckpt), ("U-Net", unet_ckpt)]:
            if ckpt_path is None or not Path(ckpt_path).exists():
                continue
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model = build_model(ckpt["config"]["model"])
            model.load_state_dict(ckpt["model_state_dict"])
            res = benchmark_model(model, sample, rollout_steps=10, warmup=2, repeats=3)
            bench_names.append(label)
            bench_times.append(res.one_step_ms)
            bench_params.append(res.n_params or 0)

        if bench_names:
            assets["benchmark.png"] = generate_benchmark_png(
                out_dir, names=bench_names, one_step_ms=bench_times, n_params=bench_params
            )
    else:
        print("[assets] No checkpoints found — skipping benchmark.png")

    print(f"\n[assets] Generated {len(assets)} assets in {out_dir}/:")
    for name, path in sorted(assets.items()):
        size_kb = path.stat().st_size / 1024
        print(f"  {name:30s} {size_kb:6.1f} KB")

    return assets

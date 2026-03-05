"""Publication-style static plots for HW fields and ML comparisons.

All functions accept matplotlib axes (or create their own) and return
figure objects so callers can save or display them.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_CMAP_FIELD = "RdBu_r"
_CMAP_ERROR = "inferno"
_CMAP_SPECTRUM = "tab10"


def _apply_style() -> None:
    """Apply a clean publication style."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _symmetric_norm(field: NDArray) -> TwoSlopeNorm:
    """Create a symmetric diverging norm centred at zero."""
    vmax = float(np.max(np.abs(field)))
    vmax = max(vmax, 1e-10)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


# ---------------------------------------------------------------------------
# Field snapshot panels
# ---------------------------------------------------------------------------


def plot_field(
    field: NDArray,
    *,
    title: str = "",
    ax: Any | None = None,
    cmap: str = _CMAP_FIELD,
    symmetric: bool = True,
) -> plt.Figure:
    """Plot a single 2D field with optional symmetric colour scale."""
    _apply_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    else:
        fig = ax.figure

    norm = _symmetric_norm(field) if symmetric else None
    im = ax.imshow(field.T, origin="lower", cmap=cmap, norm=norm, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    return fig


def plot_comparison_panel(
    truth: NDArray,
    pred: NDArray,
    *,
    field_name: str = "n",
    step: int | None = None,
) -> plt.Figure:
    """Three-panel figure: truth | prediction | error."""
    _apply_style()
    error = pred - truth
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    label = f" (t={step})" if step is not None else ""
    norm = _symmetric_norm(truth)

    for ax, data, title in zip(
        axes[:2],
        [truth, pred],
        [f"Truth {field_name}{label}", f"Predicted {field_name}{label}"],
        strict=True,
    ):
        im = ax.imshow(data.T, origin="lower", cmap=_CMAP_FIELD, norm=norm, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    # Error panel
    err_vmax = float(np.max(np.abs(error)))
    err_norm = TwoSlopeNorm(vmin=-err_vmax, vcenter=0, vmax=max(err_vmax, 1e-10))
    im_err = axes[2].imshow(
        error.T, origin="lower", cmap=_CMAP_FIELD, norm=err_norm, aspect="equal"
    )
    axes[2].set_title(f"Error {field_name}{label}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im_err, ax=axes[2], shrink=0.8, pad=0.02)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectra comparison
# ---------------------------------------------------------------------------


def plot_spectra(
    spectra: dict[str, tuple[NDArray, NDArray]],
    *,
    title: str = "Isotropic energy spectrum",
    xlabel: str = "k",
    ylabel: str = r"$E(k)$",
) -> plt.Figure:
    """Log-log plot of multiple isotropic spectra.

    Parameters
    ----------
    spectra : dict[str, (k_shells, spectrum)]
        Label -> (k, E(k)) pairs.
    """
    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    colors = matplotlib.colormaps.get_cmap(_CMAP_SPECTRUM)
    for i, (label, (k, spec)) in enumerate(spectra.items()):
        mask = (k > 0) & (spec > 0)
        ax.loglog(k[mask], spec[mask], label=label, color=colors(i), linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark bar chart
# ---------------------------------------------------------------------------


def plot_benchmark(
    names: list[str],
    one_step_ms: list[float],
    *,
    n_params: list[int] | None = None,
    title: str = "Inference latency (one step)",
) -> plt.Figure:
    """Horizontal bar chart of one-step inference times with optional speedup."""
    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, one_step_ms, color=["#4C72B0", "#55A868", "#C44E52"][: len(names)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, one_step_ms, strict=True):
        ax.text(
            bar.get_width() + max(one_step_ms) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} ms",
            va="center",
            fontsize=9,
        )

    if n_params is not None:
        for bar, n in zip(bars, n_params, strict=True):
            ax.text(
                bar.get_width() * 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{n / 1000:.0f}K params",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    # Speedup annotations if solver is slowest
    if len(one_step_ms) >= 2:
        slowest = max(one_step_ms)
        for bar, val in zip(bars, one_step_ms, strict=True):
            if val < slowest:
                speedup = slowest / max(val, 1e-6)
                ax.text(
                    bar.get_width() + max(one_step_ms) * 0.15,
                    bar.get_y() + bar.get_height() / 2,
                    f"({speedup:.0f}x faster)",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rollout error vs horizon
# ---------------------------------------------------------------------------


def plot_rollout_error(
    mse_values: list[float] | NDArray,
    rel_l2_values: list[float] | NDArray | None = None,
    *,
    title: str = "Rollout error vs horizon",
) -> plt.Figure:
    """Line plot of MSE (and optionally relative L2) over rollout steps."""
    _apply_style()
    steps = np.arange(1, len(mse_values) + 1)

    if rel_l2_values is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.semilogy(steps, mse_values, "o-", markersize=3, linewidth=1.5, color="#4C72B0")
        ax1.set_xlabel("Rollout step")
        ax1.set_ylabel("MSE")
        ax1.set_title("MSE vs horizon")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, rel_l2_values, "s-", markersize=3, linewidth=1.5, color="#C44E52")
        ax2.set_xlabel("Rollout step")
        ax2.set_ylabel("Relative L²")
        ax2.set_title("Relative L² vs horizon")
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))
        ax1.semilogy(steps, mse_values, "o-", markersize=3, linewidth=1.5, color="#4C72B0")
        ax1.set_xlabel("Rollout step")
        ax1.set_ylabel("MSE")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig

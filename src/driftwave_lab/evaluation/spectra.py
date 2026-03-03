"""Spectral diagnostics and comparison utilities.

Provides helpers to compute and compare isotropic energy spectra from
ground-truth solver fields and ML surrogate predictions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from driftwave_lab.solver.diagnostics import isotropic_spectrum
from driftwave_lab.solver.spectral import SpectralGrid


def spectrum_from_field(field: NDArray, grid: SpectralGrid) -> tuple[NDArray, NDArray]:
    """Thin wrapper around :func:`isotropic_spectrum` for convenience."""
    return isotropic_spectrum(field, grid)


def compare_spectra(
    fields: dict[str, NDArray],
    grid: SpectralGrid,
) -> dict[str, tuple[NDArray, NDArray]]:
    """Compute isotropic spectra for multiple fields.

    Parameters
    ----------
    fields : dict[str, NDArray]
        Mapping of label -> 2D real-space field (nx, ny).
    grid : SpectralGrid

    Returns
    -------
    dict[str, tuple[NDArray, NDArray]]
        Mapping of label -> (k_shells, spectrum).
    """
    return {label: isotropic_spectrum(f, grid) for label, f in fields.items()}


def spectral_mismatch(
    field_pred: NDArray,
    field_true: NDArray,
    grid: SpectralGrid,
) -> float:
    """Relative L2 spectral mismatch between predicted and true fields.

    Returns  ||S_pred - S_true||_2 / ||S_true||_2.
    """
    _, s_pred = isotropic_spectrum(field_pred, grid)
    _, s_true = isotropic_spectrum(field_true, grid)
    # Align lengths (should be the same, but be defensive)
    n = min(len(s_pred), len(s_true))
    diff = np.linalg.norm(s_pred[:n] - s_true[:n])
    true_norm = np.linalg.norm(s_true[:n])
    if true_norm < 1e-15:
        return float(diff)
    return float(diff / true_norm)

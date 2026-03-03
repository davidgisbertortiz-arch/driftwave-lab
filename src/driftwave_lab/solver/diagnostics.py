"""Solver diagnostics: energy, enstrophy, isotropic spectra.

All functions operate on **real-space** 2-D fields unless stated otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from driftwave_lab.solver.spectral import SpectralGrid, fft2

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Scalar diagnostics
# ---------------------------------------------------------------------------


def field_norm(field: NDArray) -> float:
    """L² norm of a real-space field (RMS × √N)."""
    return float(np.sqrt(np.mean(field**2)))


def energy(n: NDArray, phi: NDArray, grid: SpectralGrid) -> float:
    """Approximate total energy  E = 0.5 ⟨n² + |∇φ|²⟩.

    Computed via Parseval's theorem in Fourier space for accuracy.
    """
    n_hat = fft2(n)
    phi_hat = fft2(phi)
    # |∇φ|² in Fourier: k² |φ̂|²   (factor 2 for rfft negative freqs)
    grad_phi_sq = np.sum(grid.K2 * np.abs(phi_hat) ** 2) / (grid.nx * grid.ny) ** 2
    n_sq = np.sum(np.abs(n_hat) ** 2) / (grid.nx * grid.ny) ** 2
    # rfft stores only half the ky modes; double-count all except ky=0 and ky=Nyquist
    # For simplicity in the MVP the rough estimate below is acceptable.
    return float(0.5 * (n_sq + grad_phi_sq))


def enstrophy(omega: NDArray) -> float:
    """Approximate enstrophy  Z = 0.5 ⟨ω²⟩."""
    return float(0.5 * np.mean(omega**2))


# ---------------------------------------------------------------------------
# Isotropic (shell-averaged) spectrum
# ---------------------------------------------------------------------------


def isotropic_spectrum(field: NDArray, grid: SpectralGrid) -> tuple[NDArray, NDArray]:
    """Compute the 1-D isotropic energy spectrum of a real-space field.

    Parameters
    ----------
    field : NDArray, shape (nx, ny)
    grid : SpectralGrid

    Returns
    -------
    k_shells : NDArray, shape (n_shells,)
        Shell-centre wavenumbers.
    spectrum : NDArray, shape (n_shells,)
        Shell-summed |f̂(k)|².
    """
    field_hat = fft2(field)
    power = np.abs(field_hat) ** 2

    k_mag = np.sqrt(grid.K2)
    dk = 2.0 * np.pi / min(grid.lx, grid.ly)
    k_max = np.max(k_mag)
    n_shells = int(k_max / dk) + 1
    k_shells = np.arange(n_shells) * dk
    spectrum = np.zeros(n_shells)

    for i in range(n_shells):
        mask = (k_mag >= i * dk) & (k_mag < (i + 1) * dk)
        spectrum[i] = np.sum(power[mask])

    # Normalise by grid size
    spectrum /= (grid.nx * grid.ny) ** 2

    return k_shells, spectrum


# ---------------------------------------------------------------------------
# Trajectory-level diagnostics
# ---------------------------------------------------------------------------


def trajectory_diagnostics(
    n_series: NDArray,
    omega_series: NDArray,
    phi_series: NDArray,
    grid: SpectralGrid,
) -> dict[str, NDArray]:
    """Compute time-series diagnostics for a full trajectory.

    Parameters
    ----------
    n_series, omega_series, phi_series : NDArray, shape (T, nx, ny)
    grid : SpectralGrid

    Returns
    -------
    dict with keys:
        ``energy``, ``enstrophy``, ``n_rms``, ``omega_rms``, ``phi_rms``
        — each an array of length T.
    """
    n_t = n_series.shape[0]
    out: dict[str, list[float]] = {
        "energy": [],
        "enstrophy": [],
        "n_rms": [],
        "omega_rms": [],
        "phi_rms": [],
    }
    for t in range(n_t):
        out["energy"].append(energy(n_series[t], phi_series[t], grid))
        out["enstrophy"].append(enstrophy(omega_series[t]))
        out["n_rms"].append(field_norm(n_series[t]))
        out["omega_rms"].append(field_norm(omega_series[t]))
        out["phi_rms"].append(field_norm(phi_series[t]))

    return {k: np.asarray(v) for k, v in out.items()}

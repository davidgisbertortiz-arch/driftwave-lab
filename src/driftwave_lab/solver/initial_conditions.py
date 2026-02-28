"""Initial-condition generators for HW fields.

All generators return *real-space* 2-D arrays of shape ``(nx, ny)``
representing density and vorticity perturbations on the periodic grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from driftwave_lab.solver.spectral import SpectralGrid


def random_perturbation(
    grid: SpectralGrid,
    *,
    amplitude: float = 1e-4,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Small-amplitude broadband random perturbation for *n* and *ω*.

    The perturbation is generated in Fourier space with random phases and
    an isotropic amplitude envelope that decays as 1 / (1 + k²), then
    de-aliased and transformed to physical space.

    Parameters
    ----------
    grid : SpectralGrid
    amplitude : float
        Characteristic amplitude scaling.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    n0, omega0 : NDArray
        Real-space density and vorticity perturbation fields.
    """
    rng = np.random.default_rng(seed)

    def _make_field() -> NDArray:
        phase = rng.uniform(0, 2 * np.pi, size=grid.KX.shape)
        envelope = amplitude / (1.0 + grid.K2)
        field_hat = envelope * np.exp(1j * phase) * grid.dealias_mask
        # Enforce zero mean
        field_hat[0, 0] = 0.0
        return np.fft.irfft2(field_hat, s=(grid.nx, grid.ny))

    n0 = _make_field()
    omega0 = _make_field()
    return n0, omega0

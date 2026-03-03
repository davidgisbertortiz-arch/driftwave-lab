"""Fourier pseudo-spectral operators on a 2-D doubly periodic domain.

Provides wavenumber grids, spectral derivatives, Laplacian inversion, and
the Arakawa-like Poisson bracket used in the Hasegawa–Wakatani system.

All operators work on **real-valued** fields using :func:`numpy.fft.rfft2` /
:func:`numpy.fft.irfft2` for efficiency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Spectral grid
# ---------------------------------------------------------------------------


class SpectralGrid:
    """Pre-computed wavenumber arrays and de-aliasing mask for a 2-D periodic box.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in *x* and *y*.
    lx, ly : float
        Physical domain lengths.
    """

    __slots__ = (
        "K2",
        "KX",
        "KY",
        "dealias_mask",
        "dx",
        "dy",
        "inv_K2",
        "kx",
        "ky",
        "lx",
        "ly",
        "nx",
        "ny",
        "x",
        "y",
    )

    def __init__(self, nx: int, ny: int, lx: float, ly: float) -> None:
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly

        self.dx: float = lx / nx
        self.dy: float = ly / ny

        self.x: NDArray = np.linspace(0, lx, nx, endpoint=False)
        self.y: NDArray = np.linspace(0, ly, ny, endpoint=False)

        # Wavenumber vectors (rfft2 convention: full kx, half+1 ky)
        self.kx: NDArray = 2.0 * np.pi / lx * np.fft.fftfreq(nx, d=1.0 / nx)
        self.ky: NDArray = 2.0 * np.pi / ly * np.fft.rfftfreq(ny, d=1.0 / ny)
        self.KX: NDArray
        self.KY: NDArray
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")

        K2 = self.KX**2 + self.KY**2
        self.K2: NDArray = K2

        # Inverse Laplacian (zero mode → 0, no mean)
        inv_K2 = np.zeros_like(K2)
        inv_K2[K2 > 0] = 1.0 / K2[K2 > 0]
        self.inv_K2: NDArray = inv_K2

        # 2/3-rule de-aliasing mask
        kx_max = 2.0 * np.pi / lx * (nx // 2)
        ky_max = 2.0 * np.pi / ly * (ny // 2)
        mask = (np.abs(self.KX) < (2.0 / 3.0) * kx_max) & (np.abs(self.KY) < (2.0 / 3.0) * ky_max)
        self.dealias_mask: NDArray = mask.astype(np.float64)

    def __repr__(self) -> str:
        return f"SpectralGrid(nx={self.nx}, ny={self.ny}, lx={self.lx}, ly={self.ly})"


# ---------------------------------------------------------------------------
# Spectral derivative helpers
# ---------------------------------------------------------------------------


def fft2(field: NDArray) -> NDArray:
    """Forward real FFT (wrapper for consistency)."""
    return np.fft.rfft2(field)


def ifft2(field_hat: NDArray) -> NDArray:
    """Inverse real FFT (wrapper for consistency)."""
    return np.fft.irfft2(field_hat)


def deriv_x(field_hat: NDArray, grid: SpectralGrid) -> NDArray:
    """Spectral ∂/∂x  →  returns real-space result."""
    return ifft2(1j * grid.KX * field_hat)


def deriv_y(field_hat: NDArray, grid: SpectralGrid) -> NDArray:
    """Spectral ∂/∂y  →  returns real-space result."""
    return ifft2(1j * grid.KY * field_hat)


def laplacian_hat(field_hat: NDArray, grid: SpectralGrid) -> NDArray:
    """Spectral Laplacian (returns Fourier-space result)."""
    return -grid.K2 * field_hat


def solve_poisson(omega_hat: NDArray, grid: SpectralGrid) -> NDArray:
    """Solve  ∇²φ = ω  for φ̂ in Fourier space.

    Returns the Fourier-space potential ``phi_hat``.
    """
    return -grid.inv_K2 * omega_hat


# ---------------------------------------------------------------------------
# Poisson bracket  {a, b} = ∂_x a ∂_y b  −  ∂_y a ∂_x b
# ---------------------------------------------------------------------------


def poisson_bracket(a_hat: NDArray, b_hat: NDArray, grid: SpectralGrid) -> NDArray:
    """Compute the Poisson bracket {a, b} and return de-aliased Fourier result.

    Uses the simple "multiply in physical space, de-alias" approach
    (adequate for MVP; Arakawa form can be added later).
    """
    ax = deriv_x(a_hat, grid)
    ay = deriv_y(a_hat, grid)
    bx = deriv_x(b_hat, grid)
    by = deriv_y(b_hat, grid)

    bracket = ax * by - ay * bx
    return fft2(bracket) * grid.dealias_mask

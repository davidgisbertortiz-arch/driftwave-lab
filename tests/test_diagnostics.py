"""Tests for diagnostics utilities."""

from __future__ import annotations

import numpy as np
import pytest

from driftwave_lab.solver.diagnostics import (
    energy,
    enstrophy,
    field_norm,
    isotropic_spectrum,
    trajectory_diagnostics,
)
from driftwave_lab.solver.spectral import SpectralGrid


@pytest.fixture()
def grid() -> SpectralGrid:
    return SpectralGrid(nx=32, ny=32, lx=20.0, ly=20.0)


class TestScalarDiagnostics:
    def test_field_norm_zero(self, grid: SpectralGrid) -> None:
        z = np.zeros((grid.nx, grid.ny))
        assert field_norm(z) == 0.0

    def test_field_norm_positive(self, grid: SpectralGrid) -> None:
        f = np.ones((grid.nx, grid.ny))
        assert field_norm(f) > 0.0

    def test_energy_nonneg(self, grid: SpectralGrid) -> None:
        rng = np.random.default_rng(0)
        n = rng.normal(size=(grid.nx, grid.ny)) * 0.01
        phi = rng.normal(size=(grid.nx, grid.ny)) * 0.01
        assert energy(n, phi, grid) >= 0.0

    def test_enstrophy_nonneg(self) -> None:
        omega = np.random.default_rng(0).normal(size=(32, 32))
        assert enstrophy(omega) >= 0.0


class TestSpectrum:
    def test_spectrum_shape(self, grid: SpectralGrid) -> None:
        f = np.random.default_rng(0).normal(size=(grid.nx, grid.ny)) * 0.01
        k_shells, spec = isotropic_spectrum(f, grid)
        assert k_shells.shape == spec.shape
        assert len(k_shells) > 0

    def test_spectrum_nonneg(self, grid: SpectralGrid) -> None:
        f = np.random.default_rng(0).normal(size=(grid.nx, grid.ny)) * 0.01
        _, spec = isotropic_spectrum(f, grid)
        assert np.all(spec >= 0.0)


class TestTrajectoryDiagnostics:
    def test_shapes(self, grid: SpectralGrid) -> None:
        rng = np.random.default_rng(1)
        T = 5
        n = rng.normal(size=(T, grid.nx, grid.ny)) * 0.01
        omega = rng.normal(size=(T, grid.nx, grid.ny)) * 0.01
        phi = rng.normal(size=(T, grid.nx, grid.ny)) * 0.01
        diag = trajectory_diagnostics(n, omega, phi, grid)
        for key in ("energy", "enstrophy", "n_rms", "omega_rms", "phi_rms"):
            assert diag[key].shape == (T,)

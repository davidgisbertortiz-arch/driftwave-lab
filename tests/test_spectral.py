"""Tests for the spectral grid and derivative operators."""

from __future__ import annotations

import numpy as np
import pytest

from driftwave_lab.solver.spectral import (
    SpectralGrid,
    deriv_x,
    deriv_y,
    fft2,
    ifft2,
    laplacian_hat,
    poisson_bracket,
    solve_poisson,
)


@pytest.fixture()
def grid() -> SpectralGrid:
    return SpectralGrid(nx=64, ny=64, lx=2 * np.pi, ly=2 * np.pi)


class TestSpectralGrid:
    def test_grid_shapes(self, grid: SpectralGrid) -> None:
        assert grid.x.shape == (64,)
        assert grid.y.shape == (64,)
        # rfft2 shape: (nx, ny//2 + 1)
        assert grid.KX.shape == (64, 33)
        assert grid.KY.shape == (64, 33)
        assert grid.K2.shape == (64, 33)
        assert grid.dealias_mask.shape == (64, 33)

    def test_inv_k2_zero_mode(self, grid: SpectralGrid) -> None:
        """inv_K2 should be zero at the (0,0) mode."""
        assert grid.inv_K2[0, 0] == 0.0


class TestDerivatives:
    """Check spectral derivatives against known analytic functions."""

    def test_deriv_x_sin(self, grid: SpectralGrid) -> None:
        """∂/∂x sin(x) = cos(x)"""
        X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
        f = np.sin(X)
        expected = np.cos(X)
        result = deriv_x(fft2(f), grid)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_deriv_y_sin(self, grid: SpectralGrid) -> None:
        """∂/∂y sin(y) = cos(y)"""
        X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
        f = np.sin(Y)
        expected = np.cos(Y)
        result = deriv_y(fft2(f), grid)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_deriv_x_cos2(self, grid: SpectralGrid) -> None:
        """∂/∂x cos(2x) = -2 sin(2x)"""
        X, _ = np.meshgrid(grid.x, grid.y, indexing="ij")
        f = np.cos(2 * X)
        expected = -2 * np.sin(2 * X)
        result = deriv_x(fft2(f), grid)
        np.testing.assert_allclose(result, expected, atol=1e-11)


class TestLaplacian:
    def test_laplacian_sinsin(self, grid: SpectralGrid) -> None:
        """∇² sin(x)sin(y) = -2 sin(x)sin(y)"""
        X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
        f = np.sin(X) * np.sin(Y)
        expected = -2.0 * f
        result = ifft2(laplacian_hat(fft2(f), grid))
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestPoissonSolver:
    def test_poisson_roundtrip(self, grid: SpectralGrid) -> None:
        """Solve ∇²φ = ω then verify ∇²φ ≈ ω (for non-zero-mean ω)."""
        X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
        omega = np.sin(X) * np.cos(Y)
        omega_hat = fft2(omega)
        phi_hat = solve_poisson(omega_hat, grid)
        omega_back = ifft2(laplacian_hat(phi_hat, grid))
        # Laplacian of solved phi should recover omega (up to zero-mode)
        omega_zm = omega - np.mean(omega)
        np.testing.assert_allclose(omega_back, omega_zm, atol=1e-12)


class TestPoissonBracket:
    def test_bracket_antisymmetry(self, grid: SpectralGrid) -> None:
        """{a, b} = -{b, a} up to de-aliasing precision."""
        X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
        a = np.sin(X) * np.cos(Y)
        b = np.cos(X) * np.sin(2 * Y)
        a_hat, b_hat = fft2(a), fft2(b)
        ab = ifft2(poisson_bracket(a_hat, b_hat, grid))
        ba = ifft2(poisson_bracket(b_hat, a_hat, grid))
        np.testing.assert_allclose(ab, -ba, atol=1e-10)

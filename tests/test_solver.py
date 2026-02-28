"""Smoke and correctness tests for the HW solver."""

from __future__ import annotations

import numpy as np
import pytest

from driftwave_lab.solver.hw import HWParams, solve
from driftwave_lab.solver.initial_conditions import random_perturbation
from driftwave_lab.solver.spectral import SpectralGrid


@pytest.fixture()
def small_grid() -> SpectralGrid:
    return SpectralGrid(nx=32, ny=32, lx=20.0, ly=20.0)


@pytest.fixture()
def default_params() -> HWParams:
    return HWParams(
        alpha=1.0,
        kappa=1.0,
        D=0.01,
        nu=0.01,
        dt=0.05,
        n_steps=100,
        save_every=50,
    )


class TestSolverSmoke:
    """Basic sanity checks — no NaNs, correct shapes, stable output."""

    def test_solver_runs(self, small_grid: SpectralGrid, default_params: HWParams) -> None:
        n0, omega0 = random_perturbation(small_grid, amplitude=1e-4, seed=0)
        traj = solve(small_grid, default_params, n0, omega0)

        # Should have (initial + n_steps/save_every) snapshots
        expected_snaps = 1 + default_params.n_steps // default_params.save_every
        assert len(traj.times) == expected_snaps

    def test_no_nans(self, small_grid: SpectralGrid, default_params: HWParams) -> None:
        n0, omega0 = random_perturbation(small_grid, amplitude=1e-4, seed=0)
        traj = solve(small_grid, default_params, n0, omega0)
        arrays = traj.to_arrays()
        for key in ("n", "omega", "phi"):
            assert not np.any(np.isnan(arrays[key])), f"NaN in {key}"

    def test_output_shapes(self, small_grid: SpectralGrid, default_params: HWParams) -> None:
        n0, omega0 = random_perturbation(small_grid, amplitude=1e-4, seed=0)
        traj = solve(small_grid, default_params, n0, omega0)
        arrays = traj.to_arrays()
        n_snaps = len(traj.times)
        for key in ("n", "omega", "phi"):
            assert arrays[key].shape == (n_snaps, small_grid.nx, small_grid.ny)

    def test_zero_ic_stays_zero(self, small_grid: SpectralGrid) -> None:
        """Starting from zero fields, everything should stay (near) zero."""
        params = HWParams(dt=0.05, n_steps=20, save_every=20)
        n0 = np.zeros((small_grid.nx, small_grid.ny))
        omega0 = np.zeros((small_grid.nx, small_grid.ny))
        traj = solve(small_grid, params, n0, omega0)
        arrays = traj.to_arrays()
        assert np.max(np.abs(arrays["n"][-1])) < 1e-14

    def test_reproducibility(self, small_grid: SpectralGrid, default_params: HWParams) -> None:
        """Same IC → same result."""
        n0a, o0a = random_perturbation(small_grid, amplitude=1e-4, seed=7)
        n0b, o0b = random_perturbation(small_grid, amplitude=1e-4, seed=7)
        t1 = solve(small_grid, default_params, n0a, o0a)
        t2 = solve(small_grid, default_params, n0b, o0b)
        np.testing.assert_array_equal(t1.to_arrays()["n"], t2.to_arrays()["n"])

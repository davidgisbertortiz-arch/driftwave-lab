"""Hasegawa–Wakatani 2D pseudo-spectral solver.

Implements the two-field HW system on a doubly periodic domain using
Fourier pseudo-spectral spatial derivatives and an explicit RK2 (Heun)
time integrator.

**Equations solved** (see ``docs/hw_equations.md`` for full details)::

    ∂_t n   + {φ, n}   = α (φ − n) − κ ∂_y φ + D ∇²n
    ∂_t ω   + {φ, ω}   = α (φ − n)           + ν ∇²ω
    ω = ∇²φ

where {a, b} = ∂_x a ∂_y b − ∂_y a ∂_x b is the Poisson bracket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from driftwave_lab.solver.spectral import (
    SpectralGrid,
    fft2,
    ifft2,
    laplacian_hat,
    poisson_bracket,
    solve_poisson,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class HWParams:
    """Physical and numerical parameters for a Hasegawa–Wakatani run."""

    alpha: float = 1.0     # adiabaticity / parallel-electron coupling
    kappa: float = 1.0     # background density-gradient drive
    D: float = 0.01        # density diffusion
    nu: float = 0.01       # viscosity
    dt: float = 0.01       # time step
    n_steps: int = 10_000  # total integration steps
    save_every: int = 100  # snapshot cadence (in steps)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class HWState:
    """Instantaneous solver state in Fourier space."""

    n_hat: NDArray      # density fluctuation (Fourier)
    omega_hat: NDArray  # vorticity (Fourier)
    step: int = 0
    time: float = 0.0

    def fields(self, grid: SpectralGrid) -> dict[str, NDArray]:
        """Return real-space n, ω, φ."""
        phi_hat = solve_poisson(self.omega_hat, grid)
        return {
            "n": ifft2(self.n_hat),
            "omega": ifft2(self.omega_hat),
            "phi": ifft2(phi_hat),
        }


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------

def _rhs(
    n_hat: NDArray,
    omega_hat: NDArray,
    grid: SpectralGrid,
    params: HWParams,
) -> tuple[NDArray, NDArray]:
    """Evaluate ∂_t n̂ and ∂_t ω̂ in Fourier space."""
    phi_hat = solve_poisson(omega_hat, grid)

    # Poisson brackets (nonlinear terms, de-aliased)
    pb_phi_n = poisson_bracket(phi_hat, n_hat, grid)
    pb_phi_omega = poisson_bracket(phi_hat, omega_hat, grid)

    # Coupling term  α (φ̂ − n̂)
    coupling = params.alpha * (phi_hat - n_hat)

    # Density gradient drive  −κ ∂_y φ  (in Fourier: −κ i k_y φ̂)
    drive = -params.kappa * 1j * grid.KY * phi_hat

    # Diffusion / viscosity
    diff_n = params.D * laplacian_hat(n_hat, grid)
    diff_omega = params.nu * laplacian_hat(omega_hat, grid)

    dn_hat = -pb_phi_n + coupling + drive + diff_n
    domega_hat = -pb_phi_omega + coupling + diff_omega

    return dn_hat, domega_hat


# ---------------------------------------------------------------------------
# Time integrator — RK2 (Heun / improved Euler)
# ---------------------------------------------------------------------------

def _step_rk2(
    state: HWState,
    grid: SpectralGrid,
    params: HWParams,
) -> HWState:
    """Advance one RK2 (Heun) step in-place-ish, returning a new state."""
    dt = params.dt
    n0, o0 = state.n_hat, state.omega_hat

    # Stage 1
    k1n, k1o = _rhs(n0, o0, grid, params)
    n1 = n0 + dt * k1n
    o1 = o0 + dt * k1o

    # Stage 2
    k2n, k2o = _rhs(n1, o1, grid, params)

    n_new = n0 + 0.5 * dt * (k1n + k2n)
    o_new = o0 + 0.5 * dt * (k1o + k2o)

    # De-alias after each full step
    n_new *= grid.dealias_mask
    o_new *= grid.dealias_mask

    return HWState(
        n_hat=n_new,
        omega_hat=o_new,
        step=state.step + 1,
        time=state.time + dt,
    )


# ---------------------------------------------------------------------------
# Trajectory container
# ---------------------------------------------------------------------------

@dataclass
class HWTrajectory:
    """Collected snapshots from a solver run."""

    times: list[float] = field(default_factory=list)
    n_snapshots: list[NDArray] = field(default_factory=list)
    omega_snapshots: list[NDArray] = field(default_factory=list)
    phi_snapshots: list[NDArray] = field(default_factory=list)

    def to_arrays(self) -> dict[str, NDArray]:
        """Pack into a dict of numpy arrays suitable for NPZ storage."""
        return {
            "times": np.asarray(self.times),
            "n": np.stack(self.n_snapshots),
            "omega": np.stack(self.omega_snapshots),
            "phi": np.stack(self.phi_snapshots),
        }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def solve(
    grid: SpectralGrid,
    params: HWParams,
    n0: NDArray,
    omega0: NDArray,
    *,
    verbose: bool = False,
) -> HWTrajectory:
    """Run a full Hasegawa–Wakatani simulation.

    Parameters
    ----------
    grid : SpectralGrid
        Pre-computed spectral grid.
    params : HWParams
        Physical and numerical parameters.
    n0, omega0 : NDArray
        Real-space initial density and vorticity fields of shape ``(nx, ny)``.
    verbose : bool
        Print progress every ``save_every`` steps.

    Returns
    -------
    HWTrajectory
        Collected time-series of n, ω, φ snapshots.
    """
    state = HWState(
        n_hat=fft2(n0) * grid.dealias_mask,
        omega_hat=fft2(omega0) * grid.dealias_mask,
        step=0,
        time=0.0,
    )

    traj = HWTrajectory()

    # Save initial state
    _record_snapshot(traj, state, grid)

    for _ in range(params.n_steps):
        state = _step_rk2(state, grid, params)

        if state.step % params.save_every == 0:
            _record_snapshot(traj, state, grid)
            if verbose:
                n_real = ifft2(state.n_hat)
                print(
                    f"  step {state.step:>6d}  t={state.time:.3f}"
                    f"  |n|_max={np.max(np.abs(n_real)):.4e}"
                )

    return traj


def _record_snapshot(
    traj: HWTrajectory, state: HWState, grid: SpectralGrid
) -> None:
    fields = state.fields(grid)
    traj.times.append(state.time)
    traj.n_snapshots.append(fields["n"].copy())
    traj.omega_snapshots.append(fields["omega"].copy())
    traj.phi_snapshots.append(fields["phi"].copy())

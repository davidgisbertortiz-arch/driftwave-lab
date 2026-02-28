"""Pseudo-spectral Hasegawa–Wakatani solver.

Public API:
    SpectralGrid  – wavenumber grid and de-aliasing mask
    HWParams      – physical / numerical parameters
    HWState       – instantaneous Fourier-space state
    HWTrajectory  – collected snapshot time-series
    solve         – main integration driver
"""

from driftwave_lab.solver.hw import HWParams, HWState, HWTrajectory, solve
from driftwave_lab.solver.spectral import SpectralGrid

__all__ = [
    "HWParams",
    "HWState",
    "HWTrajectory",
    "SpectralGrid",
    "solve",
]

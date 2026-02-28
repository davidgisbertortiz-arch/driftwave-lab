# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **HW solver MVP** (Phase 1):
  - Pseudo-spectral 2D Hasegawa–Wakatani solver with RK2 time integration.
  - Spectral grid, FFT-based derivatives, Poisson solver, de-aliased Poisson bracket.
  - Random-perturbation initial-condition generator.
  - Diagnostics: energy, enstrophy, field norms, isotropic spectrum.
  - NPZ I/O with embedded JSON metadata.
  - `scripts/run_solver.py` CLI with YAML config support.
  - Default (`solver.yaml`) and tiny (`solver_tiny.yaml`) configurations.
  - Snapshot plot output (3-panel PNG: n, ω, φ).
  - `docs/hw_equations.md` — full documentation of implemented equations.
  - Solver unit tests (`test_spectral.py`, `test_solver.py`, `test_diagnostics.py`).
  - Dependencies: added `pyyaml>=6`, `matplotlib>=3.7`.

- Repository scaffold (Phase 0):
  - Package structure, CI, dev tooling, placeholder modules.
  - `pyproject.toml` with ruff and pytest configuration.
  - GitHub Actions CI workflow (lint + format + test).
  - Placeholder scripts for dataset generation, training, and evaluation.
  - Configuration templates for dataset, U-Net, FNO, and PINN.
  - Smoke tests for package import and placeholder scripts.
  - `CONTRIBUTING.md`, `CHANGELOG.md`, `.editorconfig`, `Makefile`.

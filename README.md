# DriftWave-Lab

**A research-grade open-source laboratory for 2D drift-wave plasma turbulence, ML surrogates, and inverse scientific machine learning.**

> Pseudo-spectral Hasegawa–Wakatani solver · Fourier Neural Operator & U-Net surrogates · PINN-based parameter inference

---

## What is this?

DriftWave-Lab is a self-contained computational laboratory that combines:

1. A **pseudo-spectral numerical solver** for the 2D Hasegawa–Wakatani (HW) equations on a doubly periodic domain.
2. **Machine-learning surrogate models** (CNN/U-Net baseline and Fourier Neural Operator) trained on solver-generated turbulence data.
3. An optional **physics-informed inverse track** for recovering PDE parameters from sparse or noisy observations.

The goal is to provide a compact but serious platform for studying drift-wave turbulence dynamics with modern scientific ML methods — physically motivated, numerically grounded, visually compelling, and structured like production-grade research software.

## Scientific motivation

The edge and scrape-off-layer regions of magnetized plasmas exhibit strongly nonlinear turbulence driven by drift-wave instabilities. The **Hasegawa–Wakatani equations** are a widely used reduced model for resistive drift-wave turbulence, capturing:

- drift-wave instability and saturation,
- nonlinear E×B advection,
- zonal-flow self-organisation,
- anomalous cross-field transport,
- rich spectral and statistical structure.

HW sits in an ideal complexity regime: much richer than toy PDEs (Burgers, heat equation), yet far cheaper and more self-contained than full gyrokinetics. The resulting turbulent fields — with coherent vortices, filamentary structures, and broadband spectra — make an excellent testbed for neural-operator surrogates and physics-informed inverse methods.

For the full scientific and technical specification, see [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Roadmap

| Phase | Objective | Status |
|-------|-----------|--------|
| **0** | Repository scaffold, CI, dev tooling | ✅ Done |
| **1** | Pseudo-spectral HW solver MVP | ✅ Done |
| **2** | Parameterised multi-trajectory dataset generation | 🔲 Planned |
| **3** | CNN / U-Net baseline surrogate | 🔲 Planned |
| **4** | FNO 2D hero model + benchmarks + visual assets | 🔲 Planned |
| **5** | PINN / inverse parameter-estimation track | 🔲 Planned |
| **6** | Polish, reproducibility, release | 🔲 Planned |

## Architecture

```text
src/driftwave_lab/
├── solver/            # Pseudo-spectral HW solver, diagnostics, ICs
├── data/              # Dataset generation, I/O, PyTorch wrappers
├── models/            # U-Net, FNO 2D, PINN inverse
├── training/          # Training loops (baseline, FNO, PINN)
├── evaluation/        # Rollout, metrics, benchmarks, spectral analysis
├── viz/               # Plots, GIFs, README asset generation
└── utils/             # Config loading, reproducibility helpers

scripts/               # CLI entry points (solver, dataset, train, …)
configs/               # YAML configuration templates
tests/                 # Smoke and unit tests
assets/                # README visual assets (GIFs, plots)
docs/                  # Technical documentation
```

## Planned visual assets

Once the solver and ML pipelines are operational, the repository will feature:

| Asset | Description |
|-------|-------------|
| `hero.gif` | Side-by-side animation: ground-truth solver vs FNO rollout |
| `error.gif` | Animated prediction-error field over rollout time |
| `spectra.png` | Energy-spectrum comparison (solver / U-Net / FNO) |
| `benchmark.png` | Runtime bar chart with speedup annotations |
| `pinn_inverse.png` | Recovered-parameter convergence (inverse track) |

## Quickstart

```bash
# Clone the repository
git clone https://github.com/davidgisbertortiz-arch/driftwave-lab.git
cd driftwave-lab

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Editable install with development dependencies
pip install -e ".[dev]"

# Verify everything works
make check          # or: ruff check . && ruff format --check . && pytest
```

### Run the HW solver

```bash
# Default run (128×128, 4000 steps) — takes ~1-2 min
python scripts/run_solver.py

# Quick smoke run (32×32, 100 steps)
python scripts/run_solver.py --config configs/solver_tiny.yaml
```

Outputs (NPZ trajectory + snapshot PNG) are written to `outputs/`.
See [`docs/hw_equations.md`](docs/hw_equations.md) for the implemented equations.

### Requirements

- Python ≥ 3.11
- Core: NumPy, PyYAML, Matplotlib

## Running checks

```bash
ruff check .              # lint
ruff format --check .     # format check
pytest                    # tests
make check                # all of the above
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and workflow guidelines.

## References

Key references underpinning this project:

1. Hasegawa & Wakatani (1983). *Plasma Edge Turbulence.* Phys. Rev. Lett. **50**, 682.
2. Li et al. (2021). *Fourier Neural Operator for Parametric PDEs.* ICLR.
3. Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. **378**, 686.
4. Gahr, Farcas & Jenko (2024). *SciML-based reduced-order models for plasma turbulence.* Phys. Plasmas **31**, 113904.

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) § 14 for the complete reference list.

## License

[MIT](LICENSE)

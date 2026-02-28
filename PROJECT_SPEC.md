# PROJECT_SPEC.md

# DriftWave-Lab

## Working title
**driftwave-lab**

## One-line pitch
A research-grade open-source repository for **2D drift-wave plasma turbulence** based on the **Hasegawa-Wakatani equations**, combining a **pseudo-spectral plasma solver**, **machine-learning surrogates** (CNN/U-Net and Fourier Neural Operator), and an optional **PINN/inverse-problem track** for parameter inference.

## Visual promise
The repository must be **eye-catching** from the README alone. Its main assets should be:
- Side-by-side animated rollouts of **ground-truth solver vs ML surrogate**.
- Error-map animations over time.
- Energy-spectrum figures.
- Runtime/speedup figures.
- Optional parameter-inference plots for the PINN track.

The project should feel like a hybrid of:
1. a **plasma physicist's toy-but-serious turbulence lab**,
2. an **ML engineer / applied scientist benchmark repo**, and
3. a **portfolio-quality scientific software project**.

---

# 1. Scientific motivation

## 1.1 Why this project exists
The edge and scrape-off-layer regions of magnetized plasmas exhibit strongly nonlinear turbulence, with drift-wave physics playing a central role in cross-field transport. Reduced models such as the **Hasegawa-Wakatani (HW)** equations are widely used as a conceptual and computational laboratory for studying:
- drift-wave instability,
- turbulence self-organization,
- zonal-flow emergence,
- anomalous transport,
- statistical and spectral structure of turbulent fields.

The HW system is therefore an ideal target for a repo that wants to look simultaneously:
- physically nontrivial,
- visually impressive,
- numerically meaningful,
- ML-friendly.

## 1.2 Why HW is a good scientific compromise
HW sits in a very attractive regime for an open-source showcase:
- **Much more interesting than Burgers' equation** or plain diffusion.
- **Much cheaper and more self-contained than gyrokinetics**.
- Rich enough to produce filamentary, vortex-dominated, turbulence-like fields.
- Widely recognized in plasma turbulence literature as a reduced model for resistive drift-wave turbulence and as a paradigm for edge-plasma dynamics.

## 1.3 Why ML / PINNs / neural operators belong here
Modern SciML methods are increasingly used to accelerate, emulate, and invert PDE-governed dynamics. Neural operators such as the **Fourier Neural Operator (FNO)** were explicitly designed to learn mappings between function spaces for PDE problems and have shown strong performance on turbulent dynamics. Physics-informed neural networks (PINNs) are useful for **inverse problems** and **parameter identification**, even if they are often harder than classical solvers for long chaotic rollouts.

So the high-level scientific thesis is:

> Use a trusted numerical solver for HW turbulence as the source of truth, then build ML surrogates and inverse tools around it in a way that is physically honest, visually compelling, and benchmarkable.

---

# 2. Core physics model

## 2.1 Recommended default PDE: Hasegawa-Wakatani (HW)
Use the **2D electrostatic Hasegawa-Wakatani system** on a doubly periodic domain.

### Canonical variables
A minimal two-field formulation uses:
- **n(x, y, t)**: density fluctuation,
- **phi(x, y, t)**: electrostatic potential,
- **omega = Laplacian(phi)**: vorticity.

### Canonical structure
The exact nondimensional form may vary slightly across references, but the implementation should remain close to a standard resistive-drift-wave HW system:

- One equation for density evolution.
- One equation for vorticity or potential evolution.
- Nonlinear advection through **E x B** Poisson brackets.
- Linear coupling controlled by an **adiabaticity / resistive coupling parameter**.
- Optional dissipative / hyperdiffusive regularization for stable numerics.

A typical form is:

- d_t n + {phi, n} = alpha (phi - n) - kappa d_y phi + D nabla^2 n
- d_t omega + {phi, omega} = alpha (phi - n) + nu nabla^2 omega
- omega = nabla^2 phi

where:
- **{a, b} = d_x a d_y b - d_y a d_x b** is the Poisson bracket,
- **alpha** controls parallel electron response / adiabaticity,
- **kappa** is the background density-gradient drive,
- **D, nu** are diffusion / viscosity-type parameters.

The final implementation should document **precisely which convention is used**, including signs and normalization.

## 2.2 Physical interpretation of key terms
- **Poisson bracket terms** represent nonlinear E x B advection.
- **alpha (phi - n)** represents finite parallel electron response / resistive coupling.
- **kappa d_y phi** is the drift-wave drive induced by background density gradient.
- **diffusion / viscosity** stabilizes unresolved small scales and regularizes the numerics.

## 2.3 Why this model is visually attractive
HW naturally generates:
- coherent vortices,
- filamentary density structures,
- scale interactions,
- zonal patterns,
- rich Fourier spectra.

That makes it ideal for:
- animated scalar fields,
- side-by-side ML rollouts,
- error maps,
- spectral comparisons,
- transport diagnostics.

---

# 3. Numerical solver requirements

## 3.1 Solver philosophy
The numerical solver is the **source of truth**. It must be:
- simple enough to understand,
- stable enough to generate datasets reliably,
- fast enough to run on a laptop,
- clean enough to become a benchmark baseline.

## 3.2 Recommended numerical approach
### Spatial discretization
Use a **pseudo-spectral Fourier method** on a periodic square domain.

Why:
- HW is naturally posed on periodic boxes in many reduced-model studies.
- Fourier differentiation is compact and accurate.
- It gives nice spectral diagnostics "for free".
- It strongly matches the visual / benchmark flavor of the repo.

### Time stepping
Recommended order of preference:
1. **ETDRK4** if implemented robustly.
2. **RK2 / Heun** for MVP simplicity.
3. Upgrade later if needed.

For the first version, it is acceptable to start with RK2 and move to ETDRK4 later once the pipeline is stable.

## 3.3 Minimum diagnostics from the solver
The solver should export, for every trajectory:
- n(x, y, t),
- phi(x, y, t),
- omega(x, y, t),
- time array,
- parameter metadata,
- field norms,
- energy-like diagnostic,
- enstrophy-like diagnostic,
- 1D isotropic spectrum or shell-averaged spectrum.

## 3.4 Solver outputs
Preferred raw storage format for MVP:
- **NPZ** for simplicity and portability.

Possible later upgrade:
- **HDF5** if datasets become large.

Each file should contain metadata such as:
- grid size,
- dt,
- total steps,
- alpha,
- kappa,
- D,
- nu,
- random seed,
- git commit hash,
- code version.

---

# 4. Machine-learning tracks

The project should have **two ML tracks**:

## Track A (mandatory): forward surrogate / emulator
This is the main README hero.

### A.1 Baseline model
Start with a simple but strong baseline:
- **CNN / U-Net next-step predictor**.

Input:
- state at time t, channels = [n, phi] or [n, omega].

Output:
- predicted state at time t + delta_t.

### A.2 Main model
The hero model should be:
- **FNO 2D** (Fourier Neural Operator),
- optionally conditioned on PDE parameters alpha, kappa, D, nu.

This model should support:
- one-step prediction,
- autoregressive rollout,
- parameter-conditioned inference.

### A.3 Why FNO is scientifically appropriate
The repo is about field evolution on periodic grids; FNO is therefore especially attractive because:
- it operates naturally with spectral structure,
- it is widely recognized for operator learning in PDEs,
- it gives the project a strong "ML engineer for scientific computing" flavor.

## Track B (optional but highly recommended): PINN / inverse problem
PINNs are more convincing here as **inverse solvers** than as the main forward simulator.

### B.1 Best optional use of PINNs in this project
Use a PINN or physics-informed inverse model to **infer PDE parameters** such as:
- alpha,
- kappa,
- D,
- nu,

from:
- sparse observations,
- partial fields,
- noisy low-frame-rate snapshots.

### B.2 Why this is better than “PINN solves the whole turbulence problem”
For chaotic/turbulent PDEs, PINNs can be hard to train for long-time forward integration. A much more convincing use is:
- take data from the trusted spectral solver,
- feed sparse or noisy observations,
- use the physics residual as part of an inverse-problem loss,
- show that the model recovers hidden parameters.

This gives the repo an extra “applied scientist / inverse modeling / scientific ML” smell without making the whole project depend on PINN fragility.

---

# 5. Recommended project scope (what the first public version should and should not do)

## 5.1 What v0.1 SHOULD do
- Run a 2D HW solver on periodic boxes.
- Generate training data from multiple trajectories and parameters.
- Train a baseline CNN/U-Net surrogate.
- Train an FNO surrogate.
- Evaluate rollout quality vs horizon.
- Produce publication-style plots.
- Produce README-ready GIFs headlessly.
- Provide benchmarks and reproducibility metadata.

## 5.2 What v0.1 SHOULD NOT try to do
- Full gyrokinetic fidelity.
- Real experimental data ingestion.
- Massive cluster-only datasets.
- Overcomplicated uncertainty quantification.
- PINN for fully chaotic long-horizon forward simulation.
- Fancy web apps before the numerical core is stable.

---

# 6. Datasets and experiment design

## 6.1 Data generation strategy
Use the in-house spectral solver to generate synthetic datasets by varying:
- random seed,
- initial perturbations,
- alpha,
- kappa,
- diffusion / viscosity parameters,
- grid resolution (optional later),
- sampling cadence.

## 6.2 Dataset splits
Recommended split strategy:
- **train / validation / test** by trajectory, not by frame only,
- plus one **OOD set** where parameter ranges differ from training.

Example:
- Train alpha in [0.5, 1.5]
- Test in-distribution alpha in [0.5, 1.5]
- OOD test alpha in [1.5, 2.5]

This is very valuable because it creates an "applied scientist" storyline around:
- interpolation vs extrapolation,
- stability vs accuracy,
- robustness to regime shift.

## 6.3 Suggested learning targets
Three acceptable formulations:
1. **Next-step prediction**: x_t -> x_{t+1}
2. **Delta prediction**: x_t -> x_{t+1} - x_t
3. **Block rollout**: x_t -> x_{t+m}

Recommended MVP:
- start with next-step prediction,
- evaluate autoregressive rollout.

---

# 7. Evaluation philosophy

The repo should be benchmarked like an **applied-scientist / scientific-ML** project, not just like a homework notebook.

## 7.1 Core metrics
### Field error metrics
- MSE / RMSE over fields,
- relative L2 error,
- rollout error vs horizon,
- error on derived fields (e.g. vorticity).

### Physics-aware metrics
- energy-like diagnostic mismatch,
- enstrophy-like diagnostic mismatch,
- spectral mismatch,
- correlation structure,
- optional topology metrics.

### Performance metrics
- training time,
- inference latency,
- rollout runtime,
- speedup vs solver.

## 7.2 Must-have evaluation plots
- Predicted vs true frames at selected times.
- Error map snapshots.
- Rollout error vs horizon.
- Spectrum comparison.
- Runtime comparison.

## 7.3 OOD / robustness tests
At least one of the following should be in v0.1:
- OOD parameter generalization,
- missing/noisy input frames,
- lower spatial resolution test,
- longer rollout than seen in training.

---

# 8. Visual assets required for the README

The README should open with strong visual evidence that the repo is not generic.

## 8.1 Required hero assets
### Asset 1: hero.gif
Side-by-side movie:
- left = ground truth solver,
- right = FNO rollout,
- shown for density or vorticity field,
- same color scale,
- 5–8 seconds,
- looped.

### Asset 2: error.gif
Animated error field over rollout time.

### Asset 3: spectra.png
Spectral comparison between:
- solver,
- baseline model,
- FNO.

### Asset 4: benchmark.png
Bar chart with:
- solver runtime,
- baseline surrogate runtime,
- FNO runtime,
- speedup annotation.

### Asset 5 (optional): pinn_inverse.png
Recovered parameter trace / convergence for the inverse problem.

## 8.2 Visual style rules
- Use the same colormap and dynamic-range logic across truth/prediction/error.
- Avoid cluttered dashboards in v0.1.
- The repository should look like a scientific software package, not a toy notebook dump.

---

# 9. Repo architecture

Recommended folder tree:

```text
README.md
LICENSE
CONTRIBUTING.md
CHANGELOG.md
pyproject.toml
PROJECT_SPEC.md

src/driftwave_lab/
  __init__.py
  solver/
    hw.py
    spectral.py
    diagnostics.py
    initial_conditions.py
  data/
    generator.py
    dataset.py
    io.py
  models/
    unet.py
    fno2d.py
    pinn_inverse.py
  training/
    train_baseline.py
    train_fno.py
    train_pinn.py
  evaluation/
    rollout.py
    metrics.py
    benchmark.py
    spectra.py
  viz/
    plots.py
    gifs.py
    readme_assets.py

scripts/
  run_solver.py
  generate_dataset.py
  train.py
  rollout_demo.py
  benchmark.py
  make_readme_assets.py

configs/
  solver.yaml
  dataset.yaml
  train_unet.yaml
  train_fno.yaml
  train_pinn.yaml

tests/
  test_solver_smoke.py
  test_dataset_shapes.py
  test_unet_forward.py
  test_fno_forward.py
  test_metrics.py

assets/
  hero.gif
  error.gif
  spectra.png
  benchmark.png
  pinn_inverse.png
```

---

# 10. Detailed technical roadmap

## Phase 1 - numerical core
Objective:
- build a robust pseudo-spectral HW solver,
- generate stable short trajectories,
- produce first turbulence visuals.

Deliverables:
- `run_solver.py`
- `test_solver_smoke.py`
- first PNG and GIF assets.

## Phase 2 - data layer
Objective:
- generate a parameterized multi-trajectory dataset,
- define clean train/val/test/OOD splits,
- document file format.

Deliverables:
- `generate_dataset.py`
- `dataset.py`
- data schema docs.

## Phase 3 - baseline ML
Objective:
- train a CNN/U-Net baseline,
- establish first rollout benchmark.

Deliverables:
- training script,
- checkpoint format,
- rollout plots.

## Phase 4 - hero model
Objective:
- train FNO 2D,
- compare quality and speed vs baseline and solver.

Deliverables:
- hero.gif,
- benchmark plot,
- spectra comparison.

## Phase 5 - inverse / PINN extension
Objective:
- infer PDE parameters from partial/noisy observations.

Deliverables:
- inverse-problem notebook / script,
- recovered-parameter plots,
- README section "inverse problems".

## Phase 6 - polish
Objective:
- make the repo star-worthy.

Deliverables:
- clean README,
- reproducibility notes,
- model cards,
- dataset card,
- release tag.

---

# 11. Scientific honesty rules

The project should be scientifically ambitious but honest.

## 11.1 What can be claimed
Safe claims:
- We simulate reduced drift-wave turbulence using HW.
- We train ML surrogates on synthetic data generated by a trusted numerical solver.
- We benchmark rollout quality, spectral statistics, and runtime.
- We optionally solve inverse parameter-identification problems with physics-informed models.

## 11.2 What should NOT be claimed
Do not claim:
- experimental validation,
- reactor-grade predictive capability,
- equivalence to gyrokinetic turbulence,
- PINNs beat solvers on full chaotic forward integration,
- real-time tokamak control relevance unless you truly benchmark that.

---

# 12. Suggested benchmark table for the README

The final README should compare at least:
- solver,
- persistence baseline,
- CNN/U-Net,
- FNO.

Suggested columns:
- one-step relative L2,
- 50-step rollout error,
- spectrum error,
- runtime per rollout,
- speedup.

Optional rows:
- PINN inverse parameter error.

---

# 13. README tone and positioning

The repo should feel like:
- advanced enough for a plasma-physics audience,
- polished enough for an ML engineer audience,
- visual enough to win curiosity,
- disciplined enough to be portfolio-grade.

The README opening should quickly communicate:
1. what physical system is solved,
2. what ML models are used,
3. why it matters,
4. and where to look first (GIFs, quickstart, benchmark table).

---

# 14. Minimum viable references for the agent to read before coding

The agent should familiarize itself with the following themes and references.

## 14.1 Core plasma-physics references
1. **Hasegawa, A. & Wakatani, M. (1983). “Plasma Edge Turbulence.” Physical Review Letters 50, 682.**
   - Classic reduced-model motivation for resistive drift-wave turbulence.

2. **Hasegawa, A. & Wakatani, M. (1987). “Self-organization of electrostatic turbulence in a cylindrical plasma.” Physical Review Letters 59, 1581.**
   - Important for the self-organization / zonal-flow narrative.

3. **Schmid, B., Manz, P., Ramisch, M., & Stroth, U. (2017). “Collisional Scaling of the Energy Transfer in Drift-Wave Zonal Flow Turbulence.” Physical Review Letters 118, 055001.**
   - Useful for understanding energy transfer and drift-wave / zonal-flow coupling in HW-type turbulence.

4. **Abramovic, I. et al. (2022). “Data-driven model discovery for plasma turbulence modelling.” Journal of Plasma Physics.**
   - Helpful because it works directly with HW / modified HW generated data and reinforces that HW is a valid data-driven plasma-turbulence playground.

## 14.2 Scientific ML / surrogate references
5. **Li, Z. et al. (2021). “Fourier Neural Operator for Parametric Partial Differential Equations.” ICLR / OpenReview.**
   - Foundational neural-operator reference for PDE surrogate learning.

6. **Raissi, M., Perdikaris, P., & Karniadakis, G. (2019). “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.” Journal of Computational Physics 378, 686-707.**
   - Foundational PINN reference.

7. **Gahr, C., Farcas, I.-G., & Jenko, F. (2024). “Scientific machine learning based reduced-order models for plasma turbulence simulations.” Physics of Plasmas 31, 113904.**
   - Very relevant because it uses HW as the representative plasma-turbulence testbed for reduced-order SciML.

8. **Clavier, B. et al. (2025). “A generative machine learning surrogate model of plasma turbulence.” Physical Review E 111, L013202.**
   - Direct evidence that HW-based plasma-turbulence surrogates are a live and credible research direction.

9. **Artigues, V., Greif, R., & Jenko, F. (2025). “Accelerating Hasegawa-Wakatani simulations with machine learning for out-of-distribution predictions.” Plasma Physics and Controlled Fusion 67, 045018.**
   - Especially relevant for OOD evaluation strategy.

10. **Chen, X. et al. (2026). “Convolution Operator Network for Forward and Inverse Problems (FI-Conv): Application to Plasma Turbulence Simulations.” arXiv:2602.04287.**
   - Recent and highly relevant reference because it couples forward prediction and inverse parameter estimation on HW turbulence.

## 14.3 Practical numerical / ML design lessons the agent should internalize
- Start with a **correct and simple solver** before chasing fancy ML.
- Use the solver to generate your own trustworthy datasets.
- Benchmark **short-horizon accuracy** and **longer-horizon statistics** separately.
- For chaotic turbulence, expect long rollouts to diverge pointwise; compare **statistics and spectra**, not only pixelwise fields.
- Use PINNs primarily for **inverse problems or parameter estimation** in this project.
- Make visual assets reproducible and headless.

---

# 15. Final target identity of the repo

If executed well, this repo should read as:

> “A compact but serious plasma-turbulence SciML lab: physically motivated, numerically grounded, visually impressive, and structured like a modern applied-science / ML-engineering project.”

That identity is more important than maximizing theoretical novelty. The repo should win on:
- clarity,
- credibility,
- visuals,
- reproducibility,
- and disciplined implementation.

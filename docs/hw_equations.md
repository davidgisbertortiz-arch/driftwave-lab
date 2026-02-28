# Hasegawa–Wakatani equations — implementation notes

## Equations implemented

This solver implements the **2D electrostatic Hasegawa–Wakatani (HW) system** on a doubly periodic square domain `[0, Lx) × [0, Ly)`.

### State variables

| Symbol | Name | Description |
|--------|------|-------------|
| `n(x,y,t)` | Density fluctuation | Perturbation of plasma density |
| `φ(x,y,t)` | Electrostatic potential | Drives E×B flow |
| `ω(x,y,t)` | Vorticity | `ω = ∇²φ` |

### PDE system

```
∂n/∂t  + {φ, n}  = α (φ − n) − κ ∂φ/∂y + D ∇²n
∂ω/∂t  + {φ, ω}  = α (φ − n)           + ν ∇²ω
ω = ∇²φ
```

where the **Poisson bracket** is defined as:

```
{a, b} = ∂a/∂x · ∂b/∂y − ∂a/∂y · ∂b/∂x
```

### Parameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| `alpha` | α | Adiabaticity / parallel electron coupling. Controls strength of the resistive coupling between density and potential. Large α → adiabatic limit (n ≈ φ). |
| `kappa` | κ | Background density-gradient drive. Source of the drift-wave instability. |
| `D` | D | Density diffusion coefficient. Regularises small scales in the density field. |
| `nu` | ν | Viscosity coefficient. Regularises small scales in the vorticity field. |

### Physical interpretation

- **Poisson bracket terms** `{φ, n}` and `{φ, ω}` represent nonlinear E×B advection of density and vorticity by the self-consistent electrostatic flow.
- **`α(φ − n)`** models the response of parallel (field-aligned) electrons to potential–density mismatches. This term couples the two evolution equations.
- **`−κ ∂φ/∂y`** represents the free-energy source: the background density gradient (assumed along x) interacts with the E×B drift (along y) to drive drift-wave instability.
- **Diffusion/viscosity** (`D ∇²n`, `ν ∇²ω`) provide numerical regularisation and represent unresolved collisional transport.

### Sign and normalisation conventions

The implementation follows the sign conventions as written above, consistent with the standard references (Hasegawa & Wakatani 1983, 1987) and the PROJECT_SPEC. The background density gradient is assumed to lie along −x, so the drive term appears as `−κ ∂φ/∂y`.

---

## Numerical method

### Spatial discretisation

**Fourier pseudo-spectral** on a uniform periodic grid of `Nx × Ny` points.

- Derivatives are computed exactly in Fourier space via multiplication by `i·kx` or `i·ky`.
- The Poisson equation `∇²φ = ω` is inverted spectrally: `φ̂(k) = −ω̂(k) / |k|²` (with the zero mode set to zero — no net potential).
- The Poisson bracket is evaluated by computing products in physical space and transforming back, with **2/3-rule de-aliasing** applied to suppress aliasing errors.

### Time integration

**RK2 (Heun / improved Euler)** — a simple two-stage explicit method:

```
k1 = f(uⁿ)
k2 = f(uⁿ + Δt · k1)
uⁿ⁺¹ = uⁿ + Δt/2 · (k1 + k2)
```

This is adequate for the MVP at moderate time steps. An upgrade to ETDRK4 (exponential time-differencing RK4) is planned for later phases; it would allow treating the linear stiff terms (diffusion) exactly.

### De-aliasing

The 2/3-rule mask zeros out Fourier modes in the outer third of the wavenumber range in both directions. It is applied:
- to each Poisson bracket evaluation (nonlinear product),
- after each full RK2 step.

---

## MVP simplifications

This first version makes the following deliberate simplifications:

1. **RK2 only** — stable for small `Δt`, but not optimal for stiff problems. Diffusion terms are treated explicitly (not split off exponentially).
2. **Simple de-aliasing** — the Poisson bracket uses the straightforward physical-space multiplication approach, not the conservative Arakawa scheme. This is acceptable for the moderate resolutions targeted here.
3. **No hyperdiffusion** — only Laplacian diffusion (`∇²`) is used. Hyperdiffusion (`∇⁴` or `∇⁶`) could be added later for more aggressive regularisation at higher resolutions.
4. **No adaptive time stepping** — `Δt` is fixed throughout the run.

---

## References

1. Hasegawa, A. & Wakatani, M. (1983). "Plasma Edge Turbulence." *Phys. Rev. Lett.* **50**, 682.
2. Hasegawa, A. & Wakatani, M. (1987). "Self-organization of electrostatic turbulence." *Phys. Rev. Lett.* **59**, 1581.
3. See `PROJECT_SPEC.md` § 14 for the complete reference list.

# Data Pipeline

This document describes the dataset generation and loading pipeline for
the driftwave-lab project (Phase 2 of the roadmap).

## Overview

The data pipeline has three stages:

1. **Parameter sampling** — deterministic Latin-style uniform sampling of
   physics parameters (α, κ, D, ν) from configurable ranges.
2. **Trajectory generation** — running the HW spectral solver for each
   parameter combination with unique IC seeds.
3. **Dataset loading** — a PyTorch `Dataset` that serves next-step
   prediction pairs to the training loop.

## Quick start

```bash
# Generate a tiny dataset for testing (~6 trajectories, 16×16 grid)
python scripts/generate_dataset.py --config configs/dataset_tiny.yaml

# Generate the full default dataset (50 + 5 OOD trajectories, 64×64 grid)
python scripts/generate_dataset.py --config configs/dataset.yaml
```

## Configuration

Dataset generation is controlled by a YAML config file.  See
[configs/dataset.yaml](../configs/dataset.yaml) for the full default and
[configs/dataset_tiny.yaml](../configs/dataset_tiny.yaml) for the CI-friendly
version.

### Key fields

| Field | Description |
|---|---|
| `grid.nx`, `grid.ny` | Spatial resolution |
| `time.n_steps` | Integration steps per trajectory |
| `time.save_every` | Snapshot cadence |
| `n_trajectories` | Total in-distribution trajectories |
| `split.train/val/test` | Fractional or absolute split sizes |
| `n_ood` | Number of out-of-distribution trajectories |
| `parameter_ranges.alpha` | Uniform range for α (in-distribution) |
| `ood.alpha` | Uniform range for α (OOD set) |
| `seed` | Master seed for full reproducibility |

## Trajectory-level splits

Splits are assigned at the **trajectory level** — all frames from one
simulation belong to the same split.  This prevents temporal data leakage.

| Split | Purpose |
|---|---|
| `train` | Model training |
| `val` | Hyperparameter tuning / early stopping |
| `test` | Final in-distribution evaluation |
| `ood` | Out-of-distribution generalisation (shifted α range) |

## File format

Each trajectory is saved as a compressed `.npz` file containing:

- `times` — shape `(T,)`
- `n` — density field, shape `(T, nx, ny)`
- `omega` — vorticity field, shape `(T, nx, ny)`
- `phi` — potential field, shape `(T, nx, ny)`
- `metadata` — JSON string with parameters, seed, split label, etc.

A `manifest.json` file in the output directory indexes all trajectories
with their parameters, split assignments and file names.

## PyTorch dataset

```python
from driftwave_lab.data.dataset import HWNextStepDataset

ds = HWNextStepDataset("data/raw/manifest.json", split="train")
inp, tgt = ds[0]   # inp: [2, H, W],  tgt: [2, H, W]
# Channels: (n_t, phi_t) → (n_{t+1}, phi_{t+1})
```

The dataset:
- Reads from the manifest to select trajectories for the requested split.
- Loads NPZ files lazily and caches them in memory.
- Indexes consecutive frame pairs across all trajectories.
- Returns `float32` tensors by default.

### DataLoader integration

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
for inp_batch, tgt_batch in loader:
    # inp_batch: [B, 2, H, W]
    pass
```

## Reproducibility

The entire pipeline is deterministic given the same `seed`:

1. The master RNG generates all per-trajectory parameters and IC seeds.
2. Each trajectory uses its own IC seed for the random perturbation.
3. Split assignment is done via a deterministic shuffle of trajectory
   indices.

Running `generate_dataset.py` twice with the same config produces
bit-identical NPZ files and manifest.

## Module reference

| Module | Key exports |
|---|---|
| `driftwave_lab.data.generator` | `DatasetConfig`, `generate_dataset`, `sample_trajectory_specs`, `TrajectorySpec` |
| `driftwave_lab.data.dataset` | `HWNextStepDataset`, `load_manifest`, `get_split_files` |
| `driftwave_lab.data.io` | `save_trajectory`, `load_trajectory` |
| `scripts/generate_dataset.py` | CLI entry point |

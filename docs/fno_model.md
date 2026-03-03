# FNO 2D Hero Model + Benchmarking

This document describes the ML surrogate pipeline introduced in **Phase 5**:
the Fourier Neural Operator (FNO) hero model, the U-Net baseline, the unified
training loop, autoregressive rollout evaluation, and runtime benchmarking.

## Architecture overview

### FNO 2D (`models/fno2d.py`)

Following [Li et al. (2021)](https://arxiv.org/abs/2010.08895):

```
Input (B,2,H,W) → Lift (1×1 conv) → [Fourier Layer × N] → Proj → Output (B,2,H,W)
```

Each **Fourier Layer** computes:

$$
v^{(l+1)}(x) = \sigma\!\bigl(W v^{(l)}(x) + \mathcal{K}(v^{(l)})(x)\bigr)
$$

where $\mathcal{K}$ is a spectral convolution via `rfft2` → learned complex
multiply on the low-frequency modes → `irfft2`, and $W$ is a pointwise 1 × 1
convolution. Activation is GeLU.

**Default hyperparameters (production):**

| Parameter | Value |
|-----------|-------|
| `modes`   | 16    |
| `width`   | 64    |
| `n_layers`| 4     |
| Parameters| ~2.4 M |

### U-Net baseline (`models/unet.py`)

A 3-level encoder-decoder CNN with skip connections, batch-norm, and
**circular padding** (respecting the periodic domain).

**Default hyperparameters:**

| Parameter      | Value |
|----------------|-------|
| `base_filters` | 32    |
| Levels         | 3     |
| Parameters     | ~0.5 M |

### Model factory (`models/__init__.py`)

```python
from driftwave_lab.models import build_model

model = build_model({"name": "fno2d", "modes": 16, "width": 64, "n_layers": 4})
model = build_model({"name": "unet", "base_filters": 32})
```

## Training

The unified training loop (`training/train_fno.py`) handles both architectures.
It reads a YAML config, builds the model via the factory, and trains with
Adam + cosine-annealing LR schedule.

### Run training

```bash
# Production FNO (requires pre-generated dataset in data/raw/)
python scripts/train.py --config configs/train_fno.yaml

# Production U-Net
python scripts/train.py --config configs/train_unet.yaml

# Tiny configs for CI / quick smoke tests
python scripts/train.py --config configs/train_fno_tiny.yaml
python scripts/train.py --config configs/train_unet_tiny.yaml
```

**Outputs:**
- `checkpoints/<model>_best.pt` — best-validation checkpoint
- `checkpoints/history.json` — per-epoch metrics (train/val MSE, rel L2)

### Config structure

```yaml
model:
  name: fno2d          # or "unet"
  modes: 16
  width: 64
  n_layers: 4

training:
  epochs: 100
  batch_size: 8
  lr: 1.0e-3
  scheduler: cosine    # or "step" or null

data:
  dataset_dir: data/raw

checkpoint_dir: checkpoints
seed: 0
```

## Evaluation

### Metrics (`evaluation/metrics.py`)

| Function | Description |
|----------|-------------|
| `mse(pred, true)` | Mean squared error |
| `rmse(pred, true)` | Root mean squared error |
| `relative_l2(pred, true)` | Relative L² error |
| `channel_mse(pred, true)` | Per-channel MSE |
| `rollout_errors(preds, trues)` | Step-wise MSE + rel L² |

### Autoregressive rollout (`evaluation/rollout.py`)

```python
from driftwave_lab.evaluation.rollout import autoregressive_rollout, evaluate_rollout

# Generate a multi-step prediction trajectory
traj = autoregressive_rollout(model, x0, n_steps=50)

# Compare against ground truth
result = evaluate_rollout(model, truth_tensor)
# result["mse"], result["rel_l2"], result["preds"]
```

### Rollout demo script

```bash
python scripts/rollout_demo.py \
    --checkpoint checkpoints/fno2d_best.pt \
    --data data/raw/manifest.json \
    --steps 30 \
    --save outputs/rollout_fno.npz
```

## Benchmarking

### Library API (`evaluation/benchmark.py`)

```python
from driftwave_lab.evaluation.benchmark import benchmark_model

result = benchmark_model(model, sample, rollout_steps=20)
print(result.one_step_ms, result.rollout_ms, result.n_params)
```

### CLI script

```bash
python scripts/benchmark.py \
    --checkpoint checkpoints/fno2d_best.pt \
    --unet-checkpoint checkpoints/unet_best.pt \
    --resolution 64 \
    --rollout-steps 20 \
    --save outputs/benchmark.json
```

Output includes system metadata, per-model inference latencies, and parameter
counts.

## Tests

All ML pipeline tests live in `tests/test_ml_pipeline.py`:

```bash
pytest tests/test_ml_pipeline.py -v
```

Test classes:
- `TestFNO2d` — forward shape, no-NaN, gradients
- `TestUNet` — forward shape, no-NaN, gradients
- `TestBuildModel` — factory dispatch, unknown model error
- `TestMetrics` — MSE zero-self, RMSE positive, rel L² range, rollout errors
- `TestRollout` — trajectory length, evaluate vs ground truth
- `TestBenchmark` — timing produces valid results
- `TestCheckpoint` — save/load roundtrip determinism
- `TestMiniTraining` — end-to-end training with synthetic data (FNO + U-Net)

All tests use tiny models (width=8, modes=4) on 16×16 grids and run on CPU
in a few seconds.

## Design decisions

1. **Unified trainer** — `train_fno.py` handles both FNO and U-Net via the
   model factory. `train_baseline.py` re-exports for discoverability.
2. **Spectral convolution via `rfft2`** — more efficient than full `fft2`,
   naturally handles real-valued input fields.
3. **Circular padding in U-Net** — preserves the periodic boundary conditions
   from the HW solver domain.
4. **No PINNs / Streamlit** — deferred to Phase 6 per project spec.
5. **Checkpoint format** — includes full config dict for reproducibility.

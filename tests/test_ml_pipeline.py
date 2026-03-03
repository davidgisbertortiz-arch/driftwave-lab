"""Tests for the ML pipeline: models, metrics, rollout, benchmark, training.

These tests are designed to be fast (tiny models, synthetic data) and to run
without a GPU.  They do NOT require a pre-generated dataset.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping ML pipeline tests")

from driftwave_lab.evaluation.benchmark import BenchmarkResult, benchmark_model
from driftwave_lab.evaluation.metrics import (
    channel_mse,
    mse,
    relative_l2,
    rmse,
    rollout_errors,
)
from driftwave_lab.evaluation.rollout import autoregressive_rollout, evaluate_rollout
from driftwave_lab.models import build_model
from driftwave_lab.models.fno2d import FNO2d
from driftwave_lab.models.unet import UNet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W = 16, 16
C_IN, C_OUT = 2, 2
BATCH = 2


@pytest.fixture()
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a (inp, tgt) pair of shape (B, 2, H, W)."""
    torch.manual_seed(0)
    inp = torch.randn(BATCH, C_IN, H, W)
    tgt = torch.randn(BATCH, C_OUT, H, W)
    return inp, tgt


@pytest.fixture()
def fno_tiny() -> FNO2d:
    return FNO2d(in_channels=C_IN, out_channels=C_OUT, modes=4, width=8, n_layers=2)


@pytest.fixture()
def unet_tiny() -> UNet:
    return UNet(in_channels=C_IN, out_channels=C_OUT, base_filters=4)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestFNO2d:
    def test_forward_shape(self, fno_tiny: FNO2d, sample_batch):
        inp, _ = sample_batch
        out = fno_tiny(inp)
        assert out.shape == (BATCH, C_OUT, H, W)

    def test_no_nan(self, fno_tiny: FNO2d, sample_batch):
        inp, _ = sample_batch
        out = fno_tiny(inp)
        assert not torch.isnan(out).any()

    def test_gradients(self, fno_tiny: FNO2d, sample_batch):
        inp, tgt = sample_batch
        out = fno_tiny(inp)
        loss = torch.nn.functional.mse_loss(out, tgt)
        loss.backward()
        for p in fno_tiny.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestUNet:
    def test_forward_shape(self, unet_tiny: UNet, sample_batch):
        inp, _ = sample_batch
        out = unet_tiny(inp)
        assert out.shape == (BATCH, C_OUT, H, W)

    def test_no_nan(self, unet_tiny: UNet, sample_batch):
        inp, _ = sample_batch
        out = unet_tiny(inp)
        assert not torch.isnan(out).any()

    def test_gradients(self, unet_tiny: UNet, sample_batch):
        inp, tgt = sample_batch
        out = unet_tiny(inp)
        loss = torch.nn.functional.mse_loss(out, tgt)
        loss.backward()
        for p in unet_tiny.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestBuildModel:
    def test_build_fno(self):
        cfg = {"name": "fno2d", "modes": 4, "width": 8, "n_layers": 2}
        model = build_model(cfg)
        assert isinstance(model, FNO2d)

    def test_build_unet(self):
        cfg = {"name": "unet", "base_filters": 4}
        model = build_model(cfg)
        assert isinstance(model, UNet)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_model({"name": "nonexistent"})


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_mse_zero(self):
        x = torch.randn(2, 2, 8, 8)
        assert mse(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_rmse_positive(self, sample_batch):
        inp, tgt = sample_batch
        assert rmse(inp, tgt).item() > 0

    def test_relative_l2_range(self, sample_batch):
        inp, tgt = sample_batch
        val = relative_l2(inp, tgt).item()
        assert val >= 0.0

    def test_channel_mse_shape(self, sample_batch):
        inp, tgt = sample_batch
        ch = channel_mse(inp, tgt)
        assert ch.shape == (C_IN,)

    def test_rollout_errors(self):
        preds = [torch.randn(2, 8, 8) for _ in range(5)]
        trues = [torch.randn(2, 8, 8) for _ in range(5)]
        errs = rollout_errors(preds, trues)
        assert len(errs["mse"]) == 5
        assert len(errs["rel_l2"]) == 5


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------


class TestRollout:
    def test_autoregressive_rollout_length(self, fno_tiny: FNO2d):
        x0 = torch.randn(1, C_IN, H, W)
        n_steps = 5
        traj = autoregressive_rollout(fno_tiny, x0, n_steps)
        assert len(traj) == n_steps + 1  # includes initial condition

    def test_evaluate_rollout(self, fno_tiny: FNO2d):
        truth = torch.randn(6, C_IN, H, W)  # 6 frames → 5 steps
        result = evaluate_rollout(fno_tiny, truth)
        assert len(result["mse"]) == 5
        assert len(result["rel_l2"]) == 5


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_model(self, fno_tiny: FNO2d):
        sample = torch.randn(1, C_IN, H, W)
        res = benchmark_model(fno_tiny, sample, rollout_steps=3, warmup=1, repeats=2)
        assert isinstance(res, BenchmarkResult)
        assert res.one_step_ms > 0
        assert res.rollout_ms > 0
        assert res.n_params > 0

    def test_to_dict(self, fno_tiny: FNO2d):
        sample = torch.randn(1, C_IN, H, W)
        res = benchmark_model(fno_tiny, sample, rollout_steps=2, warmup=1, repeats=1)
        d = res.to_dict()
        assert "name" in d and "one_step_ms" in d


# ---------------------------------------------------------------------------
# Checkpoint save / load roundtrip
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_roundtrip(self, fno_tiny: FNO2d):
        cfg = {
            "model": {"name": "fno2d", "modes": 4, "width": 8, "n_layers": 2},
            "training": {},
            "data": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pt"
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": fno_tiny.state_dict(),
                    "config": cfg,
                },
                path,
            )

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            loaded = build_model(ckpt["config"]["model"])
            loaded.load_state_dict(ckpt["model_state_dict"])
            loaded.eval()

            x = torch.randn(1, C_IN, H, W)
            fno_tiny.eval()
            with torch.no_grad():
                y1 = fno_tiny(x)
                y2 = loaded(x)
            assert torch.allclose(y1, y2, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: mini training loop (synthetic data, no dataset files)
# ---------------------------------------------------------------------------


class TestMiniTraining:
    """Verify the training loop runs for 1 epoch on synthetic data."""

    def _make_synthetic_dataset(self, tmp_dir: Path, n_traj: int = 4, n_steps: int = 5):
        """Create a tiny manifest + NPZ files for testing."""
        trajectories = []
        for i in range(n_traj):
            fname = f"traj_{i:03d}.npz"
            n_arr = np.random.randn(n_steps, H, W).astype(np.float32)
            phi_arr = np.random.randn(n_steps, H, W).astype(np.float32)
            np.savez(str(tmp_dir / fname), n=n_arr, phi=phi_arr)
            split = "train" if i < n_traj - 1 else "val"
            trajectories.append({"file": fname, "split": split, "kappa": 1.0})

        manifest = {"format_version": 1, "trajectories": trajectories}
        with open(tmp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

    def test_train_fno_tiny(self):
        from driftwave_lab.training.train_fno import train

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            self._make_synthetic_dataset(data_dir)

            cfg = {
                "model": {
                    "name": "fno2d",
                    "in_channels": 2,
                    "out_channels": 2,
                    "modes": 4,
                    "width": 8,
                    "n_layers": 2,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 4,
                    "lr": 1e-3,
                    "scheduler": "cosine",
                },
                "data": {"dataset_dir": str(data_dir)},
                "checkpoint_dir": str(tmp_path / "ckpts"),
                "seed": 42,
            }

            result = train(cfg, device="cpu", verbose=False)
            assert "history" in result
            assert len(result["history"]["train_mse"]) == 2
            assert Path(result["checkpoint_path"]).exists()

    def test_train_unet_tiny(self):
        from driftwave_lab.training.train_fno import train

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            self._make_synthetic_dataset(data_dir)

            cfg = {
                "model": {
                    "name": "unet",
                    "in_channels": 2,
                    "out_channels": 2,
                    "base_filters": 4,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "lr": 1e-3,
                    "scheduler": "cosine",
                },
                "data": {"dataset_dir": str(data_dir)},
                "checkpoint_dir": str(tmp_path / "ckpts"),
                "seed": 42,
            }

            result = train(cfg, device="cpu", verbose=False)
            assert Path(result["checkpoint_path"]).exists()

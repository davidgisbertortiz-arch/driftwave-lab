"""Tests for the PyTorch dataset wrapper (data/dataset.py).

These tests generate a tiny dataset in a temp directory, then exercise
HWNextStepDataset and the manifest helpers.  Tests are skipped if
``torch`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from driftwave_lab.data.generator import DatasetConfig, generate_dataset

# Skip entire module if torch is missing
torch = pytest.importorskip("torch")

from driftwave_lab.data.dataset import (  # noqa: E402
    HWNextStepDataset,
    get_split_files,
    get_split_metadata,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Shared fixture: tiny generated dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a tiny dataset once for the entire module."""
    out = tmp_path_factory.mktemp("dataset")
    cfg = DatasetConfig(
        nx=16, ny=16, lx=20.0, ly=20.0,
        dt=0.05, n_steps=20, save_every=10,
        n_train=2, n_val=1, n_test=1, n_ood=1,
        seed=42,
        output_dir=str(out),
    )
    generate_dataset(cfg)
    return out


@pytest.fixture()
def manifest_path(tiny_data_dir: Path) -> Path:
    return tiny_data_dir / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


class TestManifestHelpers:
    def test_load_manifest(self, manifest_path: Path):
        m = load_manifest(manifest_path)
        assert "trajectories" in m
        assert "config" in m

    def test_get_split_files(self, manifest_path: Path, tiny_data_dir: Path):
        m = load_manifest(manifest_path)
        train_files = get_split_files(m, "train", data_dir=tiny_data_dir)
        assert len(train_files) == 2
        assert all(f.suffix == ".npz" for f in train_files)
        assert all(f.exists() for f in train_files)

    def test_get_split_metadata(self, manifest_path: Path):
        m = load_manifest(manifest_path)
        val_meta = get_split_metadata(m, "val")
        assert len(val_meta) == 1
        assert val_meta[0]["split"] == "val"

    def test_ood_split(self, manifest_path: Path, tiny_data_dir: Path):
        m = load_manifest(manifest_path)
        ood_files = get_split_files(m, "ood", data_dir=tiny_data_dir)
        assert len(ood_files) == 1


# ---------------------------------------------------------------------------
# HWNextStepDataset
# ---------------------------------------------------------------------------


class TestHWNextStepDataset:
    def test_construction(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train")
        assert ds.n_trajectories == 2
        assert len(ds) > 0

    def test_sample_shapes(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train")
        inp, tgt = ds[0]
        # Default channels = ("n", "phi") → 2 channels
        assert inp.shape == (2, 16, 16)
        assert tgt.shape == (2, 16, 16)

    def test_dtype(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train")
        inp, tgt = ds[0]
        assert inp.dtype == torch.float32
        assert tgt.dtype == torch.float32

    def test_no_nans(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train")
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert not torch.isnan(inp).any(), f"NaN in input at index {i}"
            assert not torch.isnan(tgt).any(), f"NaN in target at index {i}"

    def test_preload(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train", preload=True)
        assert len(ds._traj_cache) == ds.n_trajectories  # noqa: SLF001
        inp, tgt = ds[0]
        assert inp.shape[0] == 2

    def test_val_split(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="val")
        assert ds.n_trajectories == 1
        assert len(ds) > 0

    def test_ood_split(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="ood")
        assert ds.n_trajectories == 1

    def test_empty_split_raises(self, manifest_path: Path):
        # No split named "nonexistent"
        with pytest.raises(ValueError, match="No trajectories found"):
            HWNextStepDataset(manifest_path, split="nonexistent")

    def test_spatial_shape(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train")
        assert ds.spatial_shape == (16, 16)

    def test_repr(self, manifest_path: Path):
        ds = HWNextStepDataset(manifest_path, split="train")
        r = repr(ds)
        assert "train" in r
        assert "n_traj=2" in r

    def test_consecutive_pairs(self, manifest_path: Path):
        """Verify that (input[t], target[t]) and (input[t+1], target[t+1])
        have target[t] == input[t+1] within the same trajectory."""
        ds = HWNextStepDataset(manifest_path, split="train")

        # Get pairs from the same trajectory
        # First trajectory has n_steps=20, save_every=10 → 3 snapshots → 2 pairs
        # (traj_idx, time_step) pairs: (0,0) and (0,1)
        idx_map = ds._index_map  # noqa: SLF001

        # Find consecutive pairs within same trajectory
        for i in range(len(idx_map) - 1):
            traj_a, t_a = idx_map[i]
            traj_b, t_b = idx_map[i + 1]
            if traj_a == traj_b and t_b == t_a + 1:
                _, tgt_a = ds[i]
                inp_b, _ = ds[i + 1]
                torch.testing.assert_close(tgt_a, inp_b)
                break
        else:
            pytest.skip("No consecutive same-traj pairs found")

    def test_total_samples(self, manifest_path: Path):
        """Total samples = sum over trajectories of (T_i - 1)."""
        ds = HWNextStepDataset(manifest_path, split="train")
        # 2 trajectories, each with 3 snapshots → 2*(3-1) = 4 samples
        assert len(ds) == 4

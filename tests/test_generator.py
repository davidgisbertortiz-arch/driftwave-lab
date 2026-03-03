"""Tests for dataset generation (data/generator.py) and data pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from driftwave_lab.data.generator import (
    DatasetConfig,
    GenerationResult,
    TrajectorySpec,
    generate_dataset,
    run_single_trajectory,
    sample_trajectory_specs,
)


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.nx == 64
        assert cfg.n_train == 30
        assert cfg.alpha_range == (0.5, 1.5)
        assert cfg.ood_alpha_range == (1.5, 2.5)

    def test_from_dict_fractional_splits(self):
        d = {
            "n_trajectories": 10,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "seed": 7,
        }
        cfg = DatasetConfig.from_dict(d)
        assert cfg.n_train == 6
        assert cfg.n_val == 2
        assert cfg.n_test == 2
        assert cfg.seed == 7

    def test_from_dict_absolute_splits(self):
        d = {
            "split": {"train": 5, "val": 2, "test": 3},
            "grid": {"nx": 32, "ny": 32},
        }
        cfg = DatasetConfig.from_dict(d)
        assert cfg.n_train == 5
        assert cfg.n_val == 2
        assert cfg.n_test == 3
        assert cfg.nx == 32

    def test_from_dict_parameter_ranges(self):
        d = {
            "parameter_ranges": {"alpha": [0.1, 0.9], "kappa": [1.0, 2.0]},
            "ood": {"alpha": [2.0, 3.0]},
        }
        cfg = DatasetConfig.from_dict(d)
        assert cfg.alpha_range == (0.1, 0.9)
        assert cfg.kappa_range == (1.0, 2.0)
        assert cfg.ood_alpha_range == (2.0, 3.0)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------


class TestSampling:
    @pytest.fixture()
    def tiny_cfg(self) -> DatasetConfig:
        return DatasetConfig(
            nx=16, ny=16, lx=20.0, ly=20.0,
            dt=0.05, n_steps=20, save_every=10,
            n_train=3, n_val=1, n_test=1, n_ood=2,
            seed=42,
        )

    def test_correct_counts(self, tiny_cfg: DatasetConfig):
        specs = sample_trajectory_specs(tiny_cfg)
        splits = [s.split for s in specs]
        assert splits.count("train") == 3
        assert splits.count("val") == 1
        assert splits.count("test") == 1
        assert splits.count("ood") == 2
        assert len(specs) == 7

    def test_deterministic(self, tiny_cfg: DatasetConfig):
        s1 = sample_trajectory_specs(tiny_cfg)
        s2 = sample_trajectory_specs(tiny_cfg)
        for a, b in zip(s1, s2, strict=True):
            assert a.alpha == b.alpha
            assert a.ic_seed == b.ic_seed
            assert a.split == b.split

    def test_ood_alpha_in_range(self, tiny_cfg: DatasetConfig):
        specs = sample_trajectory_specs(tiny_cfg)
        ood = [s for s in specs if s.split == "ood"]
        for s in ood:
            assert tiny_cfg.ood_alpha_range[0] <= s.alpha <= tiny_cfg.ood_alpha_range[1]

    def test_id_alpha_in_range(self, tiny_cfg: DatasetConfig):
        specs = sample_trajectory_specs(tiny_cfg)
        id_specs = [s for s in specs if s.split != "ood"]
        for s in id_specs:
            assert tiny_cfg.alpha_range[0] <= s.alpha <= tiny_cfg.alpha_range[1]

    def test_unique_ic_seeds(self, tiny_cfg: DatasetConfig):
        specs = sample_trajectory_specs(tiny_cfg)
        seeds = [s.ic_seed for s in specs]
        assert len(set(seeds)) == len(seeds), "IC seeds must be unique"

    def test_sorted_by_index(self, tiny_cfg: DatasetConfig):
        specs = sample_trajectory_specs(tiny_cfg)
        indices = [s.index for s in specs]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Single trajectory runner
# ---------------------------------------------------------------------------


class TestRunSingle:
    def test_shapes_and_metadata(self):
        cfg = DatasetConfig(
            nx=16, ny=16, lx=20.0, ly=20.0,
            dt=0.05, n_steps=20, save_every=10,
            n_train=1, n_val=0, n_test=0, n_ood=0,
            seed=0,
        )
        spec = TrajectorySpec(
            index=0, split="train",
            alpha=1.0, kappa=1.0, D=0.01, nu=0.01,
            ic_seed=99,
        )
        arrays, meta, elapsed = run_single_trajectory(spec, cfg)

        # n_steps=20, save_every=10 → snapshots at step 0, 10, 20 → 3
        assert arrays["n"].shape == (3, 16, 16)
        assert arrays["phi"].shape == (3, 16, 16)
        assert arrays["omega"].shape == (3, 16, 16)
        assert arrays["times"].shape == (3,)
        assert meta["alpha"] == 1.0
        assert meta["ic_seed"] == 99
        assert elapsed > 0

    def test_no_nans(self):
        cfg = DatasetConfig(
            nx=16, ny=16, lx=20.0, ly=20.0,
            dt=0.05, n_steps=20, save_every=10,
            n_train=1, n_val=0, n_test=0, n_ood=0,
            seed=1,
        )
        spec = TrajectorySpec(
            index=0, split="train",
            alpha=0.8, kappa=1.1, D=0.01, nu=0.01,
            ic_seed=7,
        )
        arrays, _, _ = run_single_trajectory(spec, cfg)
        for key in ("n", "omega", "phi"):
            assert not np.any(np.isnan(arrays[key])), f"NaN found in {key}"


# ---------------------------------------------------------------------------
# Full dataset generation (end-to-end)
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    @pytest.fixture()
    def tiny_dataset(self, tmp_path: Path) -> GenerationResult:
        cfg = DatasetConfig(
            nx=16, ny=16, lx=20.0, ly=20.0,
            dt=0.05, n_steps=20, save_every=10,
            n_train=2, n_val=1, n_test=1, n_ood=1,
            seed=42,
            output_dir=str(tmp_path / "data"),
        )
        return generate_dataset(cfg)

    def test_manifest_exists(self, tiny_dataset: GenerationResult):
        assert tiny_dataset.manifest_path.exists()

    def test_correct_file_count(self, tiny_dataset: GenerationResult):
        npz_files = list(tiny_dataset.output_dir.glob("traj_*.npz"))
        assert len(npz_files) == tiny_dataset.n_trajectories

    def test_manifest_content(self, tiny_dataset: GenerationResult):
        with open(tiny_dataset.manifest_path) as f:
            manifest = json.load(f)
        assert "config" in manifest
        assert "trajectories" in manifest
        assert "splits" in manifest
        assert len(manifest["trajectories"]) == tiny_dataset.n_trajectories

    def test_split_counts_in_manifest(self, tiny_dataset: GenerationResult):
        with open(tiny_dataset.manifest_path) as f:
            manifest = json.load(f)
        splits = [e["split"] for e in manifest["trajectories"]]
        assert splits.count("train") == 2
        assert splits.count("val") == 1
        assert splits.count("test") == 1
        assert splits.count("ood") == 1

    def test_npz_loadable_and_correct_shape(self, tiny_dataset: GenerationResult):
        npz_files = sorted(tiny_dataset.output_dir.glob("traj_*.npz"))
        data = np.load(str(npz_files[0]), allow_pickle=False)
        # n_steps=20, save_every=10 → 3 snapshots
        assert data["n"].shape == (3, 16, 16)
        assert data["phi"].shape == (3, 16, 16)

    def test_no_nans_in_dataset(self, tiny_dataset: GenerationResult):
        for f in tiny_dataset.output_dir.glob("traj_*.npz"):
            data = np.load(str(f), allow_pickle=False)
            for key in ("n", "omega", "phi"):
                assert not np.any(np.isnan(data[key])), f"NaN in {f.name}/{key}"

    def test_metadata_embedded_in_npz(self, tiny_dataset: GenerationResult):
        npz_files = sorted(tiny_dataset.output_dir.glob("traj_*.npz"))
        data = dict(np.load(str(npz_files[0]), allow_pickle=False))
        assert "metadata" in data
        meta = json.loads(str(data["metadata"]))
        assert "alpha" in meta
        assert "split" in meta

    def test_reproducibility(self, tmp_path: Path):
        """Two runs with the same seed produce identical parameters."""
        cfg1 = DatasetConfig(
            nx=16, ny=16, lx=20.0, ly=20.0,
            dt=0.05, n_steps=20, save_every=10,
            n_train=2, n_val=1, n_test=1, n_ood=1,
            seed=99,
            output_dir=str(tmp_path / "run1"),
        )
        cfg2 = DatasetConfig(
            nx=16, ny=16, lx=20.0, ly=20.0,
            dt=0.05, n_steps=20, save_every=10,
            n_train=2, n_val=1, n_test=1, n_ood=1,
            seed=99,
            output_dir=str(tmp_path / "run2"),
        )
        r1 = generate_dataset(cfg1)
        r2 = generate_dataset(cfg2)

        for e1, e2 in zip(r1.manifest, r2.manifest, strict=True):
            assert e1["alpha"] == e2["alpha"]
            assert e1["kappa"] == e2["kappa"]
            assert e1["split"] == e2["split"]
            assert e1["ic_seed"] == e2["ic_seed"]

        # Field values should be identical
        f1 = sorted((tmp_path / "run1").glob("traj_*.npz"))
        f2 = sorted((tmp_path / "run2").glob("traj_*.npz"))
        for a, b in zip(f1, f2, strict=True):
            d1 = np.load(str(a))
            d2 = np.load(str(b))
            np.testing.assert_array_equal(d1["n"], d2["n"])

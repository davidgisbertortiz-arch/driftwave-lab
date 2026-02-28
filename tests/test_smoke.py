"""Smoke tests: verify the package is importable and basic structure is sound."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------

class TestPackageImport:
    """Verify that driftwave_lab and all subpackages can be imported."""

    @pytest.mark.parametrize(
        "module",
        [
            "driftwave_lab",
            "driftwave_lab.solver",
            "driftwave_lab.data",
            "driftwave_lab.models",
            "driftwave_lab.training",
            "driftwave_lab.evaluation",
            "driftwave_lab.viz",
            "driftwave_lab.utils",
        ],
    )
    def test_import_subpackage(self, module: str) -> None:
        mod = importlib.import_module(module)
        assert mod is not None

    def test_version_string(self) -> None:
        import driftwave_lab

        assert isinstance(driftwave_lab.__version__, str)
        assert driftwave_lab.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# Placeholder module import tests
# ---------------------------------------------------------------------------

class TestPlaceholderModules:
    """Ensure every placeholder module file is importable without errors."""

    @pytest.mark.parametrize(
        "module",
        [
            "driftwave_lab.solver.hw",
            "driftwave_lab.solver.spectral",
            "driftwave_lab.solver.diagnostics",
            "driftwave_lab.solver.initial_conditions",
            "driftwave_lab.data.generator",
            "driftwave_lab.data.dataset",
            "driftwave_lab.data.io",
            "driftwave_lab.models.unet",
            "driftwave_lab.models.fno2d",
            "driftwave_lab.models.pinn_inverse",
            "driftwave_lab.training.train_baseline",
            "driftwave_lab.training.train_fno",
            "driftwave_lab.training.train_pinn",
            "driftwave_lab.evaluation.rollout",
            "driftwave_lab.evaluation.metrics",
            "driftwave_lab.evaluation.benchmark",
            "driftwave_lab.evaluation.spectra",
            "driftwave_lab.viz.plots",
            "driftwave_lab.viz.gifs",
            "driftwave_lab.viz.readme_assets",
            "driftwave_lab.utils.config",
        ],
    )
    def test_import_module(self, module: str) -> None:
        mod = importlib.import_module(module)
        assert mod is not None


# ---------------------------------------------------------------------------
# Placeholder script tests
# ---------------------------------------------------------------------------

class TestPlaceholderScripts:
    """Verify placeholder scripts run without error (exit 0) and print a message."""

    @pytest.mark.parametrize(
        "script",
        [
            # run_solver.py is now a real script — tested in test_solver.py
            # generate_dataset.py is now a real script — tested in test_generator.py
            "train.py",
            "rollout_demo.py",
            "benchmark.py",
            "make_readme_assets.py",
        ],
    )
    def test_script_runs_cleanly(self, script: str) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"{script} exited with {result.returncode}: {result.stderr}"
        assert len(result.stdout) > 0, f"{script} produced no output"

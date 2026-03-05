"""Tests for viz and evaluation.spectra modules.

Lightweight tests that verify function signatures, return types, and
basic behaviour without running the full solver or requiring ML checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# evaluation.spectra
# ---------------------------------------------------------------------------


class TestSpectra:
    """Tests for evaluation.spectra helper functions."""

    def test_spectrum_from_field_shape(self) -> None:
        from driftwave_lab.evaluation.spectra import spectrum_from_field
        from driftwave_lab.solver.spectral import SpectralGrid

        grid = SpectralGrid(32, 32, 2 * np.pi, 2 * np.pi)
        field = np.random.default_rng(0).standard_normal((32, 32))
        k, spectrum = spectrum_from_field(field, grid)
        assert k.ndim == 1
        assert spectrum.ndim == 1
        assert len(k) == len(spectrum)
        assert np.all(spectrum >= 0)

    def test_compare_spectra_keys(self) -> None:
        from driftwave_lab.evaluation.spectra import compare_spectra
        from driftwave_lab.solver.spectral import SpectralGrid

        grid = SpectralGrid(32, 32, 2 * np.pi, 2 * np.pi)
        rng = np.random.default_rng(42)
        fields = {
            "solver": rng.standard_normal((32, 32)),
            "fno": rng.standard_normal((32, 32)),
        }
        result = compare_spectra(fields, grid)
        assert set(result.keys()) == {"solver", "fno"}
        for k, spec in result.values():
            assert k.ndim == 1
            assert spec.ndim == 1

    def test_spectral_mismatch_is_nonnegative(self) -> None:
        from driftwave_lab.evaluation.spectra import spectral_mismatch
        from driftwave_lab.solver.spectral import SpectralGrid

        grid = SpectralGrid(32, 32, 2 * np.pi, 2 * np.pi)
        rng = np.random.default_rng(7)
        a = rng.standard_normal((32, 32))
        b = rng.standard_normal((32, 32))
        err = spectral_mismatch(a, b, grid)
        assert err >= 0.0

    def test_spectral_mismatch_identical_is_zero(self) -> None:
        from driftwave_lab.evaluation.spectra import spectral_mismatch
        from driftwave_lab.solver.spectral import SpectralGrid

        grid = SpectralGrid(32, 32, 2 * np.pi, 2 * np.pi)
        field = np.random.default_rng(3).standard_normal((32, 32))
        err = spectral_mismatch(field, field, grid)
        assert err == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# viz.plots — verify figures are returned without crashing
# ---------------------------------------------------------------------------


class TestPlots:
    """Tests for viz.plots — no display, just verify Figure creation."""

    @pytest.fixture(autouse=True)
    def _use_agg(self) -> None:
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_spectra_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from driftwave_lab.viz.plots import plot_spectra

        spectra = {
            "solver": (np.arange(1, 17, dtype=float), np.logspace(0, -3, 16)),
            "fno": (np.arange(1, 17, dtype=float), np.logspace(0, -3, 16) * 0.9),
        }
        fig = plot_spectra(spectra)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_benchmark_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from driftwave_lab.viz.plots import plot_benchmark

        fig = plot_benchmark(
            names=["Solver", "FNO", "UNet"],
            one_step_ms=[10.0, 0.5, 1.2],
            n_params=[0, 500_000, 800_000],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_rollout_error_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from driftwave_lab.viz.plots import plot_rollout_error

        fig = plot_rollout_error(
            mse_values=np.linspace(0, 0.1, 20),
            rel_l2_values=np.linspace(0, 0.5, 20),
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison_panel_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from driftwave_lab.viz.plots import plot_comparison_panel

        rng = np.random.default_rng(0)
        truth = rng.standard_normal((32, 32))
        pred = truth + 0.01 * rng.standard_normal((32, 32))
        fig = plot_comparison_panel(truth, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_field_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from driftwave_lab.viz.plots import plot_field

        field = np.random.default_rng(1).standard_normal((32, 32))
        fig = plot_field(field, title="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# viz.gifs — test frame rendering helper
# ---------------------------------------------------------------------------


class TestGifs:
    """Test GIF utility helpers."""

    @pytest.fixture(autouse=True)
    def _use_agg(self) -> None:
        import matplotlib

        matplotlib.use("Agg")

    def test_render_frame_to_array_shape(self) -> None:
        import matplotlib.pyplot as plt

        from driftwave_lab.viz.gifs import _render_frame_to_array

        fig, ax = plt.subplots(figsize=(3, 3), dpi=50)
        ax.plot([0, 1], [0, 1])
        arr = _render_frame_to_array(fig)
        assert arr.ndim == 3  # H, W, C
        assert arr.shape[2] == 3  # RGB
        assert arr.dtype == np.uint8
        plt.close(fig)

    def test_make_hero_gif_creates_file(self, tmp_path) -> None:
        from driftwave_lab.viz.gifs import make_hero_gif

        rng = np.random.default_rng(0)
        n_frames = 4
        truth = [rng.standard_normal((16, 16)) for _ in range(n_frames)]
        pred = [rng.standard_normal((16, 16)) for _ in range(n_frames)]
        out = tmp_path / "hero.gif"
        make_hero_gif(truth, pred, out, fps=2)
        assert out.exists()
        assert out.stat().st_size > 0

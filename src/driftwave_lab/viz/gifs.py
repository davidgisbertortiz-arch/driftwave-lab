"""Headless GIF generation for README assets.

Uses matplotlib to render frames and PIL to assemble them into
optimised GIFs.  Falls back to matplotlib.animation if PIL is not
available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray


def _render_frame_to_array(fig: plt.Figure) -> NDArray:
    """Render a matplotlib figure to a uint8 RGB array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()  # drop alpha


def _symmetric_vlim(fields: list[NDArray]) -> float:
    """Global symmetric vmin/vmax across all frames."""
    return float(max(np.max(np.abs(f)) for f in fields))


# ---------------------------------------------------------------------------
# Hero GIF: side-by-side truth vs prediction
# ---------------------------------------------------------------------------


def make_hero_gif(
    truth_frames: list[NDArray],
    pred_frames: list[NDArray],
    out_path: str | Path,
    *,
    field_name: str = "n",
    fps: int = 8,
    dpi: int = 100,
) -> Path:
    """Create a side-by-side truth|prediction GIF.

    Parameters
    ----------
    truth_frames, pred_frames : list of 2D arrays
        Matching sequences of spatial fields (nx, ny).
    out_path : path
        Where to save the GIF.
    field_name : str
        Label for colourbar.
    fps : int
        Frames per second.
    dpi : int
        Resolution.

    Returns
    -------
    Path to the saved GIF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = min(len(truth_frames), len(pred_frames))
    vmax = max(_symmetric_vlim(truth_frames[:n_frames]), _symmetric_vlim(pred_frames[:n_frames]))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    pil_frames: list[Any] = []
    for i in range(n_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=dpi)

        im1 = ax1.imshow(
            truth_frames[i].T, origin="lower", cmap="RdBu_r", norm=norm, aspect="equal"
        )
        ax1.set_title(f"Solver ({field_name})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2.imshow(pred_frames[i].T, origin="lower", cmap="RdBu_r", norm=norm, aspect="equal")
        ax2.set_title(f"FNO rollout ({field_name})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        fig.colorbar(im1, ax=[ax1, ax2], shrink=0.85, pad=0.02, label=field_name)
        fig.suptitle(f"Step {i}", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])

        rgb = _render_frame_to_array(fig)
        plt.close(fig)
        pil_frames.append(rgb)

    _save_gif(pil_frames, out_path, fps=fps)
    return out_path


# ---------------------------------------------------------------------------
# Error GIF: animated error field
# ---------------------------------------------------------------------------


def make_error_gif(
    truth_frames: list[NDArray],
    pred_frames: list[NDArray],
    out_path: str | Path,
    *,
    field_name: str = "n",
    fps: int = 8,
    dpi: int = 100,
) -> Path:
    """Animated absolute-error field over rollout time."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = min(len(truth_frames), len(pred_frames))
    errors = [np.abs(pred_frames[i] - truth_frames[i]) for i in range(n_frames)]
    emax = float(max(np.max(e) for e in errors))
    emax = max(emax, 1e-10)

    pil_frames: list[Any] = []
    for i in range(n_frames):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=dpi)
        im = ax.imshow(
            errors[i].T, origin="lower", cmap="inferno", vmin=0, vmax=emax, aspect="equal"
        )
        ax.set_title(f"|Error| step {i}  (max={np.max(errors[i]):.2e})", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=f"|{field_name}_err|")
        fig.tight_layout()

        rgb = _render_frame_to_array(fig)
        plt.close(fig)
        pil_frames.append(rgb)

    _save_gif(pil_frames, out_path, fps=fps)
    return out_path


# ---------------------------------------------------------------------------
# GIF writer (PIL or fallback)
# ---------------------------------------------------------------------------


def _save_gif(frames: list[NDArray], path: Path, *, fps: int = 8) -> None:
    """Save a list of RGB uint8 arrays as an optimised GIF."""
    try:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            str(path),
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
    except ImportError:
        # Fallback: use matplotlib animation
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100))
        ax.axis("off")
        im = ax.imshow(frames[0])

        def update(i: int) -> list:
            im.set_data(frames[i])
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
        ani.save(str(path), writer="pillow", fps=fps)
        plt.close(fig)

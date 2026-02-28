"""PyTorch-compatible dataset wrappers for HW trajectory data.

Provides :class:`HWNextStepDataset` — an iterable-style ``torch.utils.data.Dataset``
that yields ``(input, target)`` tensor pairs for **next-step prediction**:

* ``input``  — ``[C_in, H, W]`` tensor  (channels: n_t, phi_t)
* ``target`` — ``[C_out, H, W]`` tensor (channels: n_{t+1}, phi_{t+1})

The dataset reads from a manifest file produced by :func:`generate_dataset`
and loads trajectories lazily (or eagerly if requested).  Split selection is
done at construction time via the ``split`` argument.

Requires **PyTorch** (``pip install torch``).  Import will raise a clear
``ImportError`` if torch is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    _TorchDataset = object  # type: ignore[assignment,misc]


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for driftwave_lab.data.dataset.  "
            "Install it with:  pip install torch"
        )


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a ``manifest.json`` and return its contents."""
    with open(manifest_path) as f:
        return json.load(f)


def get_split_files(
    manifest: dict[str, Any],
    split: str,
    *,
    data_dir: str | Path | None = None,
) -> list[Path]:
    """Return file paths for trajectories belonging to *split*.

    Parameters
    ----------
    manifest : dict
        Parsed ``manifest.json``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``, ``"ood"``.
    data_dir : Path, optional
        Base directory containing NPZ files.  If *None*, uses the
        directory that contains the manifest file.

    Returns
    -------
    list[Path]
        Sorted list of NPZ file paths for the requested split.
    """
    base = Path(data_dir) if data_dir is not None else Path(".")
    files = sorted(
        base / entry["file"]
        for entry in manifest["trajectories"]
        if entry["split"] == split
    )
    return files


def get_split_metadata(
    manifest: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Return metadata entries for trajectories belonging to *split*."""
    return [
        entry for entry in manifest["trajectories"]
        if entry["split"] == split
    ]


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------

def _load_fields(path: Path) -> tuple[NDArray, NDArray]:
    """Load n and phi arrays from an NPZ file.

    Returns
    -------
    n : NDArray  shape (T, H, W)
    phi : NDArray  shape (T, H, W)
    """
    data = np.load(str(path), allow_pickle=False)
    return data["n"], data["phi"]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class HWNextStepDataset(_TorchDataset):
    """Map-style PyTorch dataset for next-step prediction on HW fields.

    Each sample is a consecutive ``(input, target)`` pair drawn from one
    of the trajectories assigned to the requested ``split``.

    Channels
    --------
    * ``input``  shape ``[2, H, W]``  — ``(n_t, phi_t)``
    * ``target`` shape ``[2, H, W]``  — ``(n_{t+1}, phi_{t+1})``

    Parameters
    ----------
    manifest_path : str or Path
        Path to ``manifest.json`` produced by the generator.
    split : str
        ``"train"``, ``"val"``, ``"test"`` or ``"ood"``.
    channels : tuple[str, ...]
        Which fields to stack as channels; default ``("n", "phi")``.
    preload : bool
        If *True*, load all trajectories into memory at init time.
        If *False* (default), load lazily and cache.
    dtype : str
        Numpy dtype for stored arrays (default ``"float32"``).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        *,
        channels: tuple[str, ...] = ("n", "phi"),
        preload: bool = False,
        dtype: str = "float32",
    ) -> None:
        _require_torch()

        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.split = split
        self.channels = channels
        self.dtype = np.dtype(dtype)

        self.manifest = load_manifest(self.manifest_path)
        self.files = get_split_files(
            self.manifest, split, data_dir=self.data_dir
        )

        if len(self.files) == 0:
            raise ValueError(
                f"No trajectories found for split '{split}' in {self.manifest_path}"
            )

        # Build an index mapping: global_idx → (traj_idx, time_step)
        # Each trajectory has T snapshots → T-1 consecutive pairs.
        self._traj_cache: dict[int, dict[str, NDArray]] = {}
        self._index_map: list[tuple[int, int]] = []
        self._traj_lengths: list[int] = []

        for traj_idx, fpath in enumerate(self.files):
            if preload:
                self._cache_trajectory(traj_idx)
                n_steps = self._traj_cache[traj_idx]["n"].shape[0]
            else:
                # Peek at the file to get length without loading all data
                with np.load(str(fpath), allow_pickle=False) as data:
                    n_steps = data["n"].shape[0]

            self._traj_lengths.append(n_steps)
            for t in range(n_steps - 1):
                self._index_map.append((traj_idx, t))

    # ------------------------------------------------------------------

    def _cache_trajectory(self, traj_idx: int) -> dict[str, NDArray]:
        """Load a trajectory into the cache if not already there."""
        if traj_idx not in self._traj_cache:
            fpath = self.files[traj_idx]
            data = dict(np.load(str(fpath), allow_pickle=False))
            # Cast to target dtype
            cached: dict[str, NDArray] = {}
            for key in ("n", "omega", "phi"):
                if key in data:
                    cached[key] = data[key].astype(self.dtype, copy=False)
            self._traj_cache[traj_idx] = cached
        return self._traj_cache[traj_idx]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        traj_idx, t = self._index_map[idx]
        data = self._cache_trajectory(traj_idx)

        # Stack requested channels → (C, H, W)
        inp_arrays = [data[ch][t] for ch in self.channels]
        tgt_arrays = [data[ch][t + 1] for ch in self.channels]

        inp = np.stack(inp_arrays, axis=0)  # (C, H, W)
        tgt = np.stack(tgt_arrays, axis=0)

        return torch.from_numpy(inp), torch.from_numpy(tgt)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_trajectories(self) -> int:
        return len(self.files)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """(H, W) of the fields in this dataset."""
        data = self._cache_trajectory(0)
        return data["n"].shape[1], data["n"].shape[2]

    def __repr__(self) -> str:
        return (
            f"HWNextStepDataset(split='{self.split}', "
            f"n_traj={self.n_trajectories}, "
            f"n_samples={len(self)}, "
            f"channels={self.channels})"
        )

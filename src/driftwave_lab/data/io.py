"""I/O helpers for NPZ trajectory storage.

Provides ``save_trajectory`` / ``load_trajectory`` for the solver output
format, including embedded metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def save_trajectory(
    path: str | Path,
    *,
    times: NDArray,
    n: NDArray,
    omega: NDArray,
    phi: NDArray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a solver trajectory to a compressed ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Destination file (will get ``.npz`` suffix if missing).
    times, n, omega, phi : NDArray
        Time array and field snapshots (T, nx, ny).
    metadata : dict, optional
        Scalar metadata (parameters, seed, etc.) stored as a JSON string
        inside the archive under the key ``"metadata"``.

    Returns
    -------
    Path to the written file.
    """
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "times": times,
        "n": n,
        "omega": omega,
        "phi": phi,
    }
    if metadata is not None:
        save_dict["metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(str(out), **save_dict)
    # np.savez_compressed auto-appends .npz if not present
    actual = out if out.suffix == ".npz" else out.with_suffix(out.suffix + ".npz")
    if not actual.exists():
        actual = Path(str(out) + ".npz")
    return actual


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load a trajectory ``.npz`` and return arrays + metadata.

    Returns
    -------
    dict with keys ``times``, ``n``, ``omega``, ``phi``, and optionally
    ``metadata`` (parsed back to a Python dict).
    """
    import json

    data = dict(np.load(str(path), allow_pickle=False))
    if "metadata" in data:
        data["metadata"] = json.loads(str(data["metadata"]))
    return data

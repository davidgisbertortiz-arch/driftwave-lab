"""Field-error, physics-aware, and performance metrics.

All functions operate on plain tensors / arrays so they can be used
for any model without coupling to a specific architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Field error metrics
# ---------------------------------------------------------------------------


def mse(pred: Tensor, true: Tensor) -> Tensor:
    """Mean squared error (scalar)."""
    return ((pred - true) ** 2).mean()


def rmse(pred: Tensor, true: Tensor) -> Tensor:
    """Root mean squared error (scalar)."""
    return mse(pred, true).sqrt()


def relative_l2(pred: Tensor, true: Tensor) -> Tensor:
    """Relative L2 error: ||pred - true||_2 / ||true||_2 (scalar)."""
    diff_norm = torch.norm(pred - true)
    true_norm = torch.norm(true)
    return diff_norm / true_norm.clamp(min=1e-12)


def channel_mse(pred: Tensor, true: Tensor) -> Tensor:
    """Per-channel MSE.  Shape: (C,) given (B, C, H, W) inputs."""
    return ((pred - true) ** 2).mean(dim=(0, 2, 3))


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def rollout_errors(
    preds: list[Tensor],
    trues: list[Tensor],
) -> dict[str, list[float]]:
    """Compute per-step MSE and relative-L2 across a rollout.

    Parameters
    ----------
    preds, trues : list[Tensor]
        Each element is a ``(B, C, H, W)`` or ``(C, H, W)`` tensor.

    Returns
    -------
    dict with ``"mse"`` and ``"rel_l2"`` lists of floats (one per step).
    """
    mse_list: list[float] = []
    rel_list: list[float] = []
    for p, t in zip(preds, trues, strict=True):
        mse_list.append(mse(p, t).item())
        rel_list.append(relative_l2(p, t).item())
    return {"mse": mse_list, "rel_l2": rel_list}

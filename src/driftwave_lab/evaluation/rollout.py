"""Autoregressive rollout evaluation.

Given a trained one-step model and a ground-truth trajectory, produce an
autoregressive prediction sequence and compute error metrics at each step.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from driftwave_lab.evaluation.metrics import rollout_errors


@torch.no_grad()
def autoregressive_rollout(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    *,
    device: torch.device | str = "cpu",
) -> list[Tensor]:
    """Run an autoregressive rollout starting from *x0*.

    Parameters
    ----------
    model : nn.Module
        Trained one-step predictor: ``(B, C, H, W) -> (B, C, H, W)``.
    x0 : Tensor  (B, C, H, W) or (C, H, W)
        Initial state.
    n_steps : int
        Number of autoregressive steps.
    device : device
        Computation device.

    Returns
    -------
    list[Tensor]
        ``n_steps + 1`` tensors (including x0) on CPU.
    """
    model.eval()
    x = x0.unsqueeze(0).to(device) if x0.dim() == 3 else x0.to(device)
    preds = [x.cpu()]
    for _ in range(n_steps):
        x = model(x)
        preds.append(x.cpu())
    return preds


def evaluate_rollout(
    model: torch.nn.Module,
    truth: list[Tensor],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Compare an autoregressive rollout against ground truth.

    Parameters
    ----------
    model : nn.Module
        Trained one-step predictor.
    truth : list[Tensor]
        Ground-truth frames ``[x_0, x_1, ..., x_T]``.
    device : device
        Computation device.

    Returns
    -------
    dict with keys ``"preds"``, ``"mse"``, ``"rel_l2"``.
    """
    x0 = truth[0]
    n_steps = len(truth) - 1
    preds = autoregressive_rollout(model, x0, n_steps, device=device)
    errors = rollout_errors(preds[1:], truth[1:])
    return {"preds": preds, **errors}

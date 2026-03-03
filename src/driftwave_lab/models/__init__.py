"""ML surrogate models (U-Net, FNO, PINN inverse).

Factory function :func:`build_model` creates a model from a config dict.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Instantiate a model from a config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``"name"`` (``"unet"`` or ``"fno2d"``).
        Remaining keys are forwarded as constructor kwargs.

    Returns
    -------
    nn.Module
    """
    name = cfg["name"].lower()
    if name == "unet":
        from driftwave_lab.models.unet import UNet

        return UNet(
            in_channels=cfg.get("in_channels", 2),
            out_channels=cfg.get("out_channels", 2),
            base_filters=cfg.get("base_filters", 32),
        )
    if name in ("fno2d", "fno"):
        from driftwave_lab.models.fno2d import FNO2d

        return FNO2d(
            in_channels=cfg.get("in_channels", 2),
            out_channels=cfg.get("out_channels", 2),
            modes=cfg.get("modes", 12),
            width=cfg.get("width", 32),
            n_layers=cfg.get("n_layers", 4),
        )
    raise ValueError(f"Unknown model name: {name!r}. Choose 'unet' or 'fno2d'.")

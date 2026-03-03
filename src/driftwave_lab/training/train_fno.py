"""Unified training loop for ML surrogates (FNO / U-Net).

Example
-------
>>> from driftwave_lab.training.train_fno import train
>>> history = train("configs/train_fno.yaml")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from driftwave_lab.data.dataset import HWNextStepDataset
from driftwave_lab.evaluation.metrics import mse, relative_l2
from driftwave_lab.models import build_model


# ------------------------------------------------------------------
# Training  helpers
# ------------------------------------------------------------------

def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3))
    return None


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    """Run one training or validation epoch.

    If *optimizer* is ``None`` the epoch runs in eval mode (no grad).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_mse = 0.0
    total_rel = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = torch.nn.functional.mse_loss(pred, tgt)

            if is_train:
                optimizer.zero_grad()  # type: ignore[union-attr]
                loss.backward()
                optimizer.step()  # type: ignore[union-attr]

            total_mse += mse(pred, tgt).item()
            total_rel += relative_l2(pred, tgt).item()
            n_batches += 1

    return {
        "mse": total_mse / max(n_batches, 1),
        "rel_l2": total_rel / max(n_batches, 1),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def train(
    cfg: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a surrogate model according to *cfg* and return history.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (see ``configs/train_fno.yaml``).
    device : torch.device or str, optional
        Override device; defaults to CUDA if available.
    verbose : bool
        Print epoch logs.

    Returns
    -------
    dict with keys ``"history"``, ``"model"``, ``"checkpoint_path"``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- data ---
    data_cfg = cfg["data"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    manifest_path = dataset_dir / "manifest.json"

    train_ds = HWNextStepDataset(manifest_path, split="train", preload=True)
    val_ds = HWNextStepDataset(manifest_path, split="val", preload=True)

    tr_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg.get("batch_size", 8),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg.get("batch_size", 8),
        shuffle=False,
        drop_last=False,
    )

    # --- model ---
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {cfg['model']['name']}  params: {n_params:,}  device: {device}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tr_cfg.get("lr", 1e-3),
        weight_decay=tr_cfg.get("weight_decay", 0.0),
    )
    scheduler = _make_scheduler(
        optimizer,
        tr_cfg.get("scheduler", "cosine"),
        tr_cfg.get("epochs", 50),
    )

    # --- training loop ---
    history: dict[str, list[float]] = {
        "train_mse": [],
        "train_rel_l2": [],
        "val_mse": [],
        "val_rel_l2": [],
    }
    best_val = float("inf")
    epochs = tr_cfg.get("epochs", 50)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['model']['name']}_best.pt"

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_metrics = _run_epoch(model, train_loader, optimizer, device)
        v_metrics = _run_epoch(model, val_loader, None, device)

        if scheduler is not None:
            scheduler.step()

        for k in ("mse", "rel_l2"):
            history[f"train_{k}"].append(t_metrics[k])
            history[f"val_{k}"].append(v_metrics[k])

        if v_metrics["mse"] < best_val:
            best_val = v_metrics["mse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mse": best_val,
                    "config": cfg,
                },
                ckpt_path,
            )

        if verbose and (epoch <= 5 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train_mse={t_metrics['mse']:.4e}  "
                f"val_mse={v_metrics['mse']:.4e}  "
                f"val_rel={v_metrics['rel_l2']:.4f}"
            )

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"Training complete in {elapsed:.1f}s  best_val_mse={best_val:.4e}")

    return {
        "history": history,
        "model": model,
        "checkpoint_path": str(ckpt_path),
        "elapsed_s": elapsed,
    }

"""Runtime benchmark: solver vs surrogates.

Measures one-step and multi-step inference latencies for ML models and
compares them against the HW spectral solver.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BenchmarkResult:
    """Timing result for a single model / method."""

    name: str
    one_step_ms: float
    rollout_ms: float
    rollout_steps: int
    n_params: int | None = None
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "one_step_ms": round(self.one_step_ms, 3),
            "rollout_ms": round(self.rollout_ms, 3),
            "rollout_steps": self.rollout_steps,
            "n_params": self.n_params,
            "device": self.device,
            **self.extra,
        }


def benchmark_model(
    model: torch.nn.Module,
    sample: torch.Tensor,
    *,
    rollout_steps: int = 20,
    warmup: int = 3,
    repeats: int = 5,
    device: torch.device | str = "cpu",
) -> BenchmarkResult:
    """Time a PyTorch model for one-step and multi-step inference.

    Returns
    -------
    BenchmarkResult
    """
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    x = sample.to(dev)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    n_params = sum(p.numel() for p in model.parameters())

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    # One-step timing
    times_1: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times_1.append((time.perf_counter() - t0) * 1000)

    # Rollout timing
    times_r: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            cur = x.clone()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(rollout_steps):
                cur = model(cur)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times_r.append((time.perf_counter() - t0) * 1000)

    return BenchmarkResult(
        name=model.__class__.__name__,
        one_step_ms=min(times_1),
        rollout_ms=min(times_r),
        rollout_steps=rollout_steps,
        n_params=n_params,
        device=str(dev),
    )


def system_info() -> dict[str, str]:
    """Capture basic system / hardware metadata."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info

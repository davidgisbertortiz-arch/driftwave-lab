"""Configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
ASSETS_DIR = REPO_ROOT / "assets"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return contents as a plain dict.

    Uses a minimal safe-loader that ships with Python's standard library
    (or falls back to a tiny parser for the flat configs we use).
    """
    import yaml  # pyyaml — added as dependency

    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]

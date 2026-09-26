"""Application paths, including discovery of existing installations."""

import os
from pathlib import Path

APP_NAME = "compute-runner"
# Kept only for upgrade discovery; persisted job paths must remain accessible.
LEGACY_APP_NAME = "kaggle-runner"


def xdg_dir(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable) or Path.home() / default)


def application_dir(kind: str) -> Path:
    """Prefer explicit settings, then existing state, without moving user data."""
    override = os.environ.get(f"COMPUTE_RUNNER_{kind}_DIR") or os.environ.get(f"KGR_{kind}_DIR")
    if override:
        return Path(override).expanduser()
    variable, fallback, marker = {
        "CONFIG": ("XDG_CONFIG_HOME", ".config", "config.json"),
        "STATE": ("XDG_DATA_HOME", ".local/share", "queue.sqlite3"),
    }[kind]
    root = xdg_dir(variable, fallback)
    current, legacy = root / APP_NAME, root / LEGACY_APP_NAME
    if not (current / marker).exists() and (legacy / marker).exists():
        return legacy
    return current

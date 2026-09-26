"""Application paths, including discovery of existing installations."""

import os
import re
import time
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


def experiment_folder(name: str) -> str:
    """A job name as a folder name: readable, portable and never empty or hidden."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:80] or "workload"


def run_folder(number: int, created: float) -> str:
    """NNN_YYYY-MM-DD_HH-MM-SS: sorts by run number and shows when it was submitted (local time)."""
    return f"{number:03d}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(created))}"


def run_number(folder: str) -> int | None:
    match = re.match(r"(\d+)_", folder)
    return int(match.group(1)) if match else None

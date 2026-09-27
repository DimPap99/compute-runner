"""Linux user service management. No system-wide changes or lingering configuration."""

import subprocess
import sys

from .paths import LEGACY_APP_NAME, xdg_dir
from .store import config_path

UNIT_NAME = "compute-runner.service"
LEGACY_UNIT_NAME = f"{LEGACY_APP_NAME}.service"
DESCRIPTION = "Description=Persistent compute workload queue"
LEGACY_DESCRIPTION = "Description=Persistent Kaggle workload queue"


def _quote(value):
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines are not supported in service paths")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def unit_text(config, python=None):
    executable = python or sys.executable
    return f"""[Unit]
{DESCRIPTION}
After=network-online.target

[Service]
Type=simple
ExecStart={_quote(executable)} -m compute_runner --state-dir {_quote(config.state_dir)} worker run
WorkingDirectory={str(config.state_dir).replace("%", "%%")}
Environment={_quote("COMPUTE_RUNNER_CONFIG_DIR=" + str(config_path().parent))}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=15
TimeoutStopSec=120
UMask=0077

[Install]
WantedBy=default.target
"""


def _systemctl(*arguments, check=True):
    """Run systemctl --user; without a user service manager, say how to run the worker instead."""
    try:
        return subprocess.run(["systemctl", "--user", *arguments], check=check)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"systemd user services are unavailable here (systemctl --user {' '.join(arguments)} failed: "
            f"{error}). Run the worker in a terminal instead: compute-runner worker run"
        ) from error


def control(action):
    if action not in {"start", "stop", "restart", "status"}:
        raise ValueError("Invalid service action")
    root = xdg_dir("XDG_CONFIG_HOME", ".config") / "systemd/user"
    unit = UNIT_NAME
    legacy = root / LEGACY_UNIT_NAME
    if not (root / UNIT_NAME).exists() and legacy.is_file() and LEGACY_DESCRIPTION in legacy.read_text():
        # The previous unit runs a package that no longer exists; it can only be inspected or stopped.
        if action in {"start", "restart"}:
            raise ValueError("The installed service predates the rename; run: compute-runner service install")
        unit = LEGACY_UNIT_NAME
    return _systemctl(action, unit, "--no-pager", check=action != "status")


def install(config, *, start=True):
    root = xdg_dir("XDG_CONFIG_HOME", ".config") / "systemd/user"
    root.mkdir(parents=True, exist_ok=True)
    path = root / UNIT_NAME
    text = unit_text(config)
    if path.exists() and DESCRIPTION not in path.read_text():
        raise ValueError(f"Refusing to replace an unrelated service: {path}")
    path.write_text(text)
    legacy = root / LEGACY_UNIT_NAME
    if legacy.is_file() and LEGACY_DESCRIPTION in legacy.read_text():
        _systemctl("disable", "--now", LEGACY_UNIT_NAME)
    _systemctl("daemon-reload")
    _systemctl("enable", *(["--now"] if start else []), UNIT_NAME)
    return path

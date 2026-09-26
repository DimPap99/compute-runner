"""Linux user service management. No system-wide changes or lingering configuration."""

import subprocess
import sys

from .models import xdg_dir
from .store import config_path

UNIT_NAME = "kaggle-runner.service"


def _quote(value):
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines are not supported in service paths")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def unit_text(config, python=None):
    executable = python or sys.executable
    return f"""[Unit]
Description=Persistent Kaggle workload queue
After=network-online.target

[Service]
Type=simple
ExecStart={_quote(executable)} -m kaggle_runner --state-dir {_quote(config.state_dir)} worker run
WorkingDirectory={str(config.state_dir).replace("%", "%%")}
Environment={_quote("KGR_CONFIG_DIR=" + str(config_path().parent))}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=15
TimeoutStopSec=120
UMask=0077

[Install]
WantedBy=default.target
"""


def control(action):
    if action not in {"start", "stop", "restart", "status"}:
        raise ValueError("Invalid service action")
    return subprocess.run(["systemctl", "--user", action, UNIT_NAME, "--no-pager"], check=action != "status")


def install(config, *, start=True):
    root = xdg_dir("XDG_CONFIG_HOME", ".config") / "systemd/user"
    root.mkdir(parents=True, exist_ok=True)
    path = root / UNIT_NAME
    text = unit_text(config)
    if path.exists() and "Description=Persistent Kaggle workload queue" not in path.read_text():
        raise ValueError(f"Refusing to replace an unrelated service: {path}")
    path.write_text(text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    command = ["systemctl", "--user", "enable"]
    if start:
        command.append("--now")
    subprocess.run([*command, UNIT_NAME], check=True)
    return path

"""Public workload and durable state models. No authentication on import."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .security import validate_nonsecret_env


def xdg_dir(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable) or Path.home() / default)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobSpec(Model):
    source: Path
    name: str = "workload"
    entrypoint: str | None = None
    module: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    gpu: bool = False
    accelerator: str | None = None
    internet: bool = False
    timeout_seconds: int = Field(default=43200, ge=1, le=43200)
    datasets: list[str] = Field(default_factory=list)
    inputs: dict[str, Path] = Field(default_factory=dict)
    requirements: str | None = None
    exclude: list[str] = Field(default_factory=list)
    auto_download: bool = True
    output_patterns: list[str] | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.entrypoint and self.module:
            raise ValueError("Choose entrypoint or module, not both")
        if self.module and not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", self.module):
            raise ValueError("module must be a Python dotted module name")
        if self.accelerator:
            if not self.accelerator.startswith("Nvidia"):
                raise ValueError("v1 accepts NVIDIA GPU accelerator IDs only")
            self.gpu = True
        if self.requirements and not self.internet:
            raise ValueError("requirements installation requires internet=True")
        for name in [*self.env, *self.inputs]:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid environment/input name: {name}")
        if len({name.upper() for name in self.inputs}) != len(self.inputs):
            raise ValueError("Input names must be unique ignoring case")
        if any(key.startswith("KGR_") for key in self.env):
            raise ValueError("KGR_ environment variables are reserved")
        validate_nonsecret_env(self.env)
        for ref in self.datasets:
            if not re.fullmatch(r"[\w-]+/[\w-]+(?:/[1-9]\d*)?", ref):
                raise ValueError(f"Invalid dataset reference: {ref}")
        return self


class Config(Model):
    owner: str = ""
    cpu_limit: int = Field(default=5, ge=0)
    gpu_limit: int = Field(default=1, ge=0)
    poll_seconds: float = Field(default=30, ge=1)
    discovery_seconds: float = Field(default=300, ge=1)
    retry_seconds: float = Field(default=60, ge=1)
    reconcile_seconds: float = Field(default=300, ge=1)
    # Broad log redaction and locked-down output downloads; see README "Strict mode".
    strict: bool = False
    state_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("KGR_STATE_DIR") or xdg_dir("XDG_DATA_HOME", ".local/share") / "kaggle-runner"
        )
    )

    @model_validator(mode="after")
    def validate_owner(self):
        if self.owner and not re.fullmatch(r"[a-zA-Z0-9_-]+", self.owner):
            raise ValueError("owner must be a Kaggle username")
        self.state_dir = self.state_dir.expanduser().resolve()
        return self


JobState = Literal[
    "queued",
    "preparing",
    "submitting",
    "remote_queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "blocked",
    "needs_attention",
]
TERMINAL = {"succeeded", "failed", "cancelled"}
ACTIVE = {"submitting", "remote_queued", "running", "needs_attention"}


class Attempt(Model):
    number: int
    ref: str
    state: Literal["submitting", "accepted", "rejected", "uncertain"] = "submitting"
    started_at: float = Field(default_factory=time.time)
    version: int | None = None
    error: str | None = None


class JobRecord(Model):
    id: str
    spec: JobSpec
    snapshot: dict
    owner: str
    state: JobState = "queued"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    attempts: list[Attempt] = Field(default_factory=list)
    remote_state: str | None = None
    wait_reason: str | None = None
    error: str | None = None
    next_action_at: float = 0
    last_polled_at: float | None = None
    finished_at: float | None = None
    download_state: Literal["pending", "downloading", "complete", "error", "disabled"] = "pending"
    download_error: str | None = None
    download_retry_at: float = 0
    download_failures: int = 0
    upload_refs: dict[str, str] = Field(default_factory=dict)
    result_dir: Path
    parent_id: str | None = None

    @property
    def remote_ref(self) -> str | None:
        """The latest attempt's notebook, unless Kaggle definitively rejected it."""
        return self.attempts[-1].ref if self.attempts and self.attempts[-1].state != "rejected" else None

    @property
    def url(self) -> str | None:
        return f"https://www.kaggle.com/code/{self.remote_ref}" if self.remote_ref else None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


class BatchRecord(Model):
    id: str
    created_at: float
    jobs: list[JobRecord]
    replayed: bool = False

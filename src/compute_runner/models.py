"""Public workload and durable state models. No authentication on import."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .security import validate_nonsecret_env
from .paths import application_dir


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
    timeout_seconds: int = Field(default=43200, ge=1)
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
        return self


class Account(Model):
    """One set of credentials on one provider. Provider-specific limits are checked by its adapter."""

    provider: Literal["kaggle"] = "kaggle"
    user: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    # A credentials file for this account only; None uses the provider's standard discovery.
    credentials: Path | None = None
    cpu_limit: int = Field(default=5, ge=0)
    gpu_limit: int = Field(default=1, ge=0)

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.user}"


class Config(Model):
    # In order of preference; the first is the default.
    accounts: list[Account] = Field(default_factory=list)
    # When a job cannot start on its account: never move it, suggest another account, or move it.
    failover: Literal["off", "ask", "auto"] = "ask"
    poll_seconds: float = Field(default=30, ge=1)
    discovery_seconds: float = Field(default=300, ge=1)
    retry_seconds: float = Field(default=60, ge=1)
    reconcile_seconds: float = Field(default=300, ge=1)
    # Broad log redaction and locked-down output downloads; see README "Strict mode".
    strict: bool = False
    state_dir: Path = Field(default_factory=lambda: application_dir("STATE"))

    @model_validator(mode="before")
    @classmethod
    def upgrade(cls, data):
        # Configurations saved before multiple accounts named a single Kaggle owner.
        if isinstance(data, dict) and "owner" in data:
            data = dict(data)
            legacy = {key: data.pop(key) for key in ("owner", "cpu_limit", "gpu_limit") if key in data}
            if legacy["owner"] and not data.get("accounts"):
                data["accounts"] = [{"user": legacy.pop("owner"), **legacy}]
        return data

    @model_validator(mode="after")
    def validate_accounts(self):
        ids = [account.id for account in self.accounts]
        if len({i.casefold() for i in ids}) != len(ids):
            raise ValueError("Accounts must be unique")
        self.state_dir = self.state_dir.expanduser().resolve()
        return self

    def account(self, account_id: str | None = None) -> Account:
        """The named account, or the default one."""
        if not self.accounts:
            raise ValueError("No account is configured; run: compute-runner account add kaggle USERNAME")
        if account_id is None:
            return self.accounts[0]
        for account in self.accounts:
            if account.id.casefold() == account_id.casefold():
                return account
        raise ValueError(
            f"Unknown account {account_id}; configured: {', '.join(a.id for a in self.accounts)}"
        )


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
    account: str
    ref: str
    url: str | None = None
    state: Literal["submitting", "accepted", "rejected", "uncertain"] = "submitting"
    started_at: float = Field(default_factory=time.time)
    version: int | None = None
    error: str | None = None


class JobRecord(Model):
    id: str
    spec: JobSpec
    snapshot: dict
    account: str
    state: JobState = "queued"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    attempts: list[Attempt] = Field(default_factory=list)
    remote_state: str | None = None
    wait_reason: str | None = None
    # Set while the job waits and another account could start it now (failover "ask").
    suggested_account: str | None = None
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

    @model_validator(mode="before")
    @classmethod
    def upgrade(cls, data):
        # Records saved before multiple accounts name a Kaggle owner instead of an account.
        if isinstance(data, dict) and "owner" in data:
            data = dict(data)
            account = "kaggle:" + data.pop("owner")
            data["account"] = account
            data["attempts"] = [
                {"account": account, "url": f"https://www.kaggle.com/code/{attempt['ref']}"} | attempt
                for attempt in data.get("attempts", [])
            ]
        return data

    @property
    def remote_ref(self) -> str | None:
        """The latest attempt's remote run, unless the provider definitively rejected it."""
        return self.attempts[-1].ref if self.attempts and self.attempts[-1].state != "rejected" else None

    @property
    def url(self) -> str | None:
        return self.attempts[-1].url if self.remote_ref else None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


class BatchRecord(Model):
    id: str
    created_at: float
    jobs: list[JobRecord]
    replayed: bool = False

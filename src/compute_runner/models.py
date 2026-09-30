"""Public workload and durable state models. No authentication on import."""

from __future__ import annotations

import re
import time
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .paths import application_dir
from .security import validate_nonsecret_env


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


PROVIDERS = ("kaggle", "ssh")
Pool = Literal["cpu", "gpu"]
# An input value naming data that is not a local path: another job's outputs, or a provider dataset.
_REFERENCE = re.compile(rf"^(job|{'|'.join(PROVIDERS)}):(.+)$")


# A path on one SSH machine: NAME:/PATH, or /PATH on the job's own machine until submission names it.
SSH_PATH = re.compile(r"^(?:(?P<machine>[A-Za-z0-9_-]+):)?(?P<path>/.*)$")


def input_reference(value) -> tuple[str, str] | None:
    """("job", "ID[/PATH]") or (provider, dataset reference); None for a local path.

    An ssh reference is an absolute path, optionally after its machine's name; the others are
    never absolute.
    """
    match = _REFERENCE.match(str(value))
    if not match:
        return None
    kind, rest = match.groups()
    if (SSH_PATH.match(rest) is None) if kind == "ssh" else rest.startswith("/"):
        return None
    return kind, rest


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
    # Passed to the workload as --NAME VALUE and KGR_PARAMS_JSON, and shown with its results.
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    # Parent of the experiment folders; None uses the configured folder or results/ beside the code.
    results_dir: Path | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.accelerator:
            self.gpu = True
        if self.requirements and not self.internet:
            raise ValueError("requirements installation requires internet=True")
        self._check_command()
        self._check_names()
        validate_nonsecret_env(self.env)
        # Parameters are shown in results folders and on the command line.
        validate_nonsecret_env({name: str(value) for name, value in self.params.items()}, label="parameter")
        return self

    def _check_command(self):
        if self.entrypoint and self.module:
            raise ValueError("Choose entrypoint or module, not both")
        if self.module and not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", self.module):
            raise ValueError("module must be a Python dotted module name")

    def _check_names(self):
        for name in [*self.env, *self.inputs]:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid environment/input name: {name}")
        if len({name.upper() for name in self.inputs}) != len(self.inputs):
            raise ValueError("Input names must be unique ignoring case")
        if any(key.startswith("KGR_") for key in self.env):
            raise ValueError("KGR_ environment variables are reserved")
        for name in self.params:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
                raise ValueError(f"Invalid parameter name: {name}")

    @property
    def pool(self) -> Pool:
        """The slots a run of this spec takes on its account."""
        return "gpu" if self.gpu else "cpu"

    def command_args(self) -> list[str]:
        """args, then each parameter as --NAME VALUE; true passes --NAME alone and false omits it."""
        result = list(self.args)
        for name, value in self.params.items():
            if value is True:
                result.append(f"--{name}")
            elif value is not False:
                result += [f"--{name}", str(value)]
        return result

    def dataset_inputs(self) -> dict[str, tuple[str, str]]:
        """Inputs that name a provider dataset: {alias: (provider, reference)}."""
        found = {alias: input_reference(value) for alias, value in self.inputs.items()}
        return {alias: ref for alias, ref in found.items() if ref and ref[0] != "job"}


class SshSettings(Model):
    """How to reach one machine over SSH. Its key or password is in the credentials file."""

    host: str = Field(min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(min_length=1)
    # Older configurations named a key or password file here; account add now keeps both in the
    # credentials file (see credentials.py), which takes precedence.
    key: Path | None = None
    password_file: Path | None = None
    # Where runs, bundles and virtual environments live on the machine; relative to the home folder.
    workdir: str = ".compute-runner"
    python: str = "python3"

    @field_validator("workdir")
    @classmethod
    def own_folder(cls, value):
        # A folder of its own below home, so the runner's files never mix with the user's.
        parts = PurePosixPath(value).parts
        if not parts or PurePosixPath(value).is_absolute() or any(part in {".", ".."} for part in parts):
            raise ValueError("workdir must be a folder below the home folder, such as .compute-runner")
        return PurePosixPath(value).as_posix()


class Account(Model):
    """One set of credentials on one provider. Provider-specific limits are checked by its adapter."""

    provider: Literal["kaggle", "ssh"] = "kaggle"
    # The account's name: the Kaggle username, or a name chosen for an SSH machine.
    user: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    # An older configuration's credentials file for this account; account add now saves secrets in
    # the credentials file (see credentials.py), which takes precedence. With neither, the
    # provider's standard discovery applies.
    credentials: Path | None = None
    cpu_limit: int = Field(default=5, ge=0)
    gpu_limit: int = Field(default=1, ge=0)
    ssh: SshSettings | None = None

    @model_validator(mode="before")
    @classmethod
    def ssh_defaults(cls, data):
        # A machine offers GPU slots only when the user says how many GPUs it has.
        if isinstance(data, dict) and data.get("provider") == "ssh" and data.get("gpu_limit") is None:
            data = {**data, "gpu_limit": 0}
        return data

    @model_validator(mode="after")
    def provider_settings(self):
        if (self.provider == "ssh") != (self.ssh is not None):
            raise ValueError("SSH accounts, and only they, need host settings")
        if self.provider == "ssh" and self.credentials is not None:
            raise ValueError("SSH accounts take a key or password file, not credentials")
        return self

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
    # Broad log redaction and locked-down output downloads; see docs/operations.md, "Strict mode".
    strict: bool = False
    # Parent of the experiment folders; None puts results/ beside each workload's code.
    results_dir: Path | None = None
    # Copy a dataset the job's account cannot read from an account that can; see docs/accounts.md,
    # "Datasets across accounts".
    transfer: bool = False
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
        if self.results_dir is not None:
            self.results_dir = self.results_dir.expanduser().absolute()
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
# May hold a remote run, and so an account's slot.
ACTIVE = {"submitting", "remote_queued", "running", "needs_attention"}
# Waiting to start on their account.
PENDING = {"queued", "preparing"}
# Without a possible remote run, so they can change account.
MOVABLE = {"queued", "preparing", "blocked"}
# Settled without operator action, apart from download retries.
HALTED = {"blocked", "needs_attention"}


def checked_states(states) -> list[str] | None:
    """Job states named by a caller, sorted and without repeats; None stays None (every state)."""
    if states is None:
        return None
    known = get_args(JobState)
    if isinstance(states, str) or not states or not set(states) <= set(known):
        raise ValueError(f"States must be a nonempty list of: {', '.join(known)}")
    return sorted(set(states))


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
    # That account cannot read some of the job's datasets, so moving there copies them.
    suggested_transfer: bool = False
    error: str | None = None
    next_action_at: float = 0
    last_polled_at: float | None = None
    finished_at: float | None = None
    download_state: Literal["pending", "downloading", "complete", "error", "disabled"] = "pending"
    download_error: str | None = None
    download_retry_at: float = 0
    download_failures: int = 0
    upload_refs: dict[str, str] = Field(default_factory=dict)
    # The run folder, fixed at submission: EXPERIMENT/NNN_TIMESTAMP, or results/ID in the state
    # directory for jobs saved before experiment folders (run is None).
    result_dir: Path
    run: int | None = None
    parent_id: str | None = None
    # May copy datasets its account cannot read from an account that can.
    transfer: bool = False
    # Datasets copied for this job: {alias: {"source": ..., "digest": ..., "bytes": ...}}.
    transfers: dict[str, dict] = Field(default_factory=dict)

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

    @property
    def pool(self) -> Pool:
        return self.spec.pool

    @property
    def movable(self) -> bool:
        """It has no possible remote run, so it can change account."""
        return self.state in MOVABLE

    @property
    def outstanding(self) -> bool:
        """Its latest attempt may still run remotely, so it holds a slot on that attempt's account."""
        return bool(
            self.attempts
            and self.attempts[-1].state in {"submitting", "accepted", "uncertain"}
            and not self.terminal
        )

    def settled(self, *, downloads: bool = True) -> bool:
        """Nothing further happens without operator action, apart from download retries."""
        if self.state in HALTED:
            return True
        return self.terminal and (not downloads or self.download_state in {"complete", "disabled", "error"})


class BatchRecord(Model):
    id: str
    created_at: float
    jobs: list[JobRecord]
    replayed: bool = False

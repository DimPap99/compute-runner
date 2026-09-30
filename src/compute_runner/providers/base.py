"""The contract between the queue and a compute provider, and what every adapter shares.

One provider instance serves one account. Adapters translate their service's identities,
states and errors into these shapes; the worker never sees provider-specific values.

Data follows one contract on every provider:

- Local inputs and copied datasets are content-addressed bundles. ensure_bundle() reuses one
  the account already holds and uploads it otherwise; bundles_for() says which a job needs.
- A provider dataset is attached directly when resolve_dataset() says the account can read it.
  If it cannot, the worker copies it from an account that can (fetch_dataset, then
  ensure_bundle) when the user allowed copying.
- The workload finds every input the same way: KGR_INPUT_<ALIAS> and KGR_INPUTS_JSON, set by
  the launch package that stage() builds. Workloads never use provider paths.
- download() hands each output to an OutputSink, which decides what to fetch and where it
  goes, so every provider fills the same run folder.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Account, JobRecord, JobSpec
    from ..results import OutputSink


@dataclass
class Inventory:
    """What holds an account's capacity now, and its GPU time.

    runs maps each run holding a slot, by lowercase reference, to cpu, gpu or unknown.
    gpu_seconds is None when GPU time is not limited, and 0 when the account has none.
    """

    runs: dict[str, str] = field(default_factory=dict)
    gpu_seconds: float | None = None
    # When the provider next restores GPU time, as a Unix timestamp; None when it does not say.
    gpu_refresh_at: float | None = None
    # GPU models on the machine, where the provider can tell.
    devices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Artifact:
    """Something this runner created on an account: a launch notebook or run folder, or a bundle.

    attempt is the attempt reference it belongs to (as JobRecord attempts record it), and digest
    the bundle digest it holds, or a prefix of it, so a cleanup can tell what still needs it.
    """

    kind: str
    name: str
    bytes: int | None = None
    modified_at: float | None = None
    attempt: str | None = None
    digest: str | None = None


class Provider(ABC):
    def __init__(self, account: Account, state_dir: Path, *, strict=False):
        self.account = account
        self.state_dir = state_dir
        self.strict = strict

    # Shared by every adapter ------------------------------------------------------------------

    def attempt_folder(self, job: JobRecord, number: int) -> Path:
        """Where an attempt's launch package is built, locally."""
        return self.state_dir / "jobs" / job.id / f"attempt-{number}"

    @staticmethod
    def launch_name(job: JobRecord, number: int) -> str:
        """kgr-NAME-JOBID-aN: unique per attempt, and readable on the provider."""
        name = re.sub(r"[^a-z0-9]+", "-", job.spec.name.lower())[:16].strip("-") or "workload"
        return f"kgr-{name}-{job.id[:12]}-a{number}"

    @staticmethod
    def bundled_inputs(job: JobRecord) -> list[tuple[str, dict]]:
        """(alias, bundle) for inputs that travel as bundles: local inputs, then dataset copies."""
        return [*job.snapshot["inputs"].items(), *job.transfers.items()]

    def bundles_for(self, job: JobRecord) -> dict[str, dict]:
        """Local bundles to make available before launch, by upload key: inputs, and a project's source."""
        bundles = {"input:" + alias: bundle for alias, bundle in job.snapshot["inputs"].items()}
        if not job.snapshot["single_file"]:
            bundles["source"] = job.snapshot["source"]
        return bundles

    def inventory(self) -> Inventory:
        """The runs holding the account's slots and its GPU time, read now."""
        runs = {ref.lower(): kind for ref, kind in self.active_runs().items()}
        quota = self.quota()
        gpu = quota.get("gpu")
        return Inventory(runs, gpu["available_seconds"] if gpu else 0, quota.get("refresh_at"))

    def diagnose(self) -> dict:
        """What doctor reports for the account: quota, active runs, and any warning."""
        return dict(quota=self.quota(), active_runs=self.active_runs())

    def runtime(self, ref: str) -> dict:
        """The resources the provider saved for a run, beside its session status."""
        raise ValueError(f"Runtime diagnostics are not available for {self.account.provider} accounts")

    def input_status(self, job: JobRecord) -> list[dict]:
        """What the provider reports about each input a job attaches; never uploads."""
        raise ValueError(f"Input diagnostics are not available for {self.account.provider} accounts")

    def artifacts(self) -> list[Artifact]:
        """What this runner created on the account."""
        return []

    def delete_artifact(self, artifact: Artifact) -> None:
        raise ValueError(f"{self.account.provider} accounts cannot delete {artifact.kind}s")

    # Each adapter's own -----------------------------------------------------------------------

    @abstractmethod
    def check(self, spec: JobSpec) -> None:
        """Raise ValueError if this provider cannot run the specification."""

    @abstractmethod
    def url(self, ref: str) -> str | None: ...

    @abstractmethod
    def ensure_bundle(self, bundle: dict) -> str | None:
        """Make a content-addressed local bundle available to runs; None while it is still processing."""

    @abstractmethod
    def resolve_dataset(self, ref: str) -> str | None:
        """Pin a dataset this account can read to an immutable version; None if it cannot read it.

        None also covers a dataset or pinned version that does not exist. Every call checks
        access, even for a pinned reference, so the worker can ask any account.
        Raise RemoteError for failures that say nothing about access.
        """

    @abstractmethod
    def fetch_dataset(self, ref: str, destination: Path) -> None:
        """Download a dataset this account can read, as plain files, into an empty folder."""

    @abstractmethod
    def stage(self, job: JobRecord, number: int) -> str:
        """Build attempt number's launch package locally and return its remote reference.

        No remote calls. The reference is deterministic, so status() can find the run
        even when submit() is interrupted. The package exposes each input alias as
        KGR_INPUT_<ALIAS>: bundles by job.upload_refs["input:ALIAS"], verified against their
        digest (from job.snapshot["inputs"] or job.transfers), and datasets attached directly.
        """

    @abstractmethod
    def submit(self, job: JobRecord) -> dict:
        """Launch the staged job.attempts[-1]; may return {"version": n}.

        Raise RemoteError. definitive=True asserts that nothing was launched.
        """

    @abstractmethod
    def status(self, ref: str) -> dict:
        """{"state": ..., "detail": raw provider state, "error": str | None}.

        state is queued, running, cancelling, succeeded, failed or cancelled; None if unrecognized.
        """

    @abstractmethod
    def cancel(self, ref: str, job_id: str) -> bool:
        """Stop a run. True when it was removed before it started, so it is cancelled now;
        False when a stop was requested and polling reports the outcome."""

    @abstractmethod
    def active_runs(self) -> dict[str, str]:
        """Runs holding this account's capacity, including ones started elsewhere: {ref: cpu|gpu|unknown}."""

    @abstractmethod
    def quota(self) -> dict:
        """{"gpu": {"available_seconds": ...} or None, "refresh_at": timestamp or None, ...}

        gpu None means no GPU time is available; available_seconds None means GPU time is not limited.
        """

    @abstractmethod
    def logs(self, ref: str, *, follow: bool = False) -> Iterator[str]:
        """The stored log of a finished run, or a stream with follow=True."""

    @abstractmethod
    def live_log(self, ref: str) -> str:
        """A bounded snapshot of an unfinished run's log."""

    @abstractmethod
    def download(self, ref: str, sink: OutputSink) -> None:
        """Give sink the run's log (sink.log) and its output files.

        For each file, ask sink.target(name) for a path (None: skip it), write the bytes there
        atomically, then call sink.saved(name, path, sha256).
        """

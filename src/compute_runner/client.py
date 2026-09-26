"""Public API. Submitting is local; a worker owns remote scheduling."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .bundle import describe, snapshot
from .models import BatchRecord, Config, JobRecord, JobSpec
from .providers import Provider, connect
from .runtime import json_digest
from .store import Store, load_config, try_lock
from .worker import MOVABLE, Worker, collect_outputs, outstanding, place, settled


class Client:
    def __init__(self, *, config: Config | None = None, state_dir=None, providers=None):
        """providers optionally supplies ready Provider instances by account ID."""
        self.config = (config or load_config()).model_copy(deep=True)
        if state_dir is not None:
            self.config.state_dir = Path(state_dir).expanduser().resolve()
        self.store = Store(self.config.state_dir)
        self._providers: dict[str, Provider] = dict(providers or {})

    def provider(self, account: str | None = None) -> Provider:
        """The adapter for a configured account, or the default one; connecting does not authenticate."""
        account = self.config.account(account)
        if account.id not in self._providers:
            # Concurrent first uses (worker and download threads) share one instance.
            self._providers.setdefault(account.id, connect(account, self.config))
        return self._providers[account.id]

    def preview(self, spec: JobSpec, account: str | None = None):
        """The files a submission would upload; checks the spec against the account without remote calls."""
        self.provider(account).check(spec)
        return describe(spec)

    def submit(
        self, spec: JobSpec, *, request_key: str | None = None, account: str | None = None
    ) -> JobRecord:
        return self.submit_batch([spec], request_key=request_key, account=account).jobs[0]

    @staticmethod
    def check_batch_size(specs):
        if not 1 <= len(specs) <= 1000:
            raise ValueError("A batch must contain between 1 and 1000 jobs")

    @staticmethod
    def _fingerprint(value, request_key, account=None):
        if request_key is not None and (
            not isinstance(request_key, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", request_key)
        ):
            raise ValueError(
                "Request key must be 1–128 letters, digits, dots, underscores, colons, slashes or hyphens"
            )
        # An explicit account is part of the intent; requests without one keep their original fingerprint.
        return json_digest(value if account is None else value | {"account": account})

    def submit_batch(
        self, specs: list[JobSpec], *, request_key: str | None = None, account: str | None = None
    ) -> BatchRecord:
        """Atomically queue a batch; a repeated key returns its original immutable snapshots.

        Keys identify intent, not current file contents. New code needs a new key.
        A conflicting specification is rejected, even after the original batch finishes.
        Jobs are placed on account, or on the first configured account.
        """
        target = self.config.account(account).id
        self.check_batch_size(specs)
        normalized = []
        for spec in specs:
            spec = JobSpec.model_validate(spec.model_dump())
            spec.source = spec.source.expanduser().absolute()
            spec.inputs = {alias: path.expanduser().absolute() for alias, path in spec.inputs.items()}
            normalized.append(spec)
        fingerprint = self._fingerprint(
            {"submit": [spec.model_dump(mode="json") for spec in normalized]}, request_key, account and target
        )
        previous = self.store.request(request_key, fingerprint)
        if previous is not None:
            return previous
        for spec in normalized:
            self.provider(target).check(spec)
        # Snapshot everything before exposing any job to the worker.
        jobs = [self._new_job(spec, snapshot(spec, self.config.state_dir), target) for spec in normalized]
        batch = BatchRecord(id=uuid.uuid4().hex, created_at=time.time(), jobs=jobs)
        return self.store.add_batch(batch, request_key=request_key, fingerprint=fingerprint)

    def _new_job(self, spec, saved, account, parent_id=None):
        job_id = uuid.uuid4().hex
        return JobRecord(
            id=job_id,
            spec=spec,
            snapshot=saved,
            account=account,
            result_dir=self.config.state_dir / "results" / job_id,
            download_state="pending" if spec.auto_download else "disabled",
            parent_id=parent_id,
        )

    def submit_many(
        self, specs: list[JobSpec], *, request_key: str | None = None, account: str | None = None
    ) -> list[JobRecord]:
        """Queue atomically and return jobs in input order."""
        return self.submit_batch(specs, request_key=request_key, account=account).jobs

    def batch(self, batch_id: str) -> BatchRecord:
        return self.store.batch(batch_id)

    def agent(self):
        """Bounded, JSON-serializable API for agents; no authentication on creation."""
        from .agent import AgentClient

        return AgentClient(self)

    def get(self, job_id):
        return self.store.get(job_id)

    def list(self, *, states=None):
        return self.store.list(states)

    def cancel(self, job_id):
        """Cancel pending work locally, or ask the provider to stop an accepted run.

        The worker records a remote cancellation when it next polls the run.
        """
        job = self.get(job_id)
        if job.terminal:
            raise ValueError(f"The job has already finished ({job.state})")
        if outstanding(job):
            attempt = job.attempts[-1]
            if attempt.state != "accepted":
                raise ValueError(f"The submission is unconfirmed; inspect it first: {job.url}")
            self.provider(attempt.account).cancel(attempt.ref, job.id)
            return self.store.update(
                job_id, wait_reason=f"Cancellation requested on {attempt.account}", next_action_at=0
            )
        updated = self.store.update(
            job_id,
            expected={"queued", "preparing", "blocked"},
            state="cancelled",
            finished_at=time.time(),
            wait_reason="Cancelled locally",
            suggested_account=None,
            download_state="disabled",
        )
        if updated is None:
            raise ValueError("Job changed state during cancellation; inspect its current status")
        return updated

    def retry(self, job_id, *, request_key: str | None = None, account: str | None = None):
        return self.retry_batch(job_id, request_key=request_key, account=account).jobs[0]

    def retry_batch(self, job_id, *, request_key: str | None = None, account: str | None = None):
        """Rerun a job's saved snapshot, on its account unless another is given."""
        explicit = account and self.config.account(account).id
        fingerprint = self._fingerprint({"retry": job_id}, request_key, explicit)
        previous = self.store.request(request_key, fingerprint)
        if previous is not None:
            return previous
        job = self.get(job_id)
        if outstanding(job):
            raise ValueError(f"An execution may still exist; resolve it before rerunning: {job.url}")
        if not job.terminal and job.state != "blocked":
            raise ValueError("Retry accepts a terminal or blocked job only")
        for bundle in [job.snapshot["source"], *job.snapshot["inputs"].values()]:
            if not (self.config.state_dir / "bundles" / bundle["digest"] / "payload.zip").is_file():
                raise ValueError("Saved bundle is missing; submit a new workload")
        target = explicit or self.config.account(job.account).id
        self.provider(target).check(job.spec)
        new = self._new_job(job.spec.model_copy(deep=True), job.snapshot, target, parent_id=job.id)
        return self.store.add_batch(
            BatchRecord(id=uuid.uuid4().hex, created_at=time.time(), jobs=[new]),
            request_key=request_key,
            fingerprint=fingerprint,
        )

    def move(self, job_id, account: str) -> JobRecord:
        """Place a job that has not been submitted on another configured account."""
        job = self.get(job_id)
        target = self.config.account(account).id
        if target == job.account:
            raise ValueError(f"The job is already on {target}")
        if job.state not in MOVABLE:
            raise ValueError(f"Only jobs that have not been submitted can move; this one is {job.state}")
        self.provider(target).check(job.spec)
        moved = place(self.store, job_id, target, f"Moved from {job.account} on request")
        if moved is None:
            raise ValueError("Job changed state while moving; inspect its current status")
        return moved

    def resolve_not_submitted(self, job_id):
        """Operator assertion after independently confirming no remote execution exists.

        Does not submit a replacement. Use retry() after this explicit resolution.
        """
        job = self.get(job_id)
        if job.state != "needs_attention" or not job.attempts or job.attempts[-1].state == "accepted":
            raise ValueError("Only an unresolved, unaccepted attempt can be marked not submitted")
        job.attempts[-1].state = "rejected"
        job.attempts[-1].error = "Operator confirmed that no remote execution exists"
        return self.store.update(
            job_id,
            expected={"needs_attention"},
            state="blocked",
            attempts=job.attempts,
            error=None,
            wait_reason="Operator resolved missing submission; retry explicitly",
        )

    def wait(self, job_id, *, timeout=None, downloads=True):
        """Return once the job settles; a failed download is returned too (the worker retries it)."""
        jobs, done = self.wait_many([job_id], timeout=timeout, downloads=downloads)
        if not done:
            raise TimeoutError(f"Waiting for {job_id} timed out; the job remains tracked")
        return jobs[0]

    def wait_many(self, job_ids=None, *, batch_id=None, timeout=None, downloads=True):
        """Wait until every selected job settles or timeout passes; return (jobs, all_settled).

        Tolerates a worker restart, but fails once no worker has run for 30 seconds.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        stopped_since = None
        while True:
            _, jobs = self.store.page(batch_id=batch_id, job_ids=job_ids, limit=-1)
            now = time.monotonic()
            if all(settled(job, downloads=downloads) for job in jobs):
                return jobs, True
            if deadline is not None and now >= deadline:
                return jobs, False
            if self.worker_health()["running"]:
                stopped_since = None
            elif now - (stopped_since := stopped_since or now) >= 30:
                raise RuntimeError(
                    "No worker is running. Start compute-runner service start or compute-runner worker run"
                )
            time.sleep(min(2, self.config.poll_seconds, deadline - now if deadline else 2))

    def logs(self, job_id, *, follow=False):
        """Persisted logs of a finished run, a bounded snapshot of an unfinished one, or a stream."""
        job = self.get(job_id)
        if not job.remote_ref:
            raise ValueError("This job has not been submitted yet")
        provider = self.provider(job.attempts[-1].account)
        if not follow and not job.terminal:
            # Providers may persist logs only after a run ends.
            yield provider.live_log(job.remote_ref)
            return
        yield from provider.logs(job.remote_ref, follow=follow)

    def download(self, job_id):
        return collect_outputs(self.store, self.provider, job_id)

    def quota(self, account: str | None = None):
        """One account's quota, or every account's by ID."""
        if account is not None:
            return self.provider(account).quota()
        return {item.id: self.provider(item.id).quota() for item in self.config.accounts}

    def worker(self):
        return Worker(self.config, self.provider, self.store)

    def worker_health(self):
        path = self.config.state_dir / "worker.json"
        try:
            info = json.loads(path.read_text()) if path.exists() else {}
        except (ValueError, OSError):
            info = {}
        with (self.config.state_dir / "worker.lock").open("a+") as lock:
            running = not try_lock(lock)
        return info | {
            "running": running,
            "heartbeat_age_seconds": time.time() - info["timestamp"] if "timestamp" in info else None,
        }

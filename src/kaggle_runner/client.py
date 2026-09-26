"""Public API. Submitting is local; a worker owns remote scheduling."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from .backend import KaggleBackend
from .bundle import describe, snapshot
from .models import BatchRecord, Config, JobRecord, JobSpec
from .store import Store, load_config
from .worker import Worker, collect_outputs, outstanding, settled


class Client:
    def __init__(self, *, config: Config | None = None, state_dir=None, backend=None):
        self.config = (config or load_config()).model_copy(deep=True)
        if state_dir is not None:
            self.config.state_dir = Path(state_dir).expanduser().resolve()
        self.store = Store(self.config.state_dir)
        self._backend = backend
        if self.config.owner:
            with self.store.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                owner = db.execute("SELECT value FROM meta WHERE key='owner'").fetchone()
                if owner and owner[0] != self.config.owner:
                    raise ValueError("This state directory belongs to another Kaggle account")
                db.execute("INSERT OR IGNORE INTO meta VALUES ('owner', ?)", (self.config.owner,))

    @property
    def backend(self):
        if self._backend is None:
            self._backend = KaggleBackend(self.config.owner, self.config.state_dir)
        return self._backend

    def preview(self, spec: JobSpec):
        return describe(spec)

    def submit(self, spec: JobSpec, *, request_key: str | None = None) -> JobRecord:
        return self.submit_batch([spec], request_key=request_key).jobs[0]

    @staticmethod
    def check_batch_size(specs):
        if not 1 <= len(specs) <= 1000:
            raise ValueError("A batch must contain between 1 and 1000 jobs")

    @staticmethod
    def _fingerprint(value, request_key):
        if request_key is not None and (
            not isinstance(request_key, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", request_key)
        ):
            raise ValueError(
                "Request key must be 1–128 letters, digits, dots, underscores, colons, slashes or hyphens"
            )
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def submit_batch(self, specs: list[JobSpec], *, request_key: str | None = None) -> BatchRecord:
        """Atomically queue a batch; a repeated key returns its original immutable snapshots.

        Keys identify intent, not current file contents. New code needs a new key.
        A conflicting specification is rejected, even after the original batch finishes.
        """
        if not self.config.owner:
            raise ValueError("Configure your account first: kgr init --owner YOUR_USERNAME")
        self.check_batch_size(specs)
        normalized = []
        for spec in specs:
            spec = JobSpec.model_validate(spec.model_dump())
            spec.source = spec.source.expanduser().absolute()
            spec.inputs = {alias: path.expanduser().absolute() for alias, path in spec.inputs.items()}
            normalized.append(spec)
        fingerprint = self._fingerprint(
            {"submit": [spec.model_dump(mode="json") for spec in normalized]}, request_key
        )
        previous = self.store.request(request_key, fingerprint)
        if previous is not None:
            return previous
        # Snapshot everything before exposing any job to the worker.
        jobs = [self._new_job(spec, snapshot(spec, self.config.state_dir)) for spec in normalized]
        batch = BatchRecord(id=uuid.uuid4().hex, created_at=time.time(), jobs=jobs)
        return self.store.add_batch(batch, request_key=request_key, fingerprint=fingerprint)

    def _new_job(self, spec, saved, parent_id=None):
        job_id = uuid.uuid4().hex
        return JobRecord(
            id=job_id,
            spec=spec,
            snapshot=saved,
            owner=self.config.owner,
            result_dir=self.config.state_dir / "results" / job_id,
            download_state="pending" if spec.auto_download else "disabled",
            parent_id=parent_id,
        )

    def submit_many(self, specs: list[JobSpec], *, request_key: str | None = None) -> list[JobRecord]:
        """Queue atomically and return jobs in input order."""
        return self.submit_batch(specs, request_key=request_key).jobs

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
        job = self.get(job_id)
        if outstanding(job) or job.terminal:
            raise ValueError(
                f"Cannot cancel this local job. Stop active execution on Kaggle: {job.url or job.id}"
            )
        updated = self.store.update(
            job_id,
            expected={"queued", "preparing", "blocked"},
            state="cancelled",
            finished_at=time.time(),
            wait_reason="Cancelled locally",
            download_state="disabled",
        )
        if updated is None:
            raise ValueError("Job changed state during cancellation; inspect its current status")
        return updated

    def retry(self, job_id, *, request_key: str | None = None):
        return self.retry_batch(job_id, request_key=request_key).jobs[0]

    def retry_batch(self, job_id, *, request_key: str | None = None):
        fingerprint = self._fingerprint({"retry": job_id}, request_key)
        previous = self.store.request(request_key, fingerprint)
        if previous is not None:
            return previous
        job = self.get(job_id)
        if outstanding(job):
            raise ValueError(
                f"An execution may still exist; resolve it on Kaggle before rerunning: {job.url}"
            )
        if not job.terminal and job.state != "blocked":
            raise ValueError("Retry accepts a terminal or blocked job only")
        for bundle in [job.snapshot["source"], *job.snapshot["inputs"].values()]:
            if not (self.config.state_dir / "bundles" / bundle["digest"] / "payload.zip").is_file():
                raise ValueError("Saved bundle is missing; submit a new workload")
        new = self._new_job(job.spec.model_copy(deep=True), job.snapshot, parent_id=job.id)
        return self.store.add_batch(
            BatchRecord(id=uuid.uuid4().hex, created_at=time.time(), jobs=[new]),
            request_key=request_key,
            fingerprint=fingerprint,
        )

    def resolve_not_submitted(self, job_id):
        """Operator assertion after independently confirming no remote execution exists.

        Does not submit a replacement. Use retry() after this explicit resolution.
        """
        job = self.get(job_id)
        if job.state != "needs_attention" or not job.attempts or job.attempts[-1].state == "accepted":
            raise ValueError("Only an unresolved, unaccepted attempt can be marked not submitted")
        attempts = [attempt.model_dump() for attempt in job.attempts]
        attempts[-1]["state"] = "rejected"
        attempts[-1]["error"] = "Operator confirmed that no remote execution exists"
        return self.store.update(
            job_id,
            expected={"needs_attention"},
            state="blocked",
            attempts=attempts,
            error=None,
            wait_reason="Operator resolved missing submission; retry explicitly",
        )

    def wait(self, job_id, *, timeout=None, downloads=True):
        """Return once the job settles; a failed download is returned too (the worker retries it)."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            job = self.get(job_id)
            if settled(job, downloads=downloads):
                return job
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Waiting for {job_id} timed out; the job remains tracked")
            if not self.worker_health()["running"]:
                raise RuntimeError("No worker is running. Start kgr service start or kgr worker run")
            time.sleep(min(self.config.poll_seconds, 2))

    def logs(self, job_id, *, follow=False):
        """Persisted logs of a finished run, a bounded snapshot of an unfinished one, or a stream."""
        job = self.get(job_id)
        if not job.remote_ref:
            raise ValueError("This job has not been submitted to Kaggle yet")
        if not follow and not job.terminal:
            # Kaggle persists logs only after a session ends.
            yield self.backend.live_log(job.remote_ref)
            return
        yield from self.backend.logs(job.remote_ref, follow=follow)

    def download(self, job_id):
        return collect_outputs(self.store, self.backend, job_id)

    def quota(self):
        return self.backend.quota()

    def worker(self):
        return Worker(self.config, self.backend, self.store)

    def worker_health(self):
        path = self.config.state_dir / "worker.json"
        try:
            info = json.loads(path.read_text()) if path.exists() else {}
        except (ValueError, OSError):
            info = {}
        with (self.config.state_dir / "worker.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                running = False
                fcntl.flock(lock, fcntl.LOCK_UN)
            except BlockingIOError:
                running = True
        return info | {
            "running": running,
            "heartbeat_age_seconds": time.time() - info["timestamp"] if "timestamp" in info else None,
        }

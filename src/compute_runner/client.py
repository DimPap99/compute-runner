"""Public API. Submitting is local; a worker owns remote scheduling."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .bundle import describe, snapshot, snapshot_bundle
from .models import SSH_PATH, BatchRecord, Config, JobRecord, JobSpec, input_reference
from .providers import Provider, connect
from .results import experiment_dir, outputs_dir, verified_checkpoint
from .runtime import json_digest, safe_relative
from .store import Store, load_config, try_lock
from .worker import MOVABLE, Worker, collect_outputs, outstanding, place, settled


def resume_required(spec: JobSpec) -> JobSpec:
    """The spec with --resume required, as a parameter, an argument, or appended."""
    spec = spec.model_copy(deep=True)
    if "resume" in spec.params:
        spec.params["resume"] = "required"
        return spec
    for index, arg in enumerate(spec.args):
        if arg == "--resume":
            spec.args[index + 1 :] = ["required", *spec.args[index + 2 :]]
            return spec
        if arg.startswith("--resume="):
            spec.args[index] = "--resume=required"
            return spec
    spec.args += ["--resume", "required"]
    return spec


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
        """The files a submission would upload and its experiment folder; no remote calls."""
        spec = self._normalize(spec, self.config.account(account))
        self.provider(account).check(spec)
        experiment = self._experiment(spec)
        return describe(self._local(spec), skip=[experiment.parent]) | {"experiment_dir": str(experiment)}

    def _normalize(self, spec, account=None):
        """Absolute paths, job:ID inputs naming complete job IDs, and ssh inputs naming their machine.

        An ssh:/PATH input means the account's own machine, so it keeps meaning that machine
        if the job moves: elsewhere it can only be copied.
        """
        spec = JobSpec.model_validate(spec.model_dump())
        spec.source = spec.source.expanduser().absolute()
        if spec.results_dir is not None:
            spec.results_dir = spec.results_dir.expanduser().absolute()
        inputs = {}
        for alias, value in spec.inputs.items():
            reference = input_reference(value)
            if reference is None:
                inputs[alias] = value.expanduser().absolute()
            elif reference[0] == "job":
                job_id, _, path = reference[1].partition("/")
                job_id = self.store.resolve_id(job_id)
                inputs[alias] = Path(f"job:{job_id}/{safe_relative(path)}" if path else f"job:{job_id}")
            elif reference[0] == "ssh" and SSH_PATH.match(reference[1])["machine"] is None:
                if account is None or account.provider != "ssh":
                    raise ValueError(f"Name the machine of input {alias}: ssh:NAME:{reference[1]}")
                inputs[alias] = Path(f"ssh:{account.user}:{reference[1]}")
            else:
                inputs[alias] = value
        spec.inputs = inputs
        return spec

    def _local(self, spec):
        """spec for bundling: job outputs as local paths; provider datasets are attached remotely."""
        inputs = {}
        for alias, value in spec.inputs.items():
            reference = input_reference(value)
            if reference is None:
                inputs[alias] = value
            elif reference[0] == "job":
                inputs[alias] = self._job_output(reference[1])
        return spec.model_copy(update={"inputs": inputs})

    def _job_output(self, value):
        """The downloaded file or folder that job:ID[/PATH] names in that job's KGR_OUTPUT_DIR."""
        job_id, _, path = value.partition("/")
        job = self.get(job_id)
        if job.download_state != "complete":
            raise ValueError(f"Outputs of job {job.id} are not downloaded yet ({job.download_state})")
        found = outputs_dir(job) / path
        if not found.exists():
            raise ValueError(f"Job {job.id} has no output {path or 'folder'}")
        return found

    def _experiment(self, spec):
        folder = experiment_dir(spec, self.config)
        # Run folders would land in the source, and later snapshots would upload them.
        if spec.source.is_dir() and spec.source.resolve() in {folder.parent.resolve(), folder.resolve()}:
            raise ValueError("results_dir must not be the source folder itself or its experiment folder")
        return folder

    @staticmethod
    def _intent(spec):
        # Fields added later are left out at their defaults, so earlier request keys still replay.
        added = {"params": {}, "results_dir": None}
        unset = {key for key, empty in added.items() if getattr(spec, key) == empty}
        return spec.model_dump(mode="json", exclude=unset)

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
        normalized = [self._normalize(spec, self.config.account(target)) for spec in specs]
        fingerprint = self._fingerprint(
            {"submit": [self._intent(spec) for spec in normalized]}, request_key, account and target
        )
        previous = self.store.request(request_key, fingerprint)
        if previous is not None:
            return previous
        for spec in normalized:
            self.provider(target).check(spec)
        # Snapshot everything before exposing any job to the worker.
        jobs = []
        for spec in normalized:
            experiment = self._experiment(spec)
            saved = snapshot(self._local(spec), self.config.state_dir, skip=[experiment.parent])
            jobs.append(self._new_job(spec, saved, target, experiment))
        return self._add(jobs, request_key, fingerprint)

    def _new_job(self, spec, saved, account, experiment, parent_id=None):
        """A job for experiment; _add numbers its run folder when the batch commits."""
        return JobRecord(
            id=uuid.uuid4().hex,
            spec=spec,
            snapshot=saved,
            account=account,
            result_dir=experiment,
            download_state="pending" if spec.auto_download else "disabled",
            parent_id=parent_id,
        )

    def _add(self, jobs, request_key, fingerprint):
        batch = BatchRecord(id=uuid.uuid4().hex, created_at=time.time(), jobs=jobs)
        experiments = {job.id: job.result_dir for job in jobs}
        return self.store.add_batch(
            batch, request_key=request_key, fingerprint=fingerprint, experiments=experiments
        )

    def _same_experiment(self, job):
        return job.result_dir.parent if job.run is not None else self._experiment(job.spec)

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
            if self.provider(attempt.account).cancel(attempt.ref, job.id):
                # Removed before it started: no run remains for the worker to poll.
                return self.store.update(
                    job_id,
                    state="cancelled",
                    finished_at=time.time(),
                    wait_reason=f"Cancelled before it started on {attempt.account}",
                    download_state="disabled",
                )
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
            suggested_transfer=False,
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
        self._check_saved(job)
        target = explicit or self.config.account(job.account).id
        self.provider(target).check(job.spec)
        spec = job.spec.model_copy(deep=True)
        new = self._new_job(spec, job.snapshot, target, self._same_experiment(job), job.id)
        return self._add([new], request_key, fingerprint)

    def _check_saved(self, job):
        for bundle in [job.snapshot["source"], *job.snapshot["inputs"].values()]:
            if not (self.config.state_dir / "bundles" / bundle["digest"] / "payload.zip").is_file():
                raise ValueError("Saved bundle is missing; submit a new workload")

    def continue_run(self, job_id, *, request_key: str | None = None, account: str | None = None):
        return self.continue_batch(job_id, request_key=request_key, account=account).jobs[0]

    def continue_batch(self, job_id, *, request_key: str | None = None, account: str | None = None):
        """Resume a stopped resumable run from its last verified checkpoint as its experiment's next run.

        Reuses the job's saved code and inputs, attaches its downloaded KGR_OUTPUT_DIR/checkpoints
        as the input resume (only latest.json and the file it names), and passes --resume required.
        Runs on the job's account unless another is given.
        """
        explicit = account and self.config.account(account).id
        fingerprint = self._fingerprint({"continue": job_id}, request_key, explicit)
        previous = self.store.request(request_key, fingerprint)
        if previous is not None:
            return previous
        job = self.get(job_id)
        if not job.terminal:
            raise ValueError(f"Continue a run after it stops; this one is {job.state}")
        if job.download_state != "complete":
            raise ValueError(
                f"Its outputs are not downloaded yet ({job.download_state}); wait for outputs_ready"
            )
        folder = outputs_dir(job) / "checkpoints"
        checkpoint = verified_checkpoint(folder)
        self._check_saved(job)
        spec = resume_required(job.spec)
        saved = dict(job.snapshot, inputs=dict(job.snapshot["inputs"]))
        # The checkpoint replaces an input of that name in any casing; both would be KGR_INPUT_RESUME.
        for alias in [alias for alias in spec.inputs if alias.upper() == "RESUME"]:
            del spec.inputs[alias]
            saved["inputs"].pop(alias, None)
        spec.inputs["resume"] = Path(f"job:{job.id}/checkpoints")
        target = explicit or self.config.account(job.account).id
        self.provider(target).check(spec)
        saved["inputs"]["resume"] = snapshot_bundle(
            folder, ["latest.json", checkpoint], self.config.state_dir / "bundles"
        )
        new = self._new_job(spec, saved, target, self._same_experiment(job), job.id)
        return self._add([new], request_key, fingerprint)

    def move(self, job_id, account: str, *, transfer: bool = False) -> JobRecord:
        """Place a job that has not been submitted on another configured account.

        transfer allows copying datasets that account cannot read from one that can; it also
        applies to a job left on its account.
        """
        job = self.get(job_id)
        target = self.config.account(account).id
        if target == job.account and not (transfer and not job.transfer):
            raise ValueError(f"The job is already on {target}")
        if job.state not in MOVABLE:
            raise ValueError(f"Only jobs that have not been submitted can move; this one is {job.state}")
        self.provider(target).check(job.spec)
        reason = f"Moved from {job.account} on request" if target != job.account else "Dataset copies allowed"
        moved = place(self.store, job_id, target, reason, transfer=transfer or job.transfer)
        if moved is None:
            raise ValueError("Job changed state while moving; inspect its current status")
        return moved

    def resolve_not_submitted(self, job_id):
        """Operator assertion after independently confirming no remote execution exists.

        Covers an uncertain submission, and an accepted run the provider no longer knows, such as
        a deleted notebook. Does not submit a replacement. Use retry() after this explicit resolution.
        """
        job = self.get(job_id)
        if job.state != "needs_attention" or not job.attempts:
            raise ValueError("Only a job that needs attention can be marked not submitted")
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
        watch = self._worker_watch()
        while True:
            _, jobs = self.store.page(batch_id=batch_id, job_ids=job_ids, limit=-1)
            now = time.monotonic()
            if all(settled(job, downloads=downloads) for job in jobs):
                return jobs, True
            if deadline is not None and now >= deadline:
                return jobs, False
            watch()
            time.sleep(min(2, self.config.poll_seconds, deadline - now if deadline else 2))

    def _worker_watch(self):
        """A check to call while waiting on the worker: it raises once no worker has run for 30 seconds."""
        stopped_since = None

        def check():
            nonlocal stopped_since
            now = time.monotonic()
            if self.worker_health()["running"]:
                stopped_since = None
            elif now - (stopped_since := stopped_since or now) >= 30:
                raise RuntimeError(
                    "No worker is running. Start compute-runner service start or compute-runner worker run"
                )

        return check

    def logs(self, job_id, *, follow=False):
        """Persisted logs of a finished run, a bounded snapshot of an unfinished one, or a stream.

        Following a job that has not been launched yet waits for its launch, as long as a worker runs.
        """
        job = self.get(job_id)
        watch = self._worker_watch()
        while follow and not job.remote_ref and job.state in {"queued", "preparing", "submitting"}:
            watch()
            time.sleep(min(2, self.config.poll_seconds))
            job = self.get(job_id)
        if not job.remote_ref:
            raise ValueError(f"This job has not been submitted yet ({job.state})")
        provider = self.provider(job.attempts[-1].account)
        if not follow and not job.terminal:
            # Providers may persist logs only after a run ends.
            yield provider.live_log(job.remote_ref)
            return
        yield from provider.logs(job.remote_ref, follow=follow)

    def download(self, job_id):
        return collect_outputs(self.store, self.provider, job_id, strict=self.config.strict)

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

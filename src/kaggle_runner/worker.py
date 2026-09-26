"""Single dispatcher, persistent attempts, independent artifact downloads."""

from __future__ import annotations

import fcntl
import logging
import re
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .backend import RemoteError, safe_message
from .launcher import prepare_kernel
from .models import ACTIVE, TERMINAL, Attempt, Config
from .security import redacted_env_record
from .store import Store, atomic_json

logger = logging.getLogger(__name__)
PENDING = {"queued", "preparing"}


def collect_outputs(store, backend, job_id):
    job = store.get(job_id)
    if not job.remote_ref or not job.terminal:
        raise ValueError("Outputs may be collected after a submitted run terminates")
    job.result_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (job.result_dir / ".download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return store.get(job_id)
        try:
            store.update(job_id, download_state="downloading", download_error=None)
            backend.download(
                job.remote_ref, job.result_dir, job.spec.output_patterns, skip=source_copies(job)
            )
            updated = store.update(job_id, download_state="complete", download_error=None)
            atomic_json(
                job.result_dir / "provenance.json",
                redacted_env_record(updated.model_dump(mode="json")),
            )
            return updated
        except Exception as error:
            store.update(
                job_id,
                download_state="error",
                download_error=safe_message(error),
                download_retry_at=time.time() + 60,
            )
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def source_copies(job):
    """Match output names of the project copy the runtime made; saved bundles already hold them.

    Files the workload creates under the project folder are still downloaded, except
    __pycache__ bytecode. In-place edits of snapshot files are not collected.
    """
    names = {f"project/{name}" for name in job.snapshot["source"]["files"]}
    return lambda name: name in names or (name.startswith("project/") and "__pycache__" in name.split("/"))


def outstanding(job):
    return bool(
        job.attempts
        and job.attempts[-1].state in {"submitting", "accepted", "uncertain"}
        and job.state not in TERMINAL
    )


def settled(job, *, downloads=True):
    """Nothing further happens without operator action, apart from download retries."""
    if job.state in {"blocked", "needs_attention"}:
        return True
    return job.terminal and (not downloads or job.download_state in {"complete", "disabled", "error"})


class Worker:
    def __init__(self, config: Config, backend, store=None):
        self.config = config
        self.store = store or Store(config.state_dir)
        self.backend = backend
        self.stop_event = threading.Event()
        self.inventory = {}
        self.inventory_at = None
        self.inventory_error = None
        self.discovery_retry_at = 0
        self.pool = None
        self.downloads = {}

    def tick(self):
        """One complete, locked cycle; also useful for cron and deterministic tests."""
        with self.store.worker_lock():
            try:
                self._tick()
            finally:
                self.store.heartbeat(state="stopped", mode="once")

    def run(self):
        with self.store.worker_lock():
            previous_handlers = {}
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[sig] = signal.signal(sig, lambda *_: self.stop_event.set())
            try:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kgr-download") as pool:
                    self.pool = pool
                    while not self.stop_event.is_set():
                        try:
                            self._tick()
                        except Exception:
                            logger.exception("Worker cycle failed; saved work will be retried")
                            self.store.heartbeat(
                                state="running", error="Worker cycle failed; see service logs"
                            )
                        self.stop_event.wait(self.config.poll_seconds)
            finally:
                self.pool = None
                self.store.heartbeat(state="stopped")
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)

    def _tick(self):
        self.store.heartbeat(state="running", stage="monitoring")
        now = time.time()
        for job in self.store.list(ACTIVE):
            if outstanding(job) and job.next_action_at <= now:
                self._poll(job)
        pending = self.store.list(PENDING)
        if pending and not self.stop_event.is_set():
            self._refresh_inventory()
            self._dispatch(pending)
        self._downloads()
        self.store.heartbeat(state="running", stage="idle", discovery_error=self.inventory_error)

    def _poll(self, job):
        attempt = job.attempts[-1]
        now = time.time()
        try:
            status = self.backend.status(attempt.ref)
        except RemoteError as error:
            uncertain = attempt.state in {"submitting", "uncertain"}
            expired = now - attempt.started_at >= self.config.reconcile_seconds
            new_state = "needs_attention" if expired and (uncertain or error.kind == "missing") else job.state
            if uncertain:
                attempt.state = "uncertain"
            self.store.update(
                job.id,
                state=new_state,
                attempts=job.attempts,
                error=safe_message(error),
                wait_reason="Reconciling submission; no duplicate will launch"
                if uncertain
                else "Remote status unavailable; capacity remains reserved",
                last_polled_at=now,
                next_action_at=now + self.config.poll_seconds,
            )
            return
        remote = status["state"]
        mapping = {
            "QUEUED": "remote_queued",
            "RUNNING": "running",
            "COMPLETE": "succeeded",
            "ERROR": "failed",
            "CANCEL_ACKNOWLEDGED": "cancelled",
            "CANCEL_REQUESTED": "running",
        }
        if remote not in mapping:
            self.store.update(
                job.id,
                remote_state=remote,
                last_polled_at=now,
                state="needs_attention"
                if now - attempt.started_at >= self.config.reconcile_seconds
                else job.state,
                wait_reason="Remote execution not confirmed; capacity remains reserved",
                next_action_at=now + self.config.poll_seconds,
            )
            return
        attempt.state = "accepted"
        state = mapping[remote]
        changes = dict(
            state=state,
            remote_state=remote,
            last_polled_at=now,
            attempts=job.attempts,
            error=safe_message(status["error"]) if status.get("error") else None,
            wait_reason="Cancellation requested on Kaggle" if remote == "CANCEL_REQUESTED" else None,
            next_action_at=now + self.config.poll_seconds,
        )
        if state in TERMINAL:
            changes["finished_at"] = now
            self.inventory.pop(attempt.ref, None)
        self.store.update(job.id, **changes)

    def _refresh_inventory(self):
        now = time.time()
        if now < self.discovery_retry_at:
            return
        if self.inventory_at is not None and now - self.inventory_at < self.config.discovery_seconds:
            return
        try:
            self.store.heartbeat(state="running", stage="discovering account runs")
            self.inventory = self.backend.active_runs()
            self.inventory_at = now
            self.inventory_error = None
        except Exception as error:
            self.inventory_error = safe_message(error)
            self.discovery_retry_at = now + self.config.retry_seconds

    def _counts(self):
        jobs = [j for j in self.store.list() if outstanding(j)]
        refs = {j.remote_ref for j in jobs}
        counts = {"cpu": sum(not j.spec.gpu for j in jobs), "gpu": sum(j.spec.gpu for j in jobs)}
        for ref, resource in self.inventory.items():
            if ref not in refs:
                for kind in ("cpu", "gpu"):
                    if resource in {kind, "unknown"}:
                        counts[kind] += 1
        return counts

    def _full(self, pool):
        return self._counts()[pool] >= getattr(self.config, pool + "_limit")

    def _hold(self, job, reason, **changes):
        """Keep a pending job queued, with a visible reason."""
        self.store.update(job.id, expected=PENDING, wait_reason=reason, **changes)

    def _gpu_quota_problem(self):
        try:
            gpu = self.backend.quota().get("gpu")
        except Exception as error:
            return "GPU quota unavailable", safe_message(error)
        if gpu is None or gpu["available_seconds"] <= 0:
            return "Waiting for available GPU quota", None
        return None

    def _dispatch(self, pending):
        blocked_pools = set()
        for original in pending:
            if self.stop_event.is_set():
                break
            job = self.store.get(original.id)
            if job.state not in PENDING:
                continue
            pool = "gpu" if job.spec.gpu else "cpu"
            if job.next_action_at > time.time():
                blocked_pools.add(pool)
                continue
            if self.inventory_at is None or self.inventory_error:
                self._hold(job, "Account discovery unavailable; waiting before new launches")
                continue
            if pool in blocked_pools or self._full(pool):
                self._hold(job, f"Waiting for {pool.upper()} capacity")
                continue
            if job.spec.gpu and (problem := self._gpu_quota_problem()):
                reason, error = problem
                self._hold(job, reason, error=error, next_action_at=time.time() + self.config.retry_seconds)
                blocked_pools.add(pool)
                continue
            if job.owner != self.config.owner:
                self.store.update(job.id, state="blocked", error="Job owner differs from worker account")
                continue
            if not self._prepare(job):
                if self.store.get(job.id).state in PENDING:
                    blocked_pools.add(pool)
                continue
            job = self.store.get(job.id)
            if job.state != "preparing":
                continue
            # Recheck occupancy after uploads; external changes can still race, and Kaggle is authoritative.
            self._refresh_inventory()
            if self.inventory_error or self._full(pool):
                continue
            if not self._push(job):
                blocked_pools.add(pool)

    def _prepare(self, job):
        job = self.store.update(
            job.id,
            expected=PENDING,
            state="preparing",
            error=None,
            wait_reason="Preparing private inputs",
        )
        if job is None:
            return False
        refs = dict(job.upload_refs)
        try:
            bundles = {"input:" + alias: bundle for alias, bundle in job.snapshot["inputs"].items()}
            if not job.snapshot["single_file"]:
                bundles["source"] = job.snapshot["source"]
            for key, bundle in bundles.items():
                if key in refs:
                    continue
                if self.store.get(job.id).state != "preparing":
                    return False
                self.store.heartbeat(state="running", stage="uploading", job_id=job.id)
                ref = self.backend.ensure_bundle(bundle)
                if ref is None:
                    self.store.update(
                        job.id,
                        expected={"preparing"},
                        upload_refs=refs,
                        wait_reason="Waiting for dataset processing",
                        next_action_at=time.time() + self.config.poll_seconds,
                    )
                    return False
                refs[key] = ref
                self.store.update(job.id, expected={"preparing"}, upload_refs=refs)
            for ref in job.spec.datasets:
                key = "dataset:" + ref
                if key not in refs:
                    refs[key] = self.backend.resolve_dataset(ref)
                    self.store.update(job.id, expected={"preparing"}, upload_refs=refs)
            return self.store.get(job.id).state == "preparing"
        except Exception as error:
            kind = getattr(error, "kind", "invalid")
            transient = kind in {"transient", "rate_limit", "capacity", "uncertain"}
            self.store.update(
                job.id,
                expected={"preparing"},
                state="queued" if transient else "blocked",
                error=safe_message(error),
                wait_reason="Upload retry pending"
                if transient
                else "Fix the reported issue, then retry this job",
                next_action_at=time.time() + self.config.retry_seconds,
                upload_refs=refs,
            )
            return False

    def _push(self, job):
        number = len(job.attempts) + 1
        name = re.sub(r"[^a-z0-9]+", "-", job.spec.name.lower()).strip("-")[:16] or "workload"
        ref = f"{job.owner}/kgr-{name}-{job.id[:12]}-a{number}"
        attempt = Attempt(number=number, ref=ref)
        job.attempts.append(attempt)
        # The stage is local. A crash before the atomic state update cannot have submitted anything.
        try:
            folder = prepare_kernel(job, self.config.state_dir)
        except Exception as error:
            self.store.update(job.id, expected={"preparing"}, state="blocked", error=safe_message(error))
            return True
        job = self.store.update(
            job.id,
            expected={"preparing"},
            state="submitting",
            wait_reason=None,
            error=None,
            attempts=job.attempts,
            next_action_at=0,
        )
        if job is None:
            return True
        try:
            result = self.backend.push(
                folder, timeout_seconds=job.spec.timeout_seconds, accelerator=job.spec.accelerator
            )
            if result["ref"].lower() != ref.lower():
                raise RemoteError(
                    "Kaggle returned an unexpected notebook identity; reconcile manually", "uncertain"
                )
        except RemoteError as error:
            attempt.error = safe_message(error)
            attempt.state = "rejected" if error.definitive else "uncertain"
            job.attempts[-1] = attempt
            retryable = error.kind in {"capacity", "quota", "rate_limit"}
            state = ("queued" if retryable else "blocked") if error.definitive else "submitting"
            self.store.update(
                job.id,
                state=state,
                attempts=job.attempts,
                error=safe_message(error),
                wait_reason="Waiting to retry rejected submission"
                if error.definitive
                else "Submission outcome uncertain; reconciling",
                next_action_at=time.time() + min(self.config.retry_seconds * 2 ** min(number - 1, 3), 600),
            )
            return not retryable
        except Exception as error:
            # No exception after an attempted mutation is safe to interpret as non-acceptance.
            attempt.state = "uncertain"
            job.attempts[-1] = attempt
            self.store.update(
                job.id,
                attempts=job.attempts,
                error=safe_message(error),
                wait_reason="Submission outcome uncertain; reconciling",
                next_action_at=time.time() + self.config.poll_seconds,
            )
            return False
        attempt.state = "accepted"
        attempt.version = result.get("version")
        job.attempts[-1] = attempt
        self.store.update(
            job.id,
            state="remote_queued",
            attempts=job.attempts,
            wait_reason=None,
            error=None,
            next_action_at=time.time() + self.config.poll_seconds,
        )
        return True

    def _downloads(self):
        for job_id, future in list(self.downloads.items()):
            if future.done():
                try:
                    future.result()
                except Exception as error:
                    logger.warning("Output collection for %s: %s", job_id, safe_message(error))
                del self.downloads[job_id]
        for job in self.store.list(TERMINAL):
            if (
                job.remote_ref
                and job.attempts[-1].state == "accepted"
                and job.spec.auto_download
                and job.download_state != "complete"
                and job.download_retry_at <= time.time()
                and job.id not in self.downloads
            ):
                if self.pool:
                    self.downloads[job.id] = self.pool.submit(
                        collect_outputs, self.store, self.backend, job.id
                    )
                else:
                    try:
                        collect_outputs(self.store, self.backend, job.id)
                    except Exception as error:
                        logger.warning("Output collection for %s: %s", job.id, safe_message(error))

"""Single dispatcher, persistent attempts, per-account capacity, independent artifact downloads."""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .models import ACTIVE, TERMINAL, Attempt, Config
from .providers import RemoteError, safe_message
from .security import redacted_env_record
from .store import Store, atomic_json, try_lock

logger = logging.getLogger(__name__)
PENDING = {"queued", "preparing"}
# Jobs without a possible remote run; they can change account.
MOVABLE = {"queued", "preparing", "blocked"}
REMOTE_STATES = {
    "queued": "remote_queued",
    "running": "running",
    "cancelling": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


def collect_outputs(store, provider, job_id):
    job = store.get(job_id)
    if not job.remote_ref or not job.terminal:
        raise ValueError("Outputs may be collected after a submitted run terminates")
    job.result_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (job.result_dir / ".download.lock").open("a+") as lock:
        if not try_lock(lock):
            return store.get(job_id)
        try:
            store.update(job_id, download_state="downloading", download_error=None)
            provider(job.attempts[-1].account).download(
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
                download_failures=job.download_failures + 1,
                # 1, 2, 4 ... minutes, then hourly; a permanent failure should not hammer the provider.
                download_retry_at=time.time() + min(60 * 2**job.download_failures, 3600),
            )
            raise


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


def occupancy(jobs, account, runs):
    """Runs holding an account's CPU and GPU slots: our outstanding attempts plus other known runs."""
    ours = [job for job in jobs if outstanding(job) and job.attempts[-1].account == account]
    refs = {job.remote_ref.lower() for job in ours}
    counts = {"cpu": sum(not job.spec.gpu for job in ours), "gpu": sum(job.spec.gpu for job in ours)}
    for ref, resource in runs.items():
        if ref.lower() not in refs:
            for kind in ("cpu", "gpu"):
                if resource in {kind, "unknown"}:
                    counts[kind] += 1
    return counts


def waiting_for_capacity(account, pool):
    return f"Waiting for {pool.upper()} capacity on {account}"


def place(store, job_id, account, reason):
    """Put a job that has no remote run on another account; None if it changed meanwhile.

    Requeueing also stops a preparation in progress, whose updates expect "preparing".
    """
    return store.update(
        job_id,
        expected=MOVABLE,
        account=account,
        state="queued",
        # Uploaded inputs belong to the previous account.
        upload_refs={},
        suggested_account=None,
        error=None,
        wait_reason=reason,
        next_action_at=0,
    )


def settled(job, *, downloads=True):
    """Nothing further happens without operator action, apart from download retries."""
    if job.state in {"blocked", "needs_attention"}:
        return True
    return job.terminal and (not downloads or job.download_state in {"complete", "disabled", "error"})


@dataclass
class Discovery:
    """What the worker last learned about one account's remote capacity."""

    runs: dict = field(default_factory=dict)
    checked_at: float | None = None
    error: str | None = None
    retry_at: float = 0
    gpu_seconds: float | None = None


class Worker:
    def __init__(self, config: Config, provider, store=None):
        """provider maps an account ID to its Provider."""
        self.config = config
        self.store = store or Store(config.state_dir)
        self.provider = provider
        self.stop_event = threading.Event()
        self.discovery = defaultdict(Discovery)
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
            self._dispatch(pending)
        self._downloads()
        atomic_json(
            self.config.state_dir / "accounts.json",
            {account: asdict(found) for account, found in self.discovery.items()},
        )
        self.store.heartbeat(state="running", stage="idle")

    def _poll(self, job):
        attempt = job.attempts[-1]
        now = time.time()
        try:
            status = self.provider(attempt.account).status(attempt.ref)
        except (RemoteError, ValueError) as error:  # ValueError: the account is no longer configured
            uncertain = attempt.state in {"submitting", "uncertain"}
            expired = now - attempt.started_at >= self.config.reconcile_seconds
            missing = getattr(error, "kind", None) == "missing"
            new_state = "needs_attention" if expired and (uncertain or missing) else job.state
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
        state = REMOTE_STATES.get(status["state"])
        if state is None:
            self.store.update(
                job.id,
                remote_state=status.get("detail"),
                last_polled_at=now,
                state="needs_attention"
                if now - attempt.started_at >= self.config.reconcile_seconds
                else job.state,
                wait_reason="Remote execution not confirmed; capacity remains reserved",
                next_action_at=now + self.config.poll_seconds,
            )
            return
        attempt.state = "accepted"
        changes = dict(
            state=state,
            remote_state=status.get("detail"),
            last_polled_at=now,
            attempts=job.attempts,
            error=safe_message(status["error"]) if status.get("error") else None,
            wait_reason=f"Cancellation requested on {attempt.account}"
            if status["state"] == "cancelling"
            else None,
            next_action_at=now + self.config.poll_seconds,
        )
        if state in TERMINAL:
            changes["finished_at"] = now
            self.discovery[attempt.account].runs.pop(attempt.ref.lower(), None)
        self.store.update(job.id, **changes)

    def _refresh_inventory(self, account):
        found = self.discovery[account]
        now = time.time()
        if now < found.retry_at:
            return
        if found.checked_at is not None and now - found.checked_at < self.config.discovery_seconds:
            return
        try:
            self.store.heartbeat(state="running", stage=f"discovering runs on {account}")
            found.runs = {ref.lower(): kind for ref, kind in self.provider(account).active_runs().items()}
            found.checked_at = now
            found.error = None
        except Exception as error:
            found.error = safe_message(error)
            found.retry_at = now + self.config.retry_seconds

    def _full(self, account, pool, *, reserve=False):
        count = occupancy(self.store.list(ACTIVE), account, self.discovery[account].runs)[pool]
        if reserve:
            # Jobs already preparing there take the next slots.
            count += sum(
                job.account == account and job.spec.gpu == (pool == "gpu")
                for job in self.store.list({"preparing"})
            )
        return count >= getattr(self.config.account(account), pool + "_limit")

    def _obstacle(self, account, pool, *, reserve=False):
        """Why no new run can start on the account now, as (reason, hold changes); None if one can."""
        self._refresh_inventory(account)
        found = self.discovery[account]
        if found.checked_at is None or found.error:
            return f"Run discovery on {account} unavailable; waiting before new launches", {}
        if self._full(account, pool, reserve=reserve):
            return waiting_for_capacity(account, pool), {}
        if pool == "gpu":
            retry = time.time() + self.config.retry_seconds
            try:
                gpu = self.provider(account).quota().get("gpu")
            except Exception as error:
                changes = dict(error=safe_message(error), next_action_at=retry)
                return f"GPU quota on {account} unavailable", changes
            found.gpu_seconds = gpu["available_seconds"] if gpu else 0
            if found.gpu_seconds <= 0:
                return f"Waiting for available GPU quota on {account}", dict(error=None, next_action_at=retry)
        return None

    def _alternative(self, job, pool, targets):
        """The first other account, in preference order, that can start the job now.

        targets caches each account's availability for this cycle. Work already preparing
        there counts, so a burst moves no more jobs than an account can start.
        """
        for account in self.config.accounts:
            key = (account.id, pool)
            if account.id == job.account:
                continue
            if key not in targets:
                targets[key] = self._obstacle(account.id, pool, reserve=True) is None
            if not targets[key]:
                continue
            try:
                self.provider(account.id).check(job.spec)
            except ValueError:
                continue
            return account.id
        return None

    def _hold(self, job, reason, suggested=None, **changes):
        """Keep a pending job queued, with a visible reason and an account that could start it now."""
        if changes or job.wait_reason != reason or job.suggested_account != suggested:
            self.store.update(
                job.id, expected=PENDING, wait_reason=reason, suggested_account=suggested, **changes
            )

    def _dispatch(self, pending):
        # Jobs wait in order within each account's resource pool; blocked records why.
        blocked, targets = {}, {}
        configured = {account.id for account in self.config.accounts}
        for original in pending:
            if self.stop_event.is_set():
                break
            job = self.store.get(original.id)
            if job.state not in PENDING:
                continue
            pool = "gpu" if job.spec.gpu else "cpu"
            if job.account not in configured:
                self.store.update(
                    job.id,
                    expected=PENDING,
                    state="blocked",
                    error=f"Account {job.account} is not configured; add it or move the job",
                )
                continue
            key = (job.account, pool)
            if job.next_action_at > time.time():
                blocked.setdefault(key, waiting_for_capacity(*key))
                if job.state != "queued":
                    continue
                # Backing off, e.g. after the provider rejected a launch for capacity or quota.
                problem = job.wait_reason or blocked[key], {}
            elif key in blocked:
                problem = blocked[key], {}
            elif problem := self._obstacle(job.account, pool):
                blocked[key] = problem[0]
            if problem:
                reason, changes = problem
                target = self._alternative(job, pool, targets) if self.config.failover != "off" else None
                # A job that started uploading stays, so jobs do not bounce between accounts.
                if not (target and self.config.failover == "auto" and job.state == "queued"):
                    self._hold(job, reason, target, **changes)
                    continue
                job = place(self.store, job.id, target, f"Moved from {job.account}: {reason}")
                targets.pop((target, pool))
                if job is None:
                    continue
                key = (job.account, pool)
            if not self._prepare(job):
                if self.store.get(job.id).state in PENDING:
                    blocked.setdefault(key, waiting_for_capacity(*key))
                continue
            job = self.store.get(job.id)
            if job.state != "preparing":
                continue
            # Recheck occupancy after uploads; external changes can still race; the provider is authoritative.
            self._refresh_inventory(job.account)
            if self.discovery[job.account].error or self._full(job.account, pool):
                continue
            if not self._submit(job):
                blocked.setdefault(key, waiting_for_capacity(*key))

    def _prepare(self, job):
        job = self.store.update(
            job.id,
            expected=PENDING,
            state="preparing",
            error=None,
            wait_reason="Preparing private inputs",
            suggested_account=None,
        )
        if job is None:
            return False
        provider = self.provider(job.account)
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
                ref = provider.ensure_bundle(bundle)
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
                    refs[key] = provider.resolve_dataset(ref)
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

    def _submit(self, job):
        provider = self.provider(job.account)
        number = len(job.attempts) + 1
        # Staging is local. A crash before the atomic state update cannot have submitted anything.
        try:
            ref = provider.stage(job, number)
        except Exception as error:
            self.store.update(job.id, expected={"preparing"}, state="blocked", error=safe_message(error))
            return True
        attempt = Attempt(number=number, account=job.account, ref=ref, url=provider.url(ref))
        job.attempts.append(attempt)
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
            result = provider.submit(job)
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

    def _collect(self, job_id):
        try:
            collect_outputs(self.store, self.provider, job_id)
        except Exception as error:
            logger.warning("Output collection for %s: %s", job_id, safe_message(error))

    def _downloads(self):
        self.downloads = {job_id: future for job_id, future in self.downloads.items() if not future.done()}
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
                    self.downloads[job.id] = self.pool.submit(self._collect, job.id)
                else:
                    self._collect(job.id)

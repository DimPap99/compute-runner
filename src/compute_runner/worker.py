"""Single dispatcher, persistent attempts, per-account capacity, independent artifact downloads."""

from __future__ import annotations

import logging
import signal
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bundle import plain_files, snapshot_bundle
from .models import ACTIVE, TERMINAL, Attempt, Config
from .providers import RemoteError, safe_message
from .results import JobOutputs, publish
from .security import redacted_env_record
from .store import Store, atomic_json

logger = logging.getLogger(__name__)
PENDING = {"queued", "preparing"}
# Jobs without a possible remote run; they can change account.
MOVABLE = {"queued", "preparing", "blocked"}
# Why preparation blocked a job, by error kind; the job's error has the details.
BLOCKED_REASONS = {
    "access": "Move the job or allow copying its dataset; see error",
    "dataset": "Check the dataset reference or its access with the user; see error",
}
REMOTE_STATES = {
    "queued": "remote_queued",
    "running": "running",
    "cancelling": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


def collect_outputs(store, provider, job_id, *, strict=False):
    """Download a terminated run's outputs into its run folder; one collector per job at a time."""
    job = store.get(job_id)
    if not job.remote_ref or not job.terminal:
        raise ValueError("Outputs may be collected after a submitted run terminates")
    with store.download_claim(job_id) as claimed:
        if not claimed:
            return store.get(job_id)
        try:
            store.update(job_id, download_state="downloading", download_error=None)
            provider(job.attempts[-1].account).download(job.remote_ref, JobOutputs(store, job, strict=strict))
            updated = store.update(job_id, download_state="complete", download_error=None)
            if job.run is None:
                # Run folders get job.json from the worker instead.
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


def queued_behind(account):
    return f"Queued behind a job preparing on {account}"


def place(store, job_id, account, reason, *, transfer=None):
    """Put a job that has no remote run on another account; None if it changed meanwhile.

    Requeueing also stops a preparation in progress, whose updates expect "preparing".
    transfer, when given, sets whether its datasets may be copied there.
    """
    changes = {} if transfer is None else {"transfer": transfer}
    return store.update(
        job_id,
        expected=MOVABLE,
        account=account,
        state="queued",
        # Uploads and attached datasets belong to the previous account; copies stay cached locally.
        upload_refs={},
        transfers={},
        suggested_account=None,
        suggested_transfer=False,
        error=None,
        wait_reason=reason,
        next_action_at=0,
        **changes,
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
        # (account, provider, dataset ref) -> (readable, checked_at)
        self.readable = {}

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
        publish(self.store)
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
            # expected: an operator may have resolved or cancelled the job during the remote call.
            self.store.update(
                job.id,
                expected={job.state},
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
                expected={job.state},
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
        self.store.update(job.id, expected={job.state}, **changes)

    def _refresh_inventory(self, account):
        found = self.discovery[account]
        now = time.time()
        if now < found.retry_at:
            return
        if found.checked_at is not None and now - found.checked_at < self.config.discovery_seconds:
            return
        try:
            self.store.heartbeat(state="running", stage=f"discovering runs on {account}")
            provider = self.provider(account)
            found.runs = {ref.lower(): kind for ref, kind in provider.active_runs().items()}
            # Checked with the runs, not per launch: failover weighs every account each cycle.
            gpu = provider.quota().get("gpu")
            found.gpu_seconds = gpu["available_seconds"] if gpu else 0
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
        """Why no new run can start on the account now; None if one can."""
        self._refresh_inventory(account)
        found = self.discovery[account]
        if found.checked_at is None or found.error:
            return f"Run discovery on {account} unavailable; waiting before new launches"
        if self._full(account, pool, reserve=reserve):
            return waiting_for_capacity(account, pool)
        if pool == "gpu" and found.gpu_seconds <= 0:
            return f"Waiting for available GPU quota on {account}"
        return None

    def _datasets(self, job):
        """The provider datasets a job attaches: {upload key: (provider, reference)}."""
        provider = self.config.account(job.account).provider
        found = {"dataset:" + ref: (provider, ref) for ref in job.spec.datasets}
        return found | {"input:" + alias: ref for alias, ref in job.spec.dataset_inputs().items()}

    def _readable(self, account, provider, ref):
        """Whether an account can read a dataset, checked at most discovery_seconds ago; None if unknown."""
        if self.config.account(account).provider != provider:
            return False
        key = (account, provider, ref)
        cached = self.readable.get(key)
        if cached and time.time() - cached[1] < self.config.discovery_seconds:
            return cached[0]
        try:
            readable = self.provider(account).resolve_dataset(ref) is not None
        except (RemoteError, ValueError):
            return None
        self.readable[key] = (readable, time.time())
        return readable

    def _unreadable(self, job, account):
        """Upload keys of the job's datasets the account cannot read; None if that is unknown now."""
        missing = []
        for key, (provider, ref) in self._datasets(job).items():
            readable = self._readable(account, provider, ref)
            if readable is None:
                return None
            if not readable:
                missing.append(key)
        return missing

    def _alternative(self, job, pool, targets):
        """The first other account, in preference order, that can start the job now: (account, copy).

        copy is true when that account cannot read some of the job's datasets, so they would be
        copied there; an account that reads them all wins over an earlier one that needs copies.
        Only aliased inputs can be copied. (None, False) when no account fits.

        targets caches each account's availability for this cycle. Work already preparing
        there counts, so a burst moves no more jobs than an account can start. An account
        that rejected one of the job's launches is skipped, so a job cannot bounce between
        two full accounts.
        """
        rejected = {attempt.account for attempt in job.attempts if attempt.state == "rejected"}
        fallback = None
        for account in self.config.accounts:
            key = (account.id, pool)
            if account.id == job.account or account.id in rejected:
                continue
            if key not in targets:
                targets[key] = self._obstacle(account.id, pool, reserve=True) is None
            if not targets[key]:
                continue
            try:
                self.provider(account.id).check(job.spec)
            except ValueError:
                continue
            missing = self._unreadable(job, account.id)
            if missing == []:
                return account.id, False
            if missing and fallback is None and all(key.startswith("input:") for key in missing):
                fallback = account.id
        return fallback, fallback is not None

    def _hold(self, job, reason, suggested=None, copy=False):
        """Keep a pending job waiting, with a visible reason and an account that could start it now."""
        if job.wait_reason != reason or job.suggested_account != suggested or job.suggested_transfer != copy:
            self.store.update(
                job.id,
                expected=PENDING,
                wait_reason=reason,
                suggested_account=suggested,
                suggested_transfer=copy,
            )

    def _dispatch(self, pending):
        # Jobs start in order within each account's resource pool. waiting[key] is why later jobs
        # there wait, and whether the account itself cannot start work now, which allows failover.
        waiting, targets = {}, {}
        configured = {account.id for account in self.config.accounts}
        for original in pending:
            if self.stop_event.is_set():
                break
            job = self.store.get(original.id)
            if job.state not in PENDING:
                continue
            pool = "gpu" if job.spec.gpu else "cpu"
            if job.account not in configured:
                # Usually an account added after this worker started; it runs once the worker restarts.
                self._hold(
                    job, f"Account {job.account} is unknown to the running worker; restart it or move the job"
                )
                continue
            key = (job.account, pool)
            if job.next_action_at > time.time():
                if job.state == "preparing":
                    # Its uploads wait for a retry or for provider processing; later jobs keep their place.
                    waiting.setdefault(key, (queued_behind(job.account), False))
                    continue
                # Queued jobs back off only after the provider rejected a launch for capacity or quota.
                waiting.setdefault(key, (waiting_for_capacity(*key), True))
                problem = job.wait_reason or waiting[key][0], True
            elif key in waiting:
                problem = waiting[key]
            elif reason := self._obstacle(job.account, pool):
                problem = waiting[key] = reason, True
            else:
                problem = None
            if problem:
                reason, account_unavailable = problem
                target, copy = None, False
                if account_unavailable and self.config.failover != "off":
                    target, copy = self._alternative(job, pool, targets)
                allowed = target and (not copy or self.config.transfer or job.transfer)
                # Only queued jobs move by themselves; one that is preparing keeps its uploads.
                if not (allowed and self.config.failover == "auto" and job.state == "queued"):
                    self._hold(job, reason, target, copy)
                    continue
                job = place(self.store, job.id, target, f"Moved from {job.account}: {reason}")
                if job is None:
                    continue
                key = (job.account, pool)
            # The job takes a slot on its account, so that account's cached availability is stale.
            targets.pop(key, None)
            if not self._prepare(job):
                if self.store.get(job.id).state == "preparing":
                    waiting.setdefault(key, (queued_behind(job.account), False))
                continue
            job = self.store.get(job.id)
            if job.state != "preparing":
                continue
            # Recheck occupancy after uploads; external changes can still race; the provider is authoritative.
            self._refresh_inventory(job.account)
            if self.discovery[job.account].error or self._full(job.account, pool):
                continue
            if not self._submit(job):
                waiting.setdefault(key, (waiting_for_capacity(*key), True))

    def _prepare(self, job):
        job = self.store.update(
            job.id,
            expected=PENDING,
            state="preparing",
            error=None,
            wait_reason="Preparing private inputs",
            suggested_account=None,
            suggested_transfer=False,
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
            for key, (dataset_provider, ref) in self._datasets(job).items():
                if key in refs:
                    continue
                if self.store.get(job.id).state != "preparing":
                    return False
                attached = self._attach(job, key, dataset_provider, ref)
                if attached is None:
                    self.store.update(
                        job.id,
                        expected={"preparing"},
                        upload_refs=refs,
                        wait_reason="Waiting for dataset processing",
                        next_action_at=time.time() + self.config.poll_seconds,
                    )
                    return False
                refs[key] = attached
                self.store.update(job.id, expected={"preparing"}, upload_refs=refs)
            return self.store.get(job.id).state == "preparing"
        except Exception as error:
            kind = getattr(error, "kind", "invalid")
            transient = kind in {"transient", "rate_limit", "capacity", "uncertain"}
            self.store.update(
                job.id,
                expected={"preparing"},
                # Still preparing: completed uploads stay, and the job keeps its account and place.
                state="preparing" if transient else "blocked",
                error=safe_message(error),
                wait_reason="Upload retry pending"
                if transient
                else BLOCKED_REASONS.get(kind, "Fix the reported issue, then retry this job"),
                next_action_at=time.time() + self.config.retry_seconds,
                upload_refs=refs,
            )
            return False

    def _attach(self, job, key, dataset_provider, ref):
        """What to attach for one dataset: pinned on the job's account, or an uploaded copy.

        None while a copy is still processing. A dataset that no configured account can find
        blocks the job before anything runs, so a wrong reference is fixed instead of launched.
        """
        account = self.config.account(job.account)
        provider = self.provider(account.id)
        alias = key.removeprefix("input:") if key.startswith("input:") else None
        copied = job.transfers.get(alias)
        if copied is None:
            if account.provider == dataset_provider and (pinned := provider.resolve_dataset(ref)):
                return pinned
            reader = self._reader(job, dataset_provider, ref)
            if reader is None:
                raise RemoteError(
                    f"No connected account can find dataset {ref}. Check the reference "
                    "(OWNER/SLUG or OWNER/SLUG/VERSION) and that one of the accounts may read it; "
                    "then submit the corrected workload, or retry this job if only access changed",
                    "dataset",
                    definitive=True,
                )
            owner, source = reader
            if alias is None:
                raise RemoteError(
                    f"{job.account} cannot read dataset {ref}, but {owner} can. Move the job to {owner}, "
                    "or list the dataset under inputs with an alias so it can be copied",
                    "access",
                    definitive=True,
                )
            if not (self.config.transfer or job.transfer):
                raise RemoteError(
                    f"{job.account} cannot read dataset {ref}, but {owner} can. Move the job to {owner}, "
                    f"or allow a copy: compute-runner agent move {job.id} --account {job.account} --transfer",
                    "access",
                    definitive=True,
                )
            copied = self._copy(owner, source)
            job.transfers[alias] = copied
            self.store.update(job.id, expected={"preparing"}, transfers=job.transfers)
        self.store.heartbeat(state="running", stage="uploading", job_id=job.id)
        return provider.ensure_bundle(copied)

    def _reader(self, job, dataset_provider, ref):
        """Another configured account that can read a dataset: (account, pinned ref), or None.

        An account that could not be asked for a passing reason makes "None" uncertain, so that
        raises a transient error instead. One with definitive errors, or with credentials that do
        not work (missing, or for another user), cannot read it.
        """
        unanswered = None
        for account in self.config.accounts:
            if account.id == job.account or account.provider != dataset_provider:
                continue
            try:
                pinned = self.provider(account.id).resolve_dataset(ref)
            except RemoteError as error:
                if not error.definitive and error.kind != "auth":
                    unanswered = unanswered or (account.id, error)
                continue
            except ValueError:
                continue
            if pinned:
                return account.id, pinned
        if unanswered:
            account_id, error = unanswered
            raise RemoteError(f"Could not check whether {account_id} can read dataset {ref}: {error}")
        return None

    def _copy(self, account, ref):
        """Bundle a dataset version read through account, once, like a local input."""
        source = f"{self.config.account(account).provider}:{ref}"
        bundles = self.config.state_dir / "bundles"
        saved = self.store.dataset_copy(source)
        if saved is None or not (bundles / saved["digest"] / "payload.zip").is_file():
            self.store.heartbeat(state="running", stage=f"copying dataset {ref}")
            with tempfile.TemporaryDirectory(prefix=".dataset-", dir=self.config.state_dir) as folder:
                self.provider(account).fetch_dataset(ref, Path(folder))
                # Copied as published: the owner's files are already on the provider, so not screened.
                bundle = snapshot_bundle(Path(folder), plain_files(Path(folder)), bundles, screen=False)
            saved = dict(digest=bundle["digest"], bytes=bundle["bytes"])
            self.store.save_dataset_copy(source, **saved)
        return dict(source=source, **saved)

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
            collect_outputs(self.store, self.provider, job_id, strict=self.config.strict)
        except Exception as error:
            logger.warning("Output collection for %s: %s", job_id, safe_message(error))

    def _downloads(self):
        self.downloads = {job_id: future for job_id, future in self.downloads.items() if not future.done()}
        # Downloads for a removed account wait until it is added again.
        configured = {account.id for account in self.config.accounts}
        for job in self.store.list(TERMINAL):
            if (
                job.remote_ref
                and job.attempts[-1].state == "accepted"
                and job.attempts[-1].account in configured
                and job.spec.auto_download
                and job.download_state != "complete"
                and job.download_retry_at <= time.time()
                and job.id not in self.downloads
            ):
                if self.pool:
                    self.downloads[job.id] = self.pool.submit(self._collect, job.id)
                else:
                    self._collect(job.id)

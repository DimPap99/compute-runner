"""Placing pending jobs: in order within each account's pool, moving them when failover allows."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..models import ACTIVE, PENDING, JobRecord, Pool
from .capacity import Capacity, Discoverer, waiting_for_capacity
from .context import Component
from .datasets import DatasetAccess
from .preparation import Preparer
from .submission import Submitter


def queued_behind(account: str) -> str:
    return f"Queued behind a job preparing on {account}"


@dataclass(frozen=True)
class Wait:
    """Why jobs wait on one account's pool, and whether the account itself cannot start work now.

    Only an account that cannot start work lets its jobs fail over to another.
    """

    reason: str
    account_unavailable: bool


@dataclass
class Cycle:
    """What one dispatch pass learned: why each (account, pool) waits, and which accounts could start work."""

    waiting: dict[tuple[str, Pool], Wait] = field(default_factory=dict)
    targets: dict[tuple[str, Pool], bool] = field(default_factory=dict)


class Dispatcher(Component):
    def __init__(
        self,
        context,
        *,
        discoverer: Discoverer,
        datasets: DatasetAccess,
        preparer: Preparer,
        submitter: Submitter,
        stop: threading.Event,
    ):
        super().__init__(context)
        self.discoverer = discoverer
        self.datasets = datasets
        self.preparer = preparer
        self.submitter = submitter
        self.stop = stop

    def dispatch(self, pending: list[JobRecord]) -> None:
        cycle = Cycle()
        configured = {account.id for account in self.config.accounts}
        for original in pending:
            if self.stop.is_set():
                break
            job = self.store.get(original.id)
            if job.state not in PENDING:
                continue
            if job.account not in configured:
                # Usually an account added after this worker started; it runs once the worker restarts.
                self._hold(
                    job, f"Account {job.account} is unknown to the running worker; restart it or move the job"
                )
                continue
            self._place(job, cycle)

    def _place(self, job: JobRecord, cycle: Cycle) -> None:
        key = (job.account, job.pool)
        backing_off = job.next_action_at > time.time()
        if backing_off and job.state == "preparing":
            # Its uploads wait for a retry or for provider processing; later jobs keep their place.
            cycle.waiting.setdefault(key, Wait(queued_behind(job.account), False))
            return
        wait = self._wait(job, key, cycle, backing_off)
        if wait:
            job = self._fail_over(job, wait, cycle)
            if job is None:
                return
        self._start(job, cycle)

    def _wait(self, job: JobRecord, key, cycle: Cycle, backing_off: bool) -> Wait | None:
        """Why the job cannot start on its account now; None if it can."""
        if backing_off:
            # Queued jobs back off only after the provider rejected a launch for capacity or quota.
            cycle.waiting.setdefault(key, Wait(waiting_for_capacity(*key), True))
            return Wait(job.wait_reason or cycle.waiting[key].reason, True)
        if key not in cycle.waiting and (reason := self.obstacle(*key)):
            cycle.waiting[key] = Wait(reason, True)
        return cycle.waiting.get(key)

    def _fail_over(self, job: JobRecord, wait: Wait, cycle: Cycle) -> JobRecord | None:
        """The job moved to an account that can start it, or None once it is left waiting."""
        target, copy = None, False
        if wait.account_unavailable and self.config.failover != "off":
            target, copy = self._alternative(job, cycle.targets)
        allowed = target and (not copy or self.config.transfer or job.transfer)
        # Only queued jobs move by themselves; one that is preparing keeps its uploads.
        if not (allowed and self.config.failover == "auto" and job.state == "queued"):
            self._hold(job, wait.reason, target, copy)
            return None
        return self.store.place(job.id, target, f"Moved from {job.account}: {wait.reason}")

    def _start(self, job: JobRecord, cycle: Cycle) -> None:
        key = (job.account, job.pool)
        # The job takes a slot on its account, so that account's cached availability is stale.
        cycle.targets.pop(key, None)
        if not self.preparer.prepare(job):
            if self.store.get(job.id).state == "preparing":
                cycle.waiting.setdefault(key, Wait(queued_behind(job.account), False))
            return
        job = self.store.get(job.id)
        if job.state != "preparing":
            return
        # Recheck occupancy after uploads; external changes can still race; the provider is authoritative.
        if self.discoverer.refresh(job.account).error or self._capacity().full(*key):
            return
        if not self.submitter.submit(job):
            cycle.waiting.setdefault(key, Wait(waiting_for_capacity(*key), True))

    def _capacity(self) -> Capacity:
        return Capacity(self.config, self.discoverer.found, self.store.list(ACTIVE))

    def obstacle(self, account: str, pool: Pool, *, reserve: bool = False) -> str | None:
        """Why no new run can start on the account now; None if one can.

        reserve counts jobs already preparing there, which take the next slots.
        """
        self.discoverer.refresh(account)
        preparing = self.store.list({"preparing"}) if reserve else []
        reserved = sum(job.account == account and job.pool == pool for job in preparing)
        return self._capacity().obstacle(account, pool, reserved=reserved)

    def _alternative(self, job: JobRecord, targets: dict) -> tuple[str | None, bool]:
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
            if (
                account.id == job.account
                or account.id in rejected
                or not self._available(account.id, job, targets)
            ):
                continue
            missing = self.datasets.unreadable(job, account.id)
            if missing == []:
                return account.id, False
            if missing and fallback is None and all(key.startswith("input:") for key in missing):
                fallback = account.id
        return fallback, fallback is not None

    def _available(self, account: str, job: JobRecord, targets: dict) -> bool:
        """Whether the account could start the job now: free there, and able to run its spec."""
        key = (account, job.pool)
        if key not in targets:
            targets[key] = self.obstacle(account, job.pool, reserve=True) is None
        if not targets[key]:
            return False
        try:
            self.provider(account).check(job.spec)
        except ValueError:
            return False
        return True

    def _hold(self, job: JobRecord, reason: str, suggested: str | None = None, copy: bool = False) -> None:
        """Keep a pending job waiting, with a visible reason and an account that could start it now."""
        if job.wait_reason != reason or job.suggested_account != suggested or job.suggested_transfer != copy:
            self.store.update(
                job.id,
                expected=PENDING,
                wait_reason=reason,
                suggested_account=suggested,
                suggested_transfer=copy,
            )

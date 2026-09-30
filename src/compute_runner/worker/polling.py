"""Following a submitted run: its remote state becomes the job's, or the job waits to reconcile."""

from __future__ import annotations

import time

from ..models import TERMINAL, Attempt, JobRecord
from ..providers import RemoteError, safe_message
from .capacity import Discoverer
from .context import Component

# Job states for the providers' run states.
REMOTE_STATES = {
    "queued": "remote_queued",
    "running": "running",
    "cancelling": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class Poller(Component):
    def __init__(self, context, discoverer: Discoverer):
        super().__init__(context)
        self.discoverer = discoverer

    def poll(self, job: JobRecord) -> None:
        attempt = job.attempts[-1]
        now = time.time()
        try:
            status = self.provider(attempt.account).status(attempt.ref)
        except (RemoteError, ValueError) as error:  # ValueError: the account is no longer configured
            self._unreachable(job, attempt, error, now)
            return
        state = REMOTE_STATES.get(status["state"])
        if state is None:
            self._unconfirmed(job, attempt, status, now)
        else:
            self._observed(job, attempt, status, state, now)

    def _expired(self, attempt: Attempt, now: float) -> bool:
        return now - attempt.started_at >= self.config.reconcile_seconds

    def _unreachable(self, job: JobRecord, attempt: Attempt, error: Exception, now: float) -> None:
        """The provider did not answer: keep the slot reserved, and ask for attention once it is too long."""
        uncertain = attempt.state in {"submitting", "uncertain"}
        missing = getattr(error, "kind", None) == "missing"
        if uncertain:
            attempt.state = "uncertain"
        # expected: an operator may have resolved or cancelled the job during the remote call.
        self.store.update(
            job.id,
            expected={job.state},
            state="needs_attention" if self._expired(attempt, now) and (uncertain or missing) else job.state,
            attempts=job.attempts,
            error=safe_message(error),
            wait_reason="Reconciling submission; no duplicate will launch"
            if uncertain
            else "Remote status unavailable; capacity remains reserved",
            last_polled_at=now,
            next_action_at=now + self.config.poll_seconds,
        )

    def _unconfirmed(self, job: JobRecord, attempt: Attempt, status: dict, now: float) -> None:
        """The provider answered with a state it does not map to one of ours."""
        self.store.update(
            job.id,
            expected={job.state},
            remote_state=status.get("detail"),
            last_polled_at=now,
            state="needs_attention" if self._expired(attempt, now) else job.state,
            wait_reason="Remote execution not confirmed; capacity remains reserved",
            next_action_at=now + self.config.poll_seconds,
        )

    def _observed(self, job: JobRecord, attempt: Attempt, status: dict, state: str, now: float) -> None:
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
            self.discoverer.forget(attempt.account, attempt.ref)
        self.store.update(job.id, expected={job.state}, **changes)

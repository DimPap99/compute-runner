"""Launching a prepared job as a new attempt, recorded before the remote request is made."""

from __future__ import annotations

import time

from ..models import Attempt, JobRecord
from ..providers import RemoteError, safe_message
from .context import Component

# Rejections after which the job queues again for another launch.
RETRYABLE = {"capacity", "quota", "rate_limit", "transient"}


class Submitter(Component):
    def submit(self, job: JobRecord) -> bool:
        """Launch the job; False when the provider did not take it for a passing reason.

        False tells the dispatcher that later jobs on the account should wait too.
        """
        provider = self.provider(job.account)
        number = len(job.attempts) + 1
        # Staging is local. A crash before the atomic state update cannot have submitted anything.
        try:
            ref = provider.stage(job, number)
        except Exception as error:
            self.store.update(job.id, expected={"preparing"}, state="blocked", error=safe_message(error))
            return True
        attempt = Attempt(number=number, account=job.account, ref=ref, url=provider.url(ref))
        job = self._record(job, attempt)
        if job is None:
            return True
        try:
            result = provider.submit(job)
        except RemoteError as error:
            return self._rejected(job, attempt, error)
        except Exception as error:
            # No exception after an attempted mutation is safe to interpret as non-acceptance.
            self._uncertain(job, attempt, error)
            return False
        self._accepted(job, attempt, result)
        return True

    def _record(self, job: JobRecord, attempt: Attempt) -> JobRecord | None:
        """Save the attempt before any remote request, so a crash can always be reconciled."""
        job.attempts.append(attempt)
        return self.store.update(
            job.id,
            expected={"preparing"},
            state="submitting",
            wait_reason=None,
            error=None,
            attempts=job.attempts,
            next_action_at=0,
        )

    def _rejected(self, job: JobRecord, attempt: Attempt, error: RemoteError) -> bool:
        attempt.error = safe_message(error)
        attempt.state = "rejected" if error.definitive else "uncertain"
        job.attempts[-1] = attempt
        # transient: a definitive failure that says nothing about the launch, such as a lost upload.
        retryable = error.kind in RETRYABLE
        state = ("queued" if retryable else "blocked") if error.definitive else "submitting"
        backoff = min(self.config.retry_seconds * 2 ** min(attempt.number - 1, 3), 600)
        self.store.update(
            job.id,
            state=state,
            attempts=job.attempts,
            error=safe_message(error),
            wait_reason="Waiting to retry rejected submission"
            if error.definitive
            else "Submission outcome uncertain; reconciling",
            next_action_at=time.time() + backoff,
        )
        return not retryable

    def _uncertain(self, job: JobRecord, attempt: Attempt, error: Exception) -> None:
        attempt.state = "uncertain"
        job.attempts[-1] = attempt
        self.store.update(
            job.id,
            attempts=job.attempts,
            error=safe_message(error),
            wait_reason="Submission outcome uncertain; reconciling",
            next_action_at=time.time() + self.config.poll_seconds,
        )

    def _accepted(self, job: JobRecord, attempt: Attempt, result: dict) -> None:
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

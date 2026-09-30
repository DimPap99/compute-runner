"""Collecting finished runs' outputs, independently of scheduling, with growing retry delays."""

from __future__ import annotations

import logging
import time

from ..models import COLLECTED, JobRecord
from ..providers import safe_message
from ..results import JobOutputs
from ..security import redacted_env_record
from ..store import atomic_json
from .context import Component

logger = logging.getLogger(__name__)


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
                    job.result_dir / "provenance.json", redacted_env_record(updated.model_dump(mode="json"))
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


class DownloadScheduler(Component):
    """Starts output collection for finished runs, in a thread pool while the worker runs."""

    def __init__(self, context):
        super().__init__(context)
        self.pool = None
        self.running = {}

    def schedule(self) -> None:
        self.running = {job_id: future for job_id, future in self.running.items() if not future.done()}
        # Downloads for a removed account wait until it is added again.
        configured = {account.id for account in self.config.accounts}
        for job in self.store.uncollected():
            if self._due(job, configured):
                if self.pool:
                    self.running[job.id] = self.pool.submit(self._collect, job.id)
                else:
                    self._collect(job.id)

    def _due(self, job: JobRecord, configured: set[str]) -> bool:
        return bool(
            job.remote_ref
            and job.attempts[-1].state == "accepted"
            and job.attempts[-1].account in configured
            and job.spec.auto_download
            # disabled: cancelled before it started, so there is nothing to collect.
            and job.download_state not in COLLECTED
            and job.download_retry_at <= time.time()
            and job.id not in self.running
        )

    def _collect(self, job_id: str) -> None:
        try:
            collect_outputs(self.store, self.provider, job_id, strict=self.config.strict)
        except Exception as error:
            logger.warning("Output collection for %s: %s", job_id, safe_message(error))

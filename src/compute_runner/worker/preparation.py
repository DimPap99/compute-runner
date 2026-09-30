"""Preparing a job for launch: its bundles uploaded and its datasets attached on its account."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

from ..models import PENDING, JobRecord
from ..providers import safe_message
from .context import Component
from .datasets import DatasetAccess

# Why preparation blocked a job, by error kind; the job's error has the details.
BLOCKED_REASONS = {
    "access": "Move the job or allow copying its dataset; see error",
    "dataset": "Check the dataset reference or its access with the user; see error",
}
# Error kinds after which preparation retries later, keeping the job's account and place.
TRANSIENT = {"transient", "rate_limit", "capacity", "uncertain"}


class Preparer(Component):
    def __init__(self, context, datasets: DatasetAccess):
        super().__init__(context)
        self.datasets = datasets

    def prepare(self, job: JobRecord) -> bool:
        """Whether the job is prepared and still preparing, so it can launch now.

        Completed uploads are kept on the job (upload_refs), so a later cycle continues where
        this one stopped: while the provider processes an upload, or after a passing failure.
        """
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
        refs = dict(job.upload_refs)
        try:
            return self._obtain_all(job, refs) and self._preparing(job)
        except Exception as error:
            self._fail(job, refs, error)
            return False

    def _obtain_all(self, job: JobRecord, refs: dict) -> bool:
        provider = self.provider(job.account)
        steps = [
            (key, partial(self._upload, provider, job, bundle))
            for key, bundle in provider.bundles_for(job).items()
        ]
        steps += [
            (key, partial(self.datasets.attach, job, key, kind, ref))
            for key, (kind, ref) in self.datasets.attached(job).items()
        ]
        return all(self._obtain(job, refs, key, fetch) for key, fetch in steps)

    def _obtain(self, job: JobRecord, refs: dict, key: str, fetch: Callable[[], str | None]) -> bool:
        """Get one upload key's reference unless the job has it; False when preparation must pause."""
        if key in refs:
            return True
        if not self._preparing(job):
            return False
        ref = fetch()
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
        return True

    def _upload(self, provider, job: JobRecord, bundle: dict) -> str | None:
        self.heartbeat(stage="uploading", job_id=job.id)
        return provider.ensure_bundle(bundle)

    def _preparing(self, job: JobRecord) -> bool:
        """Whether nobody moved or cancelled the job meanwhile."""
        return self.store.get(job.id).state == "preparing"

    def _fail(self, job: JobRecord, refs: dict, error: Exception) -> None:
        kind = getattr(error, "kind", "invalid")
        transient = kind in TRANSIENT
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

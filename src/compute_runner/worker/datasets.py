"""Provider datasets a job attaches: attached where its account can read them, copied where allowed."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ..bundle import plain_files, snapshot_bundle
from ..models import JobRecord
from ..providers import RemoteError
from .context import Component

# How each provider's dataset references look, for errors that ask the user to check one.
REFERENCE_FORMS = {
    "kaggle": "OWNER/SLUG or OWNER/SLUG/VERSION",
    "ssh": "MACHINE:/ABSOLUTE/PATH on that machine",
}


class DatasetAccess(Component):
    def __init__(self, context):
        super().__init__(context)
        # (account, provider, dataset ref) -> (readable, checked_at)
        self._readable = {}

    def attached(self, job: JobRecord) -> dict[str, tuple[str, str]]:
        """The provider datasets a job attaches: {upload key: (provider, reference)}."""
        provider = self.config.account(job.account).provider
        found = {"dataset:" + ref: (provider, ref) for ref in job.spec.datasets}
        return found | {"input:" + alias: ref for alias, ref in job.spec.dataset_inputs().items()}

    def readable(self, account: str, provider: str, ref: str) -> bool | None:
        """Whether an account can read a dataset, checked at most discovery_seconds ago; None if unknown."""
        if self.config.account(account).provider != provider:
            return False
        key = (account, provider, ref)
        cached = self._readable.get(key)
        if cached and time.time() - cached[1] < self.config.discovery_seconds:
            return cached[0]
        try:
            readable = self.provider(account).resolve_dataset(ref) is not None
        except (RemoteError, ValueError):
            return None
        self._readable[key] = (readable, time.time())
        return readable

    def unreadable(self, job: JobRecord, account: str) -> list[str] | None:
        """Upload keys of the job's datasets the account cannot read; None if that is unknown now."""
        missing = []
        for key, (provider, ref) in self.attached(job).items():
            readable = self.readable(account, provider, ref)
            if readable is None:
                return None
            if not readable:
                missing.append(key)
        return missing

    def attach(self, job: JobRecord, key: str, dataset_provider: str, ref: str) -> str | None:
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
            copied = self._copy_for(job, alias, dataset_provider, ref)
            job.transfers[alias] = copied
            self.store.update(job.id, expected={"preparing"}, transfers=job.transfers)
        self.heartbeat(stage="uploading", job_id=job.id)
        return provider.ensure_bundle(copied)

    def _copy_for(self, job: JobRecord, alias: str | None, dataset_provider: str, ref: str) -> dict:
        """A local copy of a dataset the job's account cannot read, if another can and copying is allowed."""
        reader = self._reader(job, dataset_provider, ref)
        if reader is None:
            raise RemoteError(
                f"No connected account can find dataset {ref}. Check the reference "
                f"({REFERENCE_FORMS.get(dataset_provider, 'as the provider names it')}) and that one "
                "of the accounts may read it; then submit the corrected workload, or retry this job "
                "if only access changed",
                "dataset",
                definitive=True,
            )
        owner, source = reader
        refused = f"{job.account} cannot read dataset {ref}, but {owner} can. Move the job to {owner}, "
        if alias is None:
            raise RemoteError(
                refused + "or list the dataset under inputs with an alias so it can be copied",
                "access",
                definitive=True,
            )
        if not (self.config.transfer or job.transfer):
            raise RemoteError(
                refused
                + f"or allow a copy: compute-runner agent move {job.id} --account {job.account} --transfer",
                "access",
                definitive=True,
            )
        return self._copy(owner, source)

    def _reader(self, job: JobRecord, dataset_provider: str, ref: str) -> tuple[str, str] | None:
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

    def _copy(self, account: str, ref: str) -> dict:
        """Bundle a dataset version read through account, once, like a local input."""
        source = f"{self.config.account(account).provider}:{ref}"
        bundles = self.config.state_dir / "bundles"
        saved = self.store.dataset_copy(source)
        if saved is None or not (bundles / saved["digest"] / "payload.zip").is_file():
            self.heartbeat(stage=f"copying dataset {ref}")
            with tempfile.TemporaryDirectory(prefix=".dataset-", dir=self.config.state_dir) as folder:
                self.provider(account).fetch_dataset(ref, Path(folder))
                files = plain_files(Path(folder), allow_bundle_manifest=True)
                # Copied as published: the owner's files are already on the provider, so not screened.
                bundle = snapshot_bundle(Path(folder), files, bundles, screen=False)
            saved = dict(digest=bundle["digest"], bytes=bundle["bytes"])
            self.store.save_dataset_copy(source, **saved)
        return dict(source=source, **saved)

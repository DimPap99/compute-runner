"""What this runner left behind, on each account and in the state directory, and what may go.

Only runner-made things are considered: launch notebooks and run folders, bundle datasets and
bundles, the state directory's snapshots, staging folders, upload folders and log copies, and the
queue's job records. Results folders are the user's and never touched. An item is reclaimable only
when every job that uses it is finished, its outputs are downloaded (or not wanted), and it
finished before the cutoff. Remote items this queue does not know are reported as unknown and
never deleted. Local snapshots stay unless include_snapshots, since retry and continue need them.

A job's record goes last, and only when nothing remote would be left without it: after a cleanup
that checked every account found each remote item of the job reclaimable, and deleted them.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, astuple, dataclass, field
from pathlib import Path

from .bundle import PAYLOAD
from .models import COLLECTED, JobRecord
from .providers import Artifact, safe_message, short
from .runtime import tree_size

RECLAIMABLE, KEPT, UNKNOWN = "reclaimable", "kept", "unknown"
# State directory folders a cleanup looks in, and what their entries are.
LOCAL_KINDS = {"bundles": "snapshot", "uploads": "upload", "jobs": "staging", "logs": "log"}
# A job's record in the queue, and the unpacked copy of a bundle that versions before 0.6 kept.
RECORD, UNPACKED = "record", "unpacked"


@dataclass
class Item:
    """One thing a cleanup found, and whether it may go."""

    location: str  # An account ID, or "local" for the state directory.
    kind: str
    name: str
    bytes: int | None
    modified_at: float | None
    verdict: str
    reason: str
    job_id: str | None = None
    artifact: Artifact | None = field(default=None, repr=False)
    # A record's remote items, which must be deleted before it.
    tied: list[Item] = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        value = asdict(self)
        del value["artifact"], value["tied"]
        return {key: item for key, item in value.items() if item is not None}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str
    job_id: str | None = None


class JobIndex:
    """Which jobs use a launch (by account and attempt reference) or a bundle (by digest or its prefix)."""

    def __init__(self, jobs: list[JobRecord]):
        self.by_id = {job.id: job for job in jobs}
        self.attempts = {(a.account, a.ref.lower()): job for job in jobs for a in job.attempts}
        self.digests: dict[str, list[JobRecord]] = {}
        for job in jobs:
            for digest in self._digests(job):
                self.digests.setdefault(digest, []).append(job)

    @staticmethod
    def _digests(job: JobRecord) -> set[str]:
        bundles = [job.snapshot["source"], *job.snapshot["inputs"].values(), *job.transfers.values()]
        return {bundle["digest"] for bundle in bundles}

    def users(self, digest: str) -> list[JobRecord]:
        """Jobs using the bundle; digest may be a prefix, as Kaggle dataset names hold 40 characters."""
        return [job for full, jobs in self.digests.items() if full.startswith(digest) for job in jobs]


class Cleanup:
    def __init__(
        self,
        client,
        *,
        older_than_days: float = 7,
        include_snapshots: bool = False,
        accounts: list[str] | None = None,
        local: bool = True,
    ):
        """accounts limits the remote search to those account IDs ([] for none, None for all).

        client is a Client, or anything with its config, store and provider(account).
        """
        self.client = client
        self.state_dir = client.config.state_dir
        self.older_than_days = older_than_days
        self.cutoff = time.time() - older_than_days * 86400
        self.include_snapshots = include_snapshots
        config = client.config
        self.accounts = (
            [a.id for a in config.accounts] if accounts is None else [config.account(a).id for a in accounts]
        )
        self.local = local
        self.index = JobIndex(client.store.list())
        self.errors: dict[str, str] = {}

    # Finding ----------------------------------------------------------------------------------

    def items(self) -> list[Item]:
        found = [item for account in self.accounts for item in self._remote(account)]
        if self.local:
            found += self._local()
            found += self._records(found)
        return found

    def _remote(self, account: str) -> list[Item]:
        try:
            artifacts = self.client.provider(account).artifacts()
        except Exception as error:  # Listed separately, so one broken account hides no other.
            self.errors[account] = safe_message(error)
            return []
        return [self._judge_artifact(account, artifact) for artifact in artifacts]

    def _judge_artifact(self, account: str, artifact: Artifact) -> Item:
        if artifact.attempt is not None:
            job = self.index.attempts.get((account, artifact.attempt.lower()))
            verdict = self._job(job) if job else Verdict(UNKNOWN, "not launched by this queue")
        else:
            verdict = self._digest(artifact.digest or "", artifact.modified_at, ours=False)
        return Item(
            account,
            artifact.kind,
            artifact.name,
            artifact.bytes,
            artifact.modified_at,
            verdict.verdict,
            verdict.reason,
            verdict.job_id,
            artifact,
        )

    def _local(self) -> list[Item]:
        found = []
        for folder, kind in LOCAL_KINDS.items():
            parent = self.state_dir / folder
            for path in sorted(parent.iterdir()) if parent.is_dir() else []:
                if path.name.startswith("."):
                    continue
                item = self._judge_local(kind, path)
                found.append(item)
                if kind == "snapshot" and (unpacked := self._unpacked(path)):
                    item.bytes -= unpacked.bytes
                    found.append(unpacked)
        return found

    def _unpacked(self, bundle: Path) -> Item | None:
        """The unpacked copy an earlier version kept beside a bundle's archive, which holds every file."""
        unpacked = bundle / "files"
        if not (unpacked.is_dir() and (bundle / PAYLOAD).is_file()):
            return None
        size, modified = tree_size(unpacked)
        name = unpacked.relative_to(self.state_dir).as_posix()
        return Item("local", UNPACKED, name, size, modified, RECLAIMABLE, f"a copy of its {PAYLOAD}")

    def _records(self, found: list[Item]) -> list[Item]:
        """One item per job in the queue, holding the remote items tied to it."""
        tied: dict[str, list[Item]] = {}
        for item in found:
            for job_id in self._ties(item):
                tied.setdefault(job_id, []).append(item)
        checked = not self.errors and set(self.accounts) == {a.id for a in self.client.config.accounts}
        items = []
        for job in self.index.by_id.values():
            ties = tied.get(job.id, [])
            verdict = self._record(job, ties, checked)
            size = len(job.model_dump_json())
            items.append(Item("local", RECORD, job.id, size, job.finished_at, *astuple(verdict), tied=ties))
        return items

    def _ties(self, item: Item) -> set[str]:
        """Jobs whose records must outlast a remote item: a launch's job, and every job using a bundle.

        Without them, nothing would name the item as this queue's, and no cleanup would delete it.
        """
        artifact = item.artifact
        if item.location == "local" or artifact is None:
            return set()
        if artifact.attempt is not None:
            return {item.job_id} if item.job_id else set()
        return {job.id for job in self.index.users(artifact.digest)} if artifact.digest else set()

    def _record(self, job: JobRecord, tied: list[Item], checked: bool) -> Verdict:
        """A finished job's record may go once a check of every account found its remote items may go."""
        verdict = self._job(job)
        if verdict.verdict != RECLAIMABLE:
            return verdict
        configured = {account.id for account in self.client.config.accounts}
        unknown = sorted({job.account, *(attempt.account for attempt in job.attempts)} - configured)
        if unknown:
            return Verdict(KEPT, f"its account {unknown[0]} is not configured", job.id)
        if not checked:
            return Verdict(KEPT, "records go only in a cleanup that checks every account", job.id)
        kept = next((item for item in tied if item.verdict != RECLAIMABLE), None)
        if kept is not None:
            return Verdict(KEPT, f"its {kept.kind} on {kept.location} stays: {kept.reason}", job.id)
        return verdict

    def _judge_local(self, kind: str, path: Path) -> Item:
        size, modified = tree_size(path)
        if kind == "snapshot":
            verdict = self._snapshot(path.name, modified)
        elif kind == "upload":
            verdict = self._digest(path.name, modified, ours=True)
        else:
            job = self.index.by_id.get(path.name.removesuffix(".log"))
            verdict = self._job(job) if job else self._orphan(modified)
        name = path.relative_to(self.state_dir).as_posix()
        return Item("local", kind, name, size, modified, verdict.verdict, verdict.reason, verdict.job_id)

    # Judging ----------------------------------------------------------------------------------

    def _job(self, job: JobRecord) -> Verdict:
        """Whether what a job left may go: it finished, before the cutoff, with its outputs saved."""
        if not job.terminal:
            return Verdict(KEPT, f"its job is {job.state}", job.id)
        if job.download_state not in COLLECTED:
            return Verdict(KEPT, f"its outputs are {job.download_state}", job.id)
        if (job.finished_at or job.updated_at) > self.cutoff:
            return Verdict(KEPT, "its job finished recently", job.id)
        return Verdict(RECLAIMABLE, f"its job {job.state}", job.id)

    def _digest(self, digest: str, modified: float | None, *, ours: bool) -> Verdict:
        """A bundle may go once every job using it may; one no job uses is ours only if local."""
        users = self.index.users(digest) if digest else []
        if not users:
            return self._orphan(modified) if ours else Verdict(UNKNOWN, "not used by this queue")
        for job in users:
            verdict = self._job(job)
            if verdict.verdict != RECLAIMABLE:
                return verdict
        return Verdict(RECLAIMABLE, f"used only by finished jobs ({len(users)})")

    def _snapshot(self, digest: str, modified: float | None) -> Verdict:
        """A saved snapshot jobs used stays for retry and continue, unless include_snapshots."""
        verdict = self._digest(digest, modified, ours=True)
        if verdict.verdict == RECLAIMABLE and self.index.users(digest) and not self.include_snapshots:
            return Verdict(KEPT, "retry and continue need it; --include-snapshots removes it")
        return verdict

    def _orphan(self, modified: float | None) -> Verdict:
        if modified is not None and modified > self.cutoff:
            return Verdict(KEPT, "changed recently")
        return Verdict(RECLAIMABLE, "used by no job")

    # Deleting ---------------------------------------------------------------------------------

    def delete(self, items: list[Item]) -> dict:
        """Delete the reclaimable items; others are skipped. {"deleted": n, "bytes": n, "failed": {...}}.

        Job records go last, each only if every remote item tied to it was deleted here.
        """
        reclaimable = [item for item in items if item.verdict == RECLAIMABLE]
        gone, failed = [], {}
        for item in reclaimable:
            if item.kind == RECORD:
                continue
            try:
                self._delete(item)
                gone.append(item)
            except Exception as error:  # Reported per item; the others still go.
                failed[f"{item.location} {item.name}"] = short(error)
        deleted = {id(item) for item in gone}
        records = {
            item.name: item
            for item in reclaimable
            if item.kind == RECORD and all(id(tie) in deleted for tie in item.tied)
        }
        gone += [records[job_id] for job_id in self.client.store.forget(list(records))]
        return dict(deleted=len(gone), bytes=sum(item.bytes or 0 for item in gone), failed=failed)

    def _delete(self, item: Item) -> None:
        if item.location != "local":
            self.client.provider(item.location).delete_artifact(item.artifact)
            return
        path = self._local_path(item)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def _local_path(self, item: Item) -> Path:
        """Where a local item is, refused unless it is where _local lists items of its kind."""
        parts = Path(item.name).parts
        if item.kind == UNPACKED:
            listed = len(parts) == 3 and parts[0] == "bundles" and parts[2] == "files"
        else:
            listed = len(parts) == 2 and parts[0] in LOCAL_KINDS
        if not listed or ".." in parts:
            raise ValueError(f"Not a cleanup item of the state directory: {item.name}")
        return self.state_dir / item.name

    # Reporting --------------------------------------------------------------------------------

    def report(self, items: list[Item], *, limit: int | None = None) -> dict:
        """Totals by verdict, location and kind, and the reclaimable items (at most limit of them)."""
        reclaimable = [item for item in items if item.verdict == RECLAIMABLE]
        shown = reclaimable if limit is None else reclaimable[:limit]
        value = dict(
            schema_version=1,
            older_than_days=self.older_than_days,
            include_snapshots=self.include_snapshots,
            totals={
                verdict: _total([i for i in items if i.verdict == verdict])
                for verdict in (RECLAIMABLE, KEPT, UNKNOWN)
            },
            by_location=_by(items, "location"),
            items=[item.to_json() for item in shown],
            more=len(reclaimable) - len(shown),
        )
        if self.errors:
            value["errors"] = {account: short(error) for account, error in self.errors.items()}
        return value


def _total(items: list[Item]) -> dict:
    return dict(count=len(items), bytes=sum(item.bytes or 0 for item in items))


def _by(items: list[Item], key: str) -> dict:
    """Reclaimable totals per location (or kind)."""
    groups: dict[str, list[Item]] = {}
    for item in items:
        if item.verdict == RECLAIMABLE:
            groups.setdefault(getattr(item, key), []).append(item)
    return {name: _total(group) for name, group in groups.items()}

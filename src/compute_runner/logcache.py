"""Private copies of runs' logs in the state directory, read as bounded tails."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .models import JobRecord
from .results import RUN_LOG
from .store import atomic_write


class LogCache:
    """logs/JOB_ID.log: a run's full log, fetched from its provider only when no current copy exists.

    An unfinished job gets a bounded live snapshot on every read. A finished job whose outputs
    are downloaded is read from its run folder's run.log, which the download saved; otherwise
    first use, refresh, or a copy saved before it finished makes one remote read (never follows
    a stream), and other reads work offline. Replacing a copy is atomic, even when the fetch fails.
    """

    def __init__(self, client):
        self.client = client

    def path(self, job_id: str) -> Path:
        return self.client.config.state_dir / "logs" / f"{job_id}.log"

    def tail(self, job_id: str, *, lines: int, max_bytes: int, refresh: bool = False) -> dict:
        """At most lines lines and max_bytes UTF-8 bytes from the end of a job's log, and the copy's facts."""
        job = self.client.get(job_id)
        live = job.remote_ref is not None and not job.terminal
        path, fetched = self._source(job, refresh=refresh or live)
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            stream.seek(max(0, stat.st_size - max_bytes))
            text = _last_lines(stream.read(max_bytes), lines)
        size = len(text.encode("utf-8"))
        return dict(
            id=job.id,
            text=text,
            bytes=size,
            total_bytes=stat.st_size,
            truncated=size < stat.st_size,
            path=str(path),
            fetched=fetched,
            cached_at=stat.st_mtime,
            live=live,
        )

    def _source(self, job: JobRecord, *, refresh: bool) -> tuple[Path, bool]:
        """The file to read, and whether it was fetched from the provider now."""
        cached = self.path(job.id)
        if not refresh and not self._stale(job, cached):
            return cached, False
        downloaded = job.result_dir / RUN_LOG
        # The download saved the whole log once the run ended; fetching it again would keep a second copy.
        if not refresh and job.download_state == "complete" and downloaded.is_file():
            return downloaded, False
        chunks = (chunk.encode() for chunk in self.client.logs(job.id, follow=False))
        atomic_write(cached, chunks, check_space=True)
        return cached, True

    @staticmethod
    def _stale(job: JobRecord, path: Path) -> bool:
        return not path.exists() or (job.finished_at is not None and path.stat().st_mtime < job.finished_at)


def _last_lines(data: bytes, count: int) -> str:
    """The last count lines of data, split on "\\n" only so progress-bar "\\r" updates do not use them up.

    Undecodable bytes are dropped, such as a character split at the leading byte boundary.
    """
    decoded = data.decode("utf-8", errors="ignore")
    return "".join(re.findall(r"[^\n]*\n|[^\n]+", decoded)[-count:])

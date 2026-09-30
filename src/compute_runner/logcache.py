"""Private copies of runs' logs in the state directory, read as bounded tails."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .models import JobRecord
from .store import atomic_write


class LogCache:
    """logs/JOB_ID.log: a run's full log, fetched from its provider only when the copy may be stale.

    An unfinished job gets a bounded live snapshot on every read. For a finished job, first
    use, refresh, or a copy saved before it finished makes one remote read (never follows a
    stream); other reads work offline. Replacing a copy is atomic, even when the fetch fails.
    """

    def __init__(self, client):
        self.client = client

    def path(self, job_id: str) -> Path:
        return self.client.config.state_dir / "logs" / f"{job_id}.log"

    def tail(self, job_id: str, *, lines: int, max_bytes: int, refresh: bool = False) -> dict:
        """At most lines lines and max_bytes UTF-8 bytes from the end of a job's log, and the copy's facts."""
        job = self.client.get(job_id)
        path = self.path(job.id)
        live = job.remote_ref is not None and not job.terminal
        fetched = refresh or live or self._stale(job, path)
        if fetched:
            chunks = (chunk.encode() for chunk in self.client.logs(job.id, follow=False))
            atomic_write(path, chunks, check_space=True)
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

    @staticmethod
    def _stale(job: JobRecord, path: Path) -> bool:
        return not path.exists() or (job.finished_at is not None and path.stat().st_mtime < job.finished_at)


def _last_lines(data: bytes, count: int) -> str:
    """The last count lines of data, split on "\\n" only so progress-bar "\\r" updates do not use them up.

    Undecodable bytes are dropped, such as a character split at the leading byte boundary.
    """
    decoded = data.decode("utf-8", errors="ignore")
    return "".join(re.findall(r"[^\n]*\n|[^\n]+", decoded)[-count:])

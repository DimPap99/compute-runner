"""Small, bounded JSON responses for LLM clients. The worker still owns scheduling."""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import get_args

from .backend import safe_message
from .models import JobState
from .worker import settled


def short(value, limit=400):
    if value is None:
        return None
    text = " ".join(safe_message(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summary(job):
    value = dict(
        id=job.id,
        name=short(job.spec.name, 100),
        state=job.state,
        resource=short(job.spec.accelerator, 100) or ("gpu" if job.spec.gpu else "cpu"),
        internet=job.spec.internet,
        downloads=job.download_state,
        outputs_ready=job.download_state == "complete",
    )
    for key, item in {
        "reason": short(job.wait_reason),
        "error": short(job.error),
        "download_error": short(job.download_error),
        "url": job.url,
        "output_dir": str(job.result_dir) if job.download_state == "complete" else None,
        "parent_id": job.parent_id,
    }.items():
        if item not in (None, ""):
            value[key] = item
    return value


def _page_bounds(limit, offset=0):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")


class AgentClient:
    """Use Client.agent() or AgentClient(); results contain JSON primitives only."""

    def __init__(self, client=None):
        if client is None:
            from .client import Client

            client = Client()
        self.client = client

    def _jobs(self, jobs):
        if not jobs:
            return []
        with self.client.store.connection() as db:
            rows = db.execute(
                "SELECT job_id,batch_id,position FROM batch_jobs WHERE job_id IN ("
                + ",".join("?" for _ in jobs)
                + ")",
                [job.id for job in jobs],
            ).fetchall()
        batches = {row[0]: {"batch_id": row[1], "batch_index": row[2]} for row in rows}
        return [summary(job) | batches.get(job.id, {"batch_id": None}) for job in jobs]

    def health(self):
        health = self.client.worker_health()
        age = health["heartbeat_age_seconds"]
        value = {
            "running": health["running"],
            "heartbeat_age_seconds": round(age) if age is not None else None,
        }
        if health.get("error"):
            value["error"] = short(health["error"])
        return value

    def status(self, job_ids=None, *, batch_id=None, states=None, limit=20, offset=0):
        """Read local state only. Page size is bounded; counts cover the whole selection."""
        _page_bounds(limit, offset)
        if job_ids is not None:
            if isinstance(job_ids, str) or not 1 <= len(job_ids) <= 100:
                raise ValueError("job_ids must contain between 1 and 100 IDs")
            job_ids = list(dict.fromkeys(self.client.store.resolve_ids(list(job_ids))))
        if states is not None:
            if isinstance(states, str) or not states or not set(states) <= set(get_args(JobState)):
                raise ValueError("states must be a nonempty list or set of valid job states")
            states = sorted(set(states))
        counts, jobs = self.client.store.page(
            job_ids=job_ids,
            batch_id=batch_id,
            states=states,
            limit=limit,
            offset=offset,
        )
        total = sum(counts.values())
        return dict(
            schema_version=1,
            batch_id=batch_id,
            total=total,
            counts=counts,
            jobs=self._jobs(jobs),
            next_offset=offset + len(jobs) if offset + len(jobs) < total else None,
            worker=self.health(),
        )

    def submit(self, specs, *, request_key):
        """One key per logical request. Reuse it only to replay that exact submission."""
        if request_key is None:
            raise ValueError("request_key is required for agent submissions")
        batch = self.client.submit_batch(specs, request_key=request_key)
        return self.status(batch_id=batch.id) | {"replayed": batch.replayed}

    def preview(self, specs):
        """Small upload inventory; no snapshots, queue writes, or remote calls."""
        self.client.check_batch_size(specs)
        plans = [self.client.preview(spec) for spec in specs]
        return dict(
            schema_version=1,
            dry_run=True,
            total=len(specs),
            files=sum(len(p["files"]) + sum(len(i["files"]) for i in p["inputs"].values()) for p in plans),
            bytes=sum(p["bytes"] + sum(i["bytes"] for i in p["inputs"].values()) for p in plans),
            gpu_jobs=sum(spec.gpu for spec in specs),
            internet_jobs=sum(spec.internet for spec in specs),
            private=True,
        )

    def retry(self, job_id, *, request_key):
        if request_key is None:
            raise ValueError("request_key is required for agent retries")
        batch = self.client.retry_batch(self.client.store.resolve_id(job_id), request_key=request_key)
        return self.status(batch_id=batch.id) | {"replayed": batch.replayed}

    def cancel(self, job_id):
        job_id = self.client.store.resolve_id(job_id)
        job = self.client.get(job_id)
        if job.state != "cancelled":
            self.client.cancel(job_id)
        return self.status([job_id])

    def changes(self, *, after=0, batch_id=None, limit=20):
        """Latest state per changed job, coalesced. Drain has_more before waiting again.

        Keep a cursor per state directory and batch filter. after=0 also works for
        legacy jobs. Heartbeats and unchanged remote polls produce no events.
        """
        _page_bounds(limit)
        if type(after) is not int or after < 0:
            raise ValueError("after must be a nonnegative event cursor")
        cursor, more, jobs = self.client.store.changes(after=after, batch_id=batch_id, limit=limit)
        return dict(
            schema_version=1,
            batch_id=batch_id,
            cursor=cursor,
            has_more=more,
            jobs=self._jobs(jobs),
            worker=self.health(),
        )

    def logs(self, job_id, *, tail=50, max_bytes=8192, refresh=False):
        """Cache full logs privately on disk; return at most tail lines and max_bytes UTF-8 bytes.

        An unfinished job gets a bounded live snapshot on every call (live=True). For a
        finished job, first use, refresh=True, or a cache saved before it finished makes one
        remote read (never follows a stream). Other cached reads work offline.
        Cache replacement is atomic even on fetch failure.
        """
        if type(tail) is not int or not 1 <= tail <= 500:
            raise ValueError("tail must be an integer between 1 and 500")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 65536:
            raise ValueError("max_bytes must be an integer between 1 and 65536")
        job_id = self.client.store.resolve_id(job_id)
        job = self.client.get(job_id)
        path = self.client.config.state_dir / "logs" / f"{job_id}.log"
        # An unfinished run's log keeps growing, so it is always re-read as a bounded live snapshot.
        live = job.remote_ref is not None and not job.terminal
        fetched = (
            refresh
            or live
            or not path.exists()
            or (job.finished_at is not None and path.stat().st_mtime < job.finished_at)
        )
        if fetched:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{job_id}-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    for chunk in self.client.logs(job_id, follow=False):
                        stream.write(chunk.encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            stream.seek(max(0, stat.st_size - max_bytes))
            data = stream.read(max_bytes)
        # Drops undecodable bytes (e.g. a code point split at the leading byte boundary).
        # Lines split on "\n" only, so "\r" progress-bar updates do not consume the tail.
        decoded = data.decode("utf-8", errors="ignore")
        text = "".join(re.findall(r"[^\n]*\n|[^\n]+", decoded)[-tail:])
        size = len(text.encode("utf-8"))
        return dict(
            schema_version=1,
            id=job_id,
            text=text,
            bytes=size,
            total_bytes=stat.st_size,
            truncated=size < stat.st_size,
            path=str(path),
            fetched=fetched,
            cached_at=stat.st_mtime,
            live=live,
        )

    def wait(self, job_ids=None, *, batch_id=None, timeout=300, downloads=True, limit=20):
        """Block until every selected job settles or timeout seconds pass; a timeout is not an error.

        Settled means terminal with downloads complete, disabled or failed (or downloads=False),
        or blocked/needs_attention. Reads local state only; the worker does the remote work.
        """
        _page_bounds(limit)
        if type(timeout) not in (int, float) or not 0 <= timeout <= 86400:
            raise ValueError("timeout must be between 0 and 86400 seconds")
        if (job_ids is None) == (batch_id is None):
            raise ValueError("Select either job IDs or a batch")
        if job_ids is not None:
            if isinstance(job_ids, str) or not 1 <= len(job_ids) <= 100:
                raise ValueError("job_ids must contain between 1 and 100 IDs")
            job_ids = list(dict.fromkeys(self.client.store.resolve_ids(list(job_ids))))
        started = time.monotonic()
        stopped_since = None
        while True:
            _, jobs = self.client.store.page(batch_id=batch_id, job_ids=job_ids, limit=-1)
            done = all(settled(job, downloads=downloads) for job in jobs)
            elapsed = time.monotonic() - started
            if done or elapsed >= timeout:
                break
            if self.client.worker_health()["running"]:
                stopped_since = None
            else:
                # Tolerate a service restart; a worker that stays down would make this wait pointless.
                stopped_since = stopped_since or time.monotonic()
                if time.monotonic() - stopped_since >= 30:
                    raise RuntimeError("No worker is running. Start kgr service start or kgr worker run")
            time.sleep(min(2, self.client.config.poll_seconds, max(0.1, timeout - elapsed)))
        return self.status(job_ids, batch_id=batch_id, limit=limit) | dict(
            settled=done, timed_out=not done, waited_seconds=round(elapsed)
        )

    def outputs(self, job_id, *, limit=100, offset=0):
        """List downloaded output files, relative to root, without reading them.

        Remote names are relative to /kaggle/working, so KGR_OUTPUT_DIR files appear as outputs/NAME.
        """
        _page_bounds(limit, offset)
        job = self.client.get(self.client.store.resolve_id(job_id))
        root = job.result_dir / "outputs"
        files = sorted(
            (path.relative_to(root).as_posix(), path.stat().st_size)
            for path in (root.rglob("*") if root.is_dir() else [])
            if path.is_file() and not path.name.startswith(".kgr-")
        )
        page = files[offset : offset + limit]
        log = job.result_dir / "run.log"
        value = dict(
            schema_version=1,
            id=job.id,
            state=job.state,
            downloads=job.download_state,
            outputs_ready=job.download_state == "complete",
            root=str(root),
            total=len(files),
            files=[{"path": name, "bytes": size} for name, size in page],
            next_offset=offset + len(page) if offset + len(page) < len(files) else None,
        )
        if job.download_error:
            value["download_error"] = short(job.download_error)
        if log.is_file():
            value["log_path"] = str(log)
        return value

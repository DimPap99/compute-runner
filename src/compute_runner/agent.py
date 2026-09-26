"""Small, bounded JSON responses for LLM clients. The worker still owns scheduling."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import get_args

import yaml

from .models import ACTIVE, JobState
from .providers import safe_message
from .results import RUN_RECORD, listed_outputs
from .store import atomic_write
from .worker import MOVABLE, occupancy

# Operation failures reported to callers as a message; anything else is a bug and keeps its traceback.
ERRORS = (ValueError, KeyError, RuntimeError, OSError, sqlite3.Error, yaml.YAMLError)


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
        account=job.account,
        resource=short(job.spec.accelerator, 100) or ("gpu" if job.spec.gpu else "cpu"),
        internet=job.spec.internet,
        downloads=job.download_state,
        outputs_ready=job.download_state == "complete",
        # Fixed at submission; the worker fills it.
        run_dir=str(job.result_dir),
    )
    for key, item in {
        "params": job.spec.params,
        "reason": short(job.wait_reason),
        "suggested_account": job.suggested_account,
        "suggested_transfer": job.suggested_transfer or None,
        "error": short(job.error),
        "download_error": short(job.download_error),
        "url": job.url,
        "parent_id": job.parent_id,
    }.items():
        if item not in (None, "", {}):
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

    def _job_ids(self, job_ids):
        if job_ids is None:
            return None
        if isinstance(job_ids, str) or not 1 <= len(job_ids) <= 100:
            raise ValueError("job_ids must contain between 1 and 100 IDs")
        return list(dict.fromkeys(self.client.store.resolve_ids(list(job_ids))))

    def _batch_status(self, batch):
        return self.status(batch_id=batch.id) | {"replayed": batch.replayed}

    def health(self):
        """Whether a worker holds the queue, and its heartbeat age; local only."""
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
        """Read local state only. Page size is bounded; counts cover the whole selection.

        A batch lists in submission order; other selections list the newest jobs first, so a new
        conversation sees current work on the first page.
        """
        _page_bounds(limit, offset)
        job_ids = self._job_ids(job_ids)
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
            newest_first=batch_id is None,
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

    def accounts(self):
        """Configured accounts in preference order, with slots in use and the worker's last remote check.

        Local only: counts combine this queue's runs with other runs the worker last discovered.
        """
        config = self.client.config
        try:
            seen = json.loads((config.state_dir / "accounts.json").read_text())
        except (OSError, ValueError):
            seen = {}
        jobs = self.client.store.list(ACTIVE)
        accounts = []
        for account in config.accounts:
            found = seen.get(account.id, {})
            used = occupancy(jobs, account.id, found.get("runs", {}))
            checked, quota = found.get("checked_at"), found.get("gpu_seconds")
            value = dict(
                id=account.id,
                provider=account.provider,
                cpu=dict(used=used["cpu"], limit=account.cpu_limit),
                gpu=dict(used=used["gpu"], limit=account.gpu_limit),
                # SSH machines have no GPU time limit; null elsewhere means not checked yet.
                gpu_quota_limited=account.provider != "ssh",
                gpu_quota_seconds=None if quota is None else round(quota),
                checked_age_seconds=round(time.time() - checked) if checked else None,
            )
            if found.get("error"):
                value["error"] = short(found["error"])
            accounts.append(value)
        return dict(
            schema_version=1,
            failover=config.failover,
            default=accounts[0]["id"] if accounts else None,
            accounts=accounts,
            worker=self.health(),
        )

    def submit(self, specs, *, request_key, account=None):
        """One key per logical request. Reuse it only to replay that exact submission."""
        if request_key is None:
            raise ValueError("request_key is required for agent submissions")
        return self._batch_status(self.client.submit_batch(specs, request_key=request_key, account=account))

    def preview(self, specs, *, account=None):
        """Small upload inventory; no snapshots, queue writes, or remote calls."""
        self.client.check_batch_size(specs)
        plans = [self.client.preview(spec, account) for spec in specs]
        return dict(
            schema_version=1,
            dry_run=True,
            total=len(specs),
            experiment_dirs=sorted({p["experiment_dir"] for p in plans}),
            files=sum(len(p["files"]) + sum(len(i["files"]) for i in p["inputs"].values()) for p in plans),
            bytes=sum(p["bytes"] + sum(i["bytes"] for i in p["inputs"].values()) for p in plans),
            gpu_jobs=sum(spec.gpu for spec in specs),
            internet_jobs=sum(spec.internet for spec in specs),
            private=True,
        )

    def retry(self, job_id, *, request_key, account=None):
        """Queue a finished or blocked job's saved code as a new job, on its account unless one is given."""
        if request_key is None:
            raise ValueError("request_key is required for agent retries")
        job_id = self.client.store.resolve_id(job_id)
        return self._batch_status(self.client.retry_batch(job_id, request_key=request_key, account=account))

    def continue_run(self, job_id, *, request_key, account=None):
        """Resume a stopped resumable job from its downloaded checkpoint as the next run of its experiment."""
        if request_key is None:
            raise ValueError("request_key is required for agent continuations")
        job_id = self.client.store.resolve_id(job_id)
        batch = self.client.continue_batch(job_id, request_key=request_key, account=account)
        return self._batch_status(batch)

    def move(self, job_ids=None, *, batch_id=None, account, transfer=False, limit=20):
        """Move the selected jobs that have not been submitted to another account.

        Submitted and finished jobs, and jobs already there, stay; not_moved lists jobs the account
        rejected. transfer allows copying datasets the account cannot read, also for jobs already there.
        """
        _page_bounds(limit)
        if (job_ids is None) == (batch_id is None):
            raise ValueError("Select either job IDs or a batch")
        job_ids = self._job_ids(job_ids)
        target = self.client.config.account(account).id
        _, jobs = self.client.store.page(batch_id=batch_id, job_ids=job_ids, limit=-1)
        moved, rejected = 0, []
        for job in jobs:
            if job.state not in MOVABLE or (job.account == target and (job.transfer or not transfer)):
                continue
            try:
                self.client.move(job.id, target, transfer=transfer)
                moved += 1
            except ValueError as error:
                rejected.append({"id": job.id, "error": short(error)})
        value = self.status(job_ids, batch_id=batch_id, limit=limit) | {"moved": moved}
        return value | {"not_moved": rejected} if rejected else value

    def cancel(self, job_id):
        """Cancel pending work locally, or ask the provider to stop a running job; repeating is harmless."""
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
        live = job.remote_ref is not None and not job.terminal
        fetched = (
            refresh
            or live
            or not path.exists()
            or (job.finished_at is not None and path.stat().st_mtime < job.finished_at)
        )
        if fetched:
            atomic_write(
                path,
                (chunk.encode() for chunk in self.client.logs(job_id, follow=False)),
                check_space=True,
            )
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
        job_ids = self._job_ids(job_ids)
        started = time.monotonic()
        _, done = self.client.wait_many(job_ids, batch_id=batch_id, timeout=timeout, downloads=downloads)
        return self.status(job_ids, batch_id=batch_id, limit=limit) | dict(
            settled=done, timed_out=not done, waited_seconds=round(time.monotonic() - started)
        )

    def outputs(self, job_id, *, limit=100, offset=0):
        """List downloaded output files, relative to root, without reading them.

        KGR_OUTPUT_DIR files appear as outputs/NAME, and other files the run left in its working
        directory as working/NAME. root is the run folder (older jobs: its outputs folder).
        """
        _page_bounds(limit, offset)
        job = self.client.get(self.client.store.resolve_id(job_id))
        root, files = listed_outputs(job)
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
        if (job.result_dir / RUN_RECORD).is_file():
            value["record_path"] = str(job.result_dir / RUN_RECORD)
        return value

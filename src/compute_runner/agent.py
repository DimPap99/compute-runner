"""Small, bounded JSON responses for LLM clients. The worker still owns scheduling."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter
from typing import get_args

import yaml

from .models import ACTIVE, JobState
from .providers import safe_message
from .results import RUN_RECORD, listed_outputs
from .store import atomic_write
from .worker import MOVABLE, inventory, last_discovery, occupancy, outstanding

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


def _checked(found, now):
    """How old an account's last discovery is (None: never made), and why the latest one failed."""
    checked = found.get("checked_at")
    value = dict(checked_age_seconds=None if checked is None else max(0, round(now - checked)))
    if found.get("error"):
        value["error"] = short(found["error"])
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
        for key in ("stage", "job_id"):
            if health.get(key):
                value[key] = short(health[key])
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

    def _discovery(self, live=False, account=None):
        """Configured accounts' runs, GPU seconds, check time and error, as the worker saves them.

        From the worker's last check, or asked of each provider now when live (on Kaggle, a
        request per notebook run in the last 24 hours). account keeps that account only.
        """
        accounts = [item.id for item in self.client.config.accounts if account in (None, item.id)]
        if not live:
            saved = last_discovery(self.client.config.state_dir)
            return {account_id: saved.get(account_id) or {} for account_id in accounts}
        # One at a time: Kaggle's client setup silences output by swapping the process's stdout.
        found = {}
        for account_id in accounts:
            try:
                runs, gpu_seconds = inventory(self.client.provider(account_id))
                found[account_id] = dict(runs=runs, gpu_seconds=gpu_seconds, checked_at=time.time())
            except ERRORS as error:  # One broken account must not hide the others.
                found[account_id] = dict(error=safe_message(error))
        return found

    def accounts(self, *, live=False):
        """Configured accounts in preference order, with slots in use and free, and GPU quota left.

        used counts this queue's runs and other runs discovered on the account; free is what a new
        job could take now, or None while the account's runs are unknown. Local only unless live,
        which asks every provider now instead of reading the worker's last check.
        """
        config = self.client.config
        found, now = self._discovery(live), time.time()
        jobs = self.client.store.list(ACTIVE)
        accounts = []
        for account in config.accounts:
            seen = found[account.id]
            checked = _checked(seen, now)
            known = checked["checked_age_seconds"] is not None and "error" not in checked
            used = occupancy(jobs, account.id, seen.get("runs", {}))
            quota = seen.get("gpu_seconds")
            value = dict(id=account.id, provider=account.provider)
            for pool in ("cpu", "gpu"):
                limit = getattr(account, pool + "_limit")
                # As the worker admits jobs: no GPU job starts once the GPU time is spent.
                spent = pool == "gpu" and quota is not None and quota <= 0
                free = None if not known else (0 if spent else max(0, limit - used[pool]))
                value[pool] = dict(used=used[pool], limit=limit, free=free)
            accounts.append(
                value
                | dict(
                    # SSH machines have no GPU time limit; null elsewhere means not checked yet.
                    gpu_quota_limited=account.provider != "ssh",
                    gpu_quota_seconds=None if quota is None else round(quota),
                )
                | checked
            )
        return dict(
            schema_version=1,
            failover=config.failover,
            default=accounts[0]["id"] if accounts else None,
            accounts=accounts,
            worker=self.health(),
        )

    def running(self, resource=None, *, account=None, live=False):
        """Runs holding account slots, in account preference order; local only unless live.

        This queue's submitted jobs come first on each account, then runs started elsewhere
        (without job_id) as the worker last discovered them, or as each provider reports them now
        when live; discovery gives each account's check age. resource "cpu" or "gpu" keeps runs
        holding that pool. A discovered run of unknown resource holds both, as the worker counts it.
        """
        if resource not in (None, "cpu", "gpu"):
            raise ValueError("resource must be cpu or gpu")
        selected = self.client.config.account(account).id if account else None
        found, now = self._discovery(live, selected), time.time()
        runs, ours = [], set()
        for job in self.client.store.list(ACTIVE):
            if not outstanding(job) or selected not in (None, job.attempts[-1].account):
                continue
            attempt = job.attempts[-1]
            ours.add((attempt.account, attempt.ref.lower()))
            run = dict(
                account=attempt.account,
                ref=attempt.ref,
                resource="gpu" if job.spec.gpu else "cpu",
                job_id=job.id,
                name=short(job.spec.name, 100),
                state=job.state,
                elapsed_seconds=max(0, round(now - attempt.started_at)),
            )
            extra = dict(accelerator=short(job.spec.accelerator, 100), url=job.url)
            runs.append(run | {key: item for key, item in extra.items() if item})
        for account_id, seen in found.items():
            runs += [
                dict(account=account_id, ref=ref, resource=kind if kind in ("cpu", "gpu") else "unknown")
                for ref, kind in seen.get("runs", {}).items()
                if (account_id, ref) not in ours
            ]
        runs = [
            dict(run, provider=run["account"].partition(":")[0])
            for run in runs
            if resource is None or run["resource"] in (resource, "unknown")
        ]
        # Stable: this queue's runs stay oldest first. Accounts no longer configured go last.
        order = {account_id: index for index, account_id in enumerate(found)}
        runs.sort(key=lambda run: (order.get(run["account"], len(order)), "job_id" not in run))
        counts = Counter(run["resource"] for run in runs)
        return dict(
            schema_version=1,
            resource=resource,
            account=selected,
            total=len(runs),
            counts={kind: counts[kind] for kind in ("cpu", "gpu", "unknown")},
            runs=runs,
            discovery={account_id: _checked(seen, now) for account_id, seen in found.items()},
            worker=self.health(),
        )

    def runtime(self, job_id):
        """Read Kaggle's saved accelerator metadata, separately from our request."""
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        job = self.client.get(job_id)
        if not job.account.startswith("kaggle:") or not job.remote_ref:
            raise ValueError("Runtime diagnostics require a submitted Kaggle job")
        provider = self.client.provider(job.account)
        metadata = provider._kernels("get_kernel", ApiGetKernelRequest(), job.remote_ref).metadata
        return {"schema_version": 1, "job_id": job.id, "account": job.account,
                "ref": job.remote_ref, "url": provider.url(job.remote_ref),
                "requested": {"gpu": job.spec.gpu, "accelerator": job.spec.accelerator},
                "provider": {"enable_gpu": getattr(metadata, "enable_gpu", None),
                             "machine_shape": getattr(metadata, "machine_shape", None)},
                "session": provider.status(job.remote_ref),
                "note": "Saved provider metadata; CUDA must also be confirmed inside the workload."}

    def inputs(self, job_id):
        """Read provider status for a pending job's exact input identities; never upload."""
        job = self.client.get(job_id)
        provider = self.client.provider(job.account)
        if not job.account.startswith("kaggle:"):
            raise ValueError("Input status diagnostics currently support Kaggle jobs")
        refs = {}
        if not job.snapshot["single_file"]:
            refs["source"] = f"{provider.owner}/kgr-b-{job.snapshot['source']['digest'][:40]}"
        for alias, bundle in job.snapshot["inputs"].items():
            refs[alias] = f"{provider.owner}/kgr-b-{bundle['digest'][:40]}"
        for alias, value in job.spec.inputs.items():
            if str(value).startswith("kaggle:"):
                refs[alias] = str(value)[7:]
        for alias, bundle in job.transfers.items():
            refs[alias] = f"{provider.owner}/kgr-b-{bundle['digest'][:40]}"
        for alias, ref in job.upload_refs.items():
            refs[alias.removeprefix("input:")] = ref
        rows = []
        for alias, ref in list(refs.items())[:10]:
            row = {"alias": alias, "ref": ref}
            bundle = (job.snapshot["source"] if alias == "source" else
                      job.transfers.get(alias) or job.snapshot["inputs"].get(alias))
            if bundle:
                row["bundle_bytes"] = bundle.get("bytes")
                receipt = self.client.config.state_dir / "uploads" / bundle["digest"] / "create-receipt.json"
                if receipt.exists():
                    data = json.loads(receipt.read_text())
                    if data.get("ref", "").lower() == "/".join(ref.split("/")[:2]).lower():
                        row["create_receipt"] = data
            for key, fmt in (("status", None), ("version", "json(current_version_number)")):
                try:
                    value = provider.api.dataset_status("/".join(ref.split("/")[:2]), format=fmt)
                    row[key] = value if fmt is None else json.loads(value)
                except Exception as error:
                    row[key + "_error"] = short(error)
            if "status_error" in row:
                try:
                    page = 1
                    while page <= 10:
                        datasets = provider.api.dataset_list(mine=True, page=page)
                        if not datasets:
                            break
                        matches = [d for d in datasets if d and (d.ref or "").lower() == ref.lower()]
                        if matches:
                            d = matches[0]
                            row["inventory"] = {k: short(getattr(d, k, None)) for k in
                                                ("ref", "id", "title", "last_updated", "is_private",
                                                 "total_bytes", "current_version_number")}
                            break
                        page += 1
                except Exception as error:
                    row["inventory_error"] = short(error)
            rows.append(row)
        return {"schema_version": 1, "job_id": job.id, "inputs": rows}

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

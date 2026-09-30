"""Small, bounded JSON responses for LLM clients. The worker still owns scheduling."""

from __future__ import annotations

import sqlite3
import time
from typing import get_args

import yaml

from .logcache import LogCache
from .models import JobState
from .providers import short
from .results import RUN_RECORD, listed_outputs
from .views import CapacityView

# Operation failures reported to callers as a message; anything else is a bug and keeps its traceback.
ERRORS = (ValueError, KeyError, RuntimeError, OSError, sqlite3.Error, yaml.YAMLError)


def summary(job):
    value = dict(
        id=job.id,
        name=short(job.spec.name, 100),
        state=job.state,
        account=job.account,
        resource=short(job.spec.accelerator, 100) or job.pool,
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
        batches = self.client.store.batch_positions([job.id for job in jobs])
        return [summary(job) | self._batch_fields(batches.get(job.id)) for job in jobs]

    @staticmethod
    def _batch_fields(found):
        return {"batch_id": None} if found is None else {"batch_id": found[0], "batch_index": found[1]}

    def _job_ids(self, job_ids):
        if job_ids is None:
            return None
        if isinstance(job_ids, str) or not 1 <= len(job_ids) <= 100:
            raise ValueError("job_ids must contain between 1 and 100 IDs")
        return list(dict.fromkeys(self.client.store.resolve_ids(list(job_ids))))

    @staticmethod
    def _one_selection(job_ids, batch_id):
        if (job_ids is None) == (batch_id is None):
            raise ValueError("Select either job IDs or a batch")

    def _selected(self, job_ids, batch_id, *, states=None) -> list[str]:
        """IDs of the named jobs (prefixes allowed; one ID may be a string) or a batch's, in some states."""
        self._one_selection(job_ids, batch_id)
        job_ids = self._job_ids([job_ids] if isinstance(job_ids, str) else job_ids)
        _, jobs = self.client.store.page(batch_id=batch_id, job_ids=job_ids, states=states, limit=-1)
        return [job.id for job in jobs]

    @staticmethod
    def _states(states):
        if states is None:
            return None
        if isinstance(states, str) or not states or not set(states) <= set(get_args(JobState)):
            raise ValueError("states must be a nonempty list or set of valid job states")
        return sorted(set(states))

    @staticmethod
    def _resource(resource):
        if resource not in (None, "cpu", "gpu"):
            raise ValueError("resource must be cpu or gpu")
        return resource

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

    def status(
        self, job_ids=None, *, batch_id=None, states=None, account=None, resource=None, limit=20, offset=0
    ):
        """Read local state only. Page size is bounded; counts cover the whole selection.

        A batch lists in submission order; other selections list the newest jobs first, so a new
        conversation sees current work on the first page. account and resource ("cpu" or "gpu")
        narrow any selection.
        """
        _page_bounds(limit, offset)
        counts, jobs = self.client.store.page(
            job_ids=self._job_ids(job_ids),
            batch_id=batch_id,
            states=self._states(states),
            account=account and self.client.config.account(account).id,
            pool=self._resource(resource),
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

    def accounts(self, *, live=False):
        """Configured accounts in preference order, with slots in use and free, GPU time left, and totals.

        used counts this queue's runs and other runs discovered on the account; free is what a new
        job could take now, or None while the account's runs are unknown. Local only unless live,
        which asks every provider now instead of reading the worker's last check.
        """
        view = CapacityView(self.client, live=live)
        rows = view.account_rows()
        return dict(
            schema_version=1,
            failover=self.client.config.failover,
            default=rows[0]["id"] if rows else None,
            accounts=rows,
            totals=view.totals(rows),
            worker=self.health(),
        )

    def running(self, resource=None, *, account=None, live=False):
        """Runs holding account slots, in account preference order; local only unless live.

        This queue's submitted jobs come first on each account, then runs started elsewhere
        (without job_id) as the worker last discovered them, or as each provider reports them now
        when live; discovery gives each account's check age. resource "cpu" or "gpu" keeps runs
        holding that pool. A discovered run of unknown resource holds both, as the worker counts it.
        """
        view = CapacityView(self.client, live=live, account=account)
        runs = view.runs(self._resource(resource))
        return dict(
            schema_version=1,
            resource=resource,
            account=view.selected,
            total=len(runs),
            counts=view.counts(runs),
            runs=runs,
            discovery={account.id: view.checked(account.id) for account in view.accounts},
            worker=self.health(),
        )

    def overview(self, *, limit=10):
        """The whole picture in one local call: the worker, jobs by state and account, each account's
        slots and GPU time, what holds the slots, and up to limit jobs waiting on someone.
        """
        _page_bounds(limit)
        view = CapacityView(self.client)
        rows = view.account_rows()
        by_account = self.client.store.counts_by_account()
        runs = view.runs()
        waiting, jobs = self.client.store.attention(limit)
        counts = {}
        for states in by_account.values():
            for state, count in states.items():
                counts[state] = counts.get(state, 0) + count
        return dict(
            schema_version=1,
            worker=self.health(),
            jobs=dict(total=sum(counts.values()), counts=counts),
            accounts=[row | {"jobs": by_account.get(row["id"], {})} for row in rows],
            totals=view.totals(rows),
            running=dict(total=len(runs), counts=view.counts(runs)),
            attention=dict(total=waiting, jobs=self._jobs(jobs)),
        )

    def cleanup(self, *, older_than_days=7, include_snapshots=False, account=None, local=True, limit=20):
        """What this runner left behind and what could be deleted; reports only, deleting is the user's call.

        Asks the providers for their listings, so it is not local; account limits that to one
        account (local=False leaves the state directory out).
        """
        _page_bounds(limit)
        return self.client.cleanup(
            older_than_days=older_than_days,
            include_snapshots=include_snapshots,
            accounts=None if account is None else [account],
            local=local,
            limit=limit,
        )

    def runtime(self, job_id):
        """The resources the provider saved for a submitted job's run, beside what the job asked for."""
        job = self.client.get(self.client.store.resolve_id(job_id))
        if not job.remote_ref:
            raise ValueError("Runtime diagnostics require a submitted job")
        saved = self.client.provider(job.attempts[-1].account).runtime(job.remote_ref)
        return dict(
            schema_version=1,
            job_id=job.id,
            account=job.account,
            ref=job.remote_ref,
            requested=dict(gpu=job.spec.gpu, accelerator=job.spec.accelerator),
            **saved,
            note="Saved provider metadata; CUDA must also be confirmed inside the workload.",
        )

    def inputs(self, job_id):
        """What the provider reports for each input a job attaches; never uploads."""
        job = self.client.get(self.client.store.resolve_id(job_id))
        return dict(
            schema_version=1, job_id=job.id, inputs=self.client.provider(job.account).input_status(job)
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

    def retry(self, job_ids=None, *, request_key, account=None, batch_id=None, states=None):
        """Queue finished or blocked jobs' saved code as new jobs, on their accounts unless one is given.

        Select one job, several, or a batch's jobs in some states (such as failed); they are
        queued together as one new batch, or not at all.
        """
        if request_key is None:
            raise ValueError("request_key is required for agent retries")
        job_ids = self._selected(job_ids, batch_id, states=self._states(states))
        if not job_ids:
            raise ValueError("No job in the selection to rerun")
        batch = self.client.retry_jobs(job_ids, request_key=request_key, account=account)
        return self._batch_status(batch)

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
        self._one_selection(job_ids, batch_id)
        job_ids = self._job_ids(job_ids)
        target = self.client.config.account(account).id
        _, jobs = self.client.store.page(batch_id=batch_id, job_ids=job_ids, limit=-1)
        moved, rejected = 0, []
        for job in jobs:
            if not job.movable or (job.account == target and (job.transfer or not transfer)):
                continue
            try:
                self.client.move(job.id, target, transfer=transfer)
                moved += 1
            except ValueError as error:
                rejected.append({"id": job.id, "error": short(error)})
        value = self.status(job_ids, batch_id=batch_id, limit=limit) | {"moved": moved}
        return value | {"not_moved": rejected} if rejected else value

    def cancel(self, job_ids=None, *, batch_id=None, limit=20):
        """Cancel pending work locally, or ask the provider to stop running jobs; repeating is harmless.

        One job's failure is an error. For several jobs, or a batch (whose finished jobs are left
        alone), not_cancelled lists the jobs that could not be cancelled.
        """
        _page_bounds(limit)
        selected = self._selected(job_ids, batch_id)
        if batch_id is not None:
            selected = [job_id for job_id in selected if not self.client.get(job_id).terminal]
        result = self.client.cancel_many(selected)
        if result["failed"] and len(selected) == 1 and batch_id is None:
            raise ValueError(next(iter(result["failed"].values())))
        value = self.status(job_ids=selected if batch_id is None else None, batch_id=batch_id, limit=limit)
        value["cancelled"] = len(result["cancelled"])
        failed = [{"id": job_id, "error": short(message)} for job_id, message in result["failed"].items()]
        return value | {"not_cancelled": failed} if failed else value

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
        """At most tail lines and max_bytes UTF-8 bytes of a job's log, cached privately (see LogCache)."""
        if type(tail) is not int or not 1 <= tail <= 500:
            raise ValueError("tail must be an integer between 1 and 500")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 65536:
            raise ValueError("max_bytes must be an integer between 1 and 65536")
        job_id = self.client.store.resolve_id(job_id)
        found = LogCache(self.client).tail(job_id, lines=tail, max_bytes=max_bytes, refresh=refresh)
        return dict(schema_version=1, **found)

    def wait(self, job_ids=None, *, batch_id=None, timeout=300, downloads=True, limit=20):
        """Block until every selected job settles or timeout seconds pass; a timeout is not an error.

        Settled means terminal with downloads complete, disabled or failed (or downloads=False),
        or blocked/needs_attention. Reads local state only; the worker does the remote work.
        """
        _page_bounds(limit)
        if type(timeout) not in (int, float) or not 0 <= timeout <= 86400:
            raise ValueError("timeout must be between 0 and 86400 seconds")
        self._one_selection(job_ids, batch_id)
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

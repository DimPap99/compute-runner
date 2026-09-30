"""Commands on jobs: submit, follow, rerun, move and stop them."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from ..logcache import LogCache
from ..models import HALTED, JobSpec
from .common import client, full_id, output, selected
from .options import AccountOption, BatchOption, StatesOption, workload_options
from .output import job_dict

# Enough for any tail a person reads in a terminal; the full log stays in the cache.
TAIL_BYTES = 16 * 1024 * 1024


@workload_options
def submit(
    ctx: typer.Context,
    source: Path,
    specs: list[JobSpec],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the files to upload; queue nothing")
    ] = False,
    request_key: Annotated[
        str | None, typer.Option(help="Stable key; reusing it replays the submission")
    ] = None,
    account: Annotated[str | None, typer.Option(help="Account ID such as kaggle:USER; default first")] = None,
):
    """Queue a script, notebook, project YAML or jobs: YAML; --dry-run lists the files to upload."""
    out = output(ctx)
    if dry_run:
        out.emit([client(ctx).preview(spec, account) for spec in specs])
        return
    jobs = client(ctx).submit_many(specs, request_key=request_key, account=account)
    if out.json:
        out.emit([job_dict(job) for job in jobs])
        return
    for job in jobs:
        out.line(f"Queued {job.id} ({job.spec.name}): results in {job.result_dir}")
    if not client(ctx).worker_health()["running"]:
        start = "compute-runner service start (or compute-runner worker run)"
        out.line(f"Queued locally. Start processing with: {start}")


def status(ctx: typer.Context, job_id: str):
    """Print a job's full record."""
    output(ctx).emit(client(ctx).get(full_id(ctx, job_id)))


def watch(ctx: typer.Context, job_id: str):
    """Print a job's progress every polling interval until it settles."""
    found = full_id(ctx, job_id)
    while True:
        record = client(ctx).get(found)
        output(ctx).emit(
            dict(
                id=record.id,
                state=record.state,
                remote_state=record.remote_state,
                url=record.url,
                elapsed_seconds=round((record.finished_at or time.time()) - record.created_at),
                wait_reason=record.wait_reason,
                error=record.error,
                download_state=record.download_state,
            )
        )
        if record.terminal or record.state in HALTED:
            return
        time.sleep(client(ctx).config.poll_seconds)


def wait(
    ctx: typer.Context,
    job_id: str,
    timeout: float | None = None,
    downloads: Annotated[bool, typer.Option("--downloads/--no-downloads")] = True,
):
    """Wait for a job to settle; exits 1 unless it succeeded."""
    record = client(ctx).wait(full_id(ctx, job_id), timeout=timeout, downloads=downloads)
    output(ctx).emit(record)
    if record.state != "succeeded":
        raise typer.Exit(1)


def logs(
    ctx: typer.Context,
    job_id: str,
    follow: Annotated[bool, typer.Option("--follow", help="Stream the log until the run ends")] = False,
    tail: Annotated[int | None, typer.Option(min=1, help="Only the last N lines, from the log cache")] = None,
):
    """Print a run's log, a snapshot while it runs, its last lines, or stream it with --follow."""
    if follow and tail:
        raise ValueError("Choose --follow or --tail")
    found = full_id(ctx, job_id)
    chunks = (
        [LogCache(client(ctx)).tail(found, lines=tail, max_bytes=TAIL_BYTES)["text"]]
        if tail
        else client(ctx).logs(found, follow=follow)
    )
    for chunk in chunks:
        if output(ctx).json:
            typer.echo(json.dumps({"data": chunk}))
        else:
            typer.echo(chunk, nl=False)
            sys.stdout.flush()


def download(ctx: typer.Context, job_id: str):
    """Download a finished run's outputs now; the worker also does this automatically."""
    output(ctx).emit(client(ctx).download(full_id(ctx, job_id)))


def retry(
    ctx: typer.Context,
    job_ids: Annotated[list[str] | None, typer.Argument(help="Job IDs; or select with --batch")] = None,
    batch: BatchOption = None,
    state: StatesOption = None,
    account: Annotated[str | None, typer.Option(help="Rerun there; default: each job's own account")] = None,
):
    """Rerun finished or blocked jobs' saved code as new jobs; all of them, or none."""
    chosen = selected(ctx, job_ids, batch=batch, states=state)
    if not chosen:
        raise ValueError("No job in the selection to rerun")
    batch_record = client(ctx).retry_jobs(chosen, account=account)
    if len(chosen) == 1:
        output(ctx).emit(batch_record.jobs[0])
    else:
        output(ctx).emit([job_dict(job) for job in batch_record.jobs])


def continue_run(ctx: typer.Context, job_id: str, account: str | None = None):
    """Resume a stopped resumable run from its verified checkpoint as the next run of its experiment."""
    output(ctx).emit(client(ctx).continue_run(full_id(ctx, job_id), account=account))


def move(
    ctx: typer.Context,
    job_id: str,
    account: Annotated[str, typer.Option()],
    transfer: Annotated[
        bool, typer.Option("--transfer", help="Allow copying datasets the account cannot read")
    ] = False,
):
    """Place a job that has not been submitted on another account."""
    output(ctx).emit(client(ctx).move(full_id(ctx, job_id), account, transfer=transfer))


def cancel(
    ctx: typer.Context,
    job_ids: Annotated[
        list[str] | None, typer.Argument(help="Job IDs; or select with --batch or --account")
    ] = None,
    batch: BatchOption = None,
    account: AccountOption = None,
    state: StatesOption = None,
    yes: Annotated[bool, typer.Option("--yes", help="Cancel several jobs without asking")] = False,
):
    """Cancel pending work locally, or ask providers to stop running jobs.

    One job prints its record. Several (IDs, a batch's or an account's unfinished jobs,
    optionally in some states) are confirmed first, and failures are listed.
    """
    if job_ids and len(job_ids) == 1 and not (batch or account):
        output(ctx).emit(client(ctx).cancel(full_id(ctx, job_ids[0])))
        return
    chosen = [
        found
        for found in selected(ctx, job_ids, batch=batch, account=account, states=state)
        if job_ids or not client(ctx).get(found).terminal
    ]
    if chosen and not yes:
        typer.confirm(f"Cancel {len(chosen)} jobs?", abort=True, err=True)
    result = client(ctx).cancel_many(chosen)
    out = output(ctx)
    if out.json:
        out.emit(result)
        return
    out.line(f"Cancelled {len(result['cancelled'])} of {len(chosen)} jobs")
    out.notes(f"{found}: {message}" for found, message in result["failed"].items())


def resolve(
    ctx: typer.Context,
    job_id: str,
    not_submitted: Annotated[
        bool, typer.Option(help="Assert you independently verified no remote run exists")
    ] = False,
):
    """Operator assertion that an unresolved submission created no remote run; the user's call alone."""
    if not not_submitted:
        raise ValueError(
            "Inspect the run on its provider first; --not-submitted is an explicit operator assertion"
        )
    output(ctx).emit(client(ctx).resolve_not_submitted(full_id(ctx, job_id)))

"""Agent commands emit one compact JSON object, without a global --json flag."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from ..agent import ERRORS, AgentClient
from ..models import JobSpec
from ..providers import short
from .options import workload_options

agent_app = typer.Typer(
    no_args_is_help=True,
    help="Bounded JSON API for agents; each command prints one JSON object. Scheduling stays in the "
    "worker. Workflow and permissions: skills/compute-runner/SKILL.md.",
)
JobIds = Annotated[list[str] | None, typer.Argument()]
Batch = Annotated[str | None, typer.Option("--batch")]
States = Annotated[list[str] | None, typer.Option("--state", help="Only jobs in this state; repeatable")]
Resource = Annotated[str | None, typer.Option(help="cpu or gpu")]


def response(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except ERRORS as error:
            detail = error.args[0] if isinstance(error, KeyError) and error.args else error  # no repr quotes
            typer.echo(json.dumps({"schema_version": 1, "error": short(detail)}, ensure_ascii=False))
            raise typer.Exit(1) from None
        typer.echo(json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    return wrapped


def agent(ctx: typer.Context) -> AgentClient:
    return ctx.obj["client"].agent()


# Reading state --------------------------------------------------------------------------------


@agent_app.command("overview")
@response
def overview(ctx: typer.Context, limit: int = 10):
    """Start here: worker, job counts, each account's slots and GPU time, what runs, what waits on someone."""
    return agent(ctx).overview(limit=limit)


@agent_app.command("status")
@response
def status(
    ctx: typer.Context,
    job_ids: JobIds = None,
    batch: Batch = None,
    state: States = None,
    account: Annotated[str | None, typer.Option(help="Only this account's jobs")] = None,
    resource: Resource = None,
    limit: int = 20,
    offset: int = 0,
):
    """Local status for a batch, specific jobs, or all jobs (newest first); paginated, with counts."""
    return agent(ctx).status(
        job_ids, batch_id=batch, states=state, account=account, resource=resource, limit=limit, offset=offset
    )


@agent_app.command("changes")
@response
def changes(ctx: typer.Context, after: int = 0, batch: Batch = None, limit: int = 20):
    """Return only changed jobs. Use the returned cursor as --after; drain has_more."""
    return agent(ctx).changes(after=after, batch_id=batch, limit=limit)


@agent_app.command("accounts")
@response
def accounts(ctx: typer.Context):
    """Accounts in preference order, failover policy, slots in use and free, GPU time and totals; local."""
    return agent(ctx).accounts()


@agent_app.command("running")
@response
def running(
    ctx: typer.Context,
    resource: Resource = None,
    account: Annotated[str | None, typer.Option(help="Only this account")] = None,
):
    """Runs holding each account's slots, including runs started elsewhere (no job_id); local."""
    return agent(ctx).running(resource, account=account)


@agent_app.command("runtime")
@response
def runtime(ctx: typer.Context, job_id: str):
    """Read provider accelerator metadata and session status, without changes."""
    return agent(ctx).runtime(job_id)


@agent_app.command("inputs")
@response
def inputs(ctx: typer.Context, job_id: str):
    """Read the exact pending inputs' provider status without uploading or launching."""
    return agent(ctx).inputs(job_id)


@agent_app.command("logs")
@response
def logs(ctx: typer.Context, job_id: str, tail: int = 50, max_bytes: int = 8192, refresh: bool = False):
    """Bounded log tail plus path to full cached logs; --refresh fetches a fresh remote snapshot."""
    return agent(ctx).logs(job_id, tail=tail, max_bytes=max_bytes, refresh=refresh)


@agent_app.command("wait")
@response
def wait(
    ctx: typer.Context,
    job_ids: JobIds = None,
    batch: Batch = None,
    timeout: float = 300,
    downloads: Annotated[bool, typer.Option("--downloads/--no-downloads")] = True,
    limit: int = 20,
):
    """Block until the selected jobs settle or --timeout seconds pass; check timed_out, not the exit code."""
    return agent(ctx).wait(job_ids, batch_id=batch, timeout=timeout, downloads=downloads, limit=limit)


@agent_app.command("outputs")
@response
def outputs(ctx: typer.Context, job_id: str, limit: int = 100, offset: int = 0):
    """List downloaded files (paths relative to root, the run folder), the run log and job.json."""
    return agent(ctx).outputs(job_id, limit=limit, offset=offset)


@agent_app.command("cleanup")
@response
def cleanup(
    ctx: typer.Context,
    older_than: Annotated[float, typer.Option(help="Days since a job finished")] = 7,
    account: Annotated[str | None, typer.Option(help="Only this account")] = None,
    local: Annotated[bool, typer.Option("--local/--no-local")] = True,
    remote: Annotated[bool, typer.Option("--remote/--no-remote")] = True,
    limit: int = 20,
):
    """Report what the runner left behind and what could be deleted; deleting is the user's call."""
    return agent(ctx).cleanup(
        older_than_days=older_than, account=account, local=local, remote=remote, limit=limit
    )


@agent_app.command("health")
@response
def health(ctx: typer.Context):
    """Read worker lock and heartbeat without remote calls."""
    return {"schema_version": 1, "worker": agent(ctx).health()}


# Changing work --------------------------------------------------------------------------------


@agent_app.command("submit")
@response
@workload_options
def submit(
    ctx: typer.Context,
    source: Path,
    specs: list[JobSpec],
    request_key: Annotated[
        str, typer.Option(help="Stable key for this request; reuse it after an interrupted call")
    ],
    dry_run: bool = False,
    account: Annotated[
        str | None, typer.Option(help="Account ID from agent accounts; default: the first account")
    ] = None,
):
    """Submit a script, notebook, project YAML, or jobs: YAML atomically. Uses the account's compute.

    Each job's run folder is fixed now and returned as run_dir; the worker fills it.
    """
    if dry_run:
        return agent(ctx).preview(specs, account=account)
    return agent(ctx).submit(specs, request_key=request_key, account=account)


@agent_app.command("retry")
@response
def retry(
    ctx: typer.Context,
    request_key: Annotated[str, typer.Option()],
    job_ids: JobIds = None,
    batch: Batch = None,
    state: States = None,
    account: Annotated[str | None, typer.Option(help="Account ID; default: each job's account")] = None,
):
    """Rerun jobs' saved code as new jobs when the user wants a rerun; --batch with --state failed reruns
    a batch's failures. All are queued together or none. The key makes it safe to replay.
    """
    return agent(ctx).retry(job_ids, request_key=request_key, account=account, batch_id=batch, states=state)


@agent_app.command("continue")
@response
def continue_run(
    ctx: typer.Context,
    job_id: str,
    request_key: Annotated[str, typer.Option()],
    account: Annotated[str | None, typer.Option(help="Account ID; default: the job's account")] = None,
):
    """Resume a stopped resumable job from its verified checkpoint as the next run of its experiment."""
    return agent(ctx).continue_run(job_id, request_key=request_key, account=account)


@agent_app.command("move")
@response
def move(
    ctx: typer.Context,
    account: Annotated[str, typer.Option(help="Target account ID from agent accounts")],
    job_ids: JobIds = None,
    batch: Batch = None,
    transfer: Annotated[
        bool, typer.Option("--transfer", help="Allow copying datasets the account cannot read (ask the user)")
    ] = False,
    limit: int = 20,
):
    """Move jobs that have not been submitted to another account; submitted ones stay.

    Unless the failover policy is auto, move only after the user approves the account.
    """
    return agent(ctx).move(job_ids, batch_id=batch, account=account, transfer=transfer, limit=limit)


@agent_app.command("cancel")
@response
def cancel(ctx: typer.Context, job_ids: JobIds = None, batch: Batch = None, limit: int = 20):
    """Cancel pending jobs locally, or ask providers to stop running ones. Only on the user's request.

    For several jobs or a batch, not_cancelled lists the ones that could not be cancelled.
    """
    return agent(ctx).cancel(job_ids, batch_id=batch, limit=limit)

"""Agent commands emit one compact JSON object, without a global --json flag."""

from __future__ import annotations

import json
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .agent import short

agent_app = typer.Typer(
    no_args_is_help=True, help="Bounded JSON API for agents. Scheduling stays in the worker."
)
# Operation failures reported as a message; anything else is a bug and keeps its traceback.
ERRORS = (ValueError, KeyError, RuntimeError, OSError, sqlite3.Error, yaml.YAMLError)


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


@agent_app.command("submit")
@response
def submit(
    ctx: typer.Context,
    source: Path,
    request_key: Annotated[
        str, typer.Option(help="Stable key for this request; reuse it after an interrupted call")
    ],
    entrypoint: str | None = None,
    module: str | None = None,
    gpu: Annotated[bool | None, typer.Option("--gpu/--cpu")] = None,
    internet: Annotated[bool | None, typer.Option("--internet/--no-internet")] = None,
    accelerator: str | None = None,
    timeout: int | None = None,
    arg: Annotated[list[str] | None, typer.Option("--arg")] = None,
    dry_run: bool = False,
    account: Annotated[
        str | None, typer.Option(help="Account ID from agent accounts; default: the first account")
    ] = None,
):
    """Submit a script, notebook, project YAML, or jobs: YAML atomically."""
    from .cli import workload_specs

    specs = workload_specs(
        source,
        entrypoint=entrypoint,
        module=module,
        gpu=gpu,
        internet=internet,
        accelerator=accelerator,
        timeout=timeout,
        arg=arg,
    )
    if dry_run:
        return ctx.obj["client"].agent().preview(specs)
    return ctx.obj["client"].agent().submit(specs, request_key=request_key, account=account)


@agent_app.command("status")
@response
def status(
    ctx: typer.Context,
    job_ids: Annotated[list[str] | None, typer.Argument()] = None,
    batch: str | None = None,
    state: Annotated[list[str] | None, typer.Option("--state")] = None,
    limit: int = 20,
    offset: int = 0,
):
    """Local status for a batch, specific jobs, or all jobs; paginated, with aggregate counts."""
    return ctx.obj["client"].agent().status(job_ids, batch_id=batch, states=state, limit=limit, offset=offset)


@agent_app.command("changes")
@response
def changes(ctx: typer.Context, after: int = 0, batch: str | None = None, limit: int = 20):
    """Return only changed jobs. Use the returned cursor as --after; drain has_more."""
    return ctx.obj["client"].agent().changes(after=after, batch_id=batch, limit=limit)


@agent_app.command("logs")
@response
def logs(ctx: typer.Context, job_id: str, tail: int = 50, max_bytes: int = 8192, refresh: bool = False):
    """Bounded log tail plus path to full cached logs; --refresh fetches a fresh remote snapshot."""
    return ctx.obj["client"].agent().logs(job_id, tail=tail, max_bytes=max_bytes, refresh=refresh)


@agent_app.command("wait")
@response
def wait(
    ctx: typer.Context,
    job_ids: Annotated[list[str] | None, typer.Argument()] = None,
    batch: str | None = None,
    timeout: float = 300,
    downloads: Annotated[bool, typer.Option("--downloads/--no-downloads")] = True,
    limit: int = 20,
):
    """Block until the selected jobs settle or --timeout seconds pass; check timed_out, not the exit code."""
    agent = ctx.obj["client"].agent()
    return agent.wait(job_ids, batch_id=batch, timeout=timeout, downloads=downloads, limit=limit)


@agent_app.command("outputs")
@response
def outputs(ctx: typer.Context, job_id: str, limit: int = 100, offset: int = 0):
    """List downloaded output files (paths relative to root) and the run log path."""
    return ctx.obj["client"].agent().outputs(job_id, limit=limit, offset=offset)


@agent_app.command("retry")
@response
def retry(
    ctx: typer.Context,
    job_id: str,
    request_key: Annotated[str, typer.Option()],
    account: Annotated[str | None, typer.Option(help="Account ID; default: the job's account")] = None,
):
    """Explicitly rerun saved code with a new request key, safe to replay."""
    return ctx.obj["client"].agent().retry(job_id, request_key=request_key, account=account)


@agent_app.command("accounts")
@response
def accounts(ctx: typer.Context):
    """Accounts in preference order, failover policy, slots in use and last known GPU quota; local only."""
    return ctx.obj["client"].agent().accounts()


@agent_app.command("move")
@response
def move(
    ctx: typer.Context,
    account: Annotated[str, typer.Option(help="Target account ID from agent accounts")],
    job_ids: Annotated[list[str] | None, typer.Argument()] = None,
    batch: str | None = None,
    limit: int = 20,
):
    """Move jobs that have not been submitted to another account; submitted ones stay."""
    return ctx.obj["client"].agent().move(job_ids, batch_id=batch, account=account, limit=limit)


@agent_app.command("cancel")
@response
def cancel(ctx: typer.Context, job_id: str):
    """Cancel a pending job locally, or ask its provider to stop a running one."""
    return ctx.obj["client"].agent().cancel(job_id)


@agent_app.command("health")
@response
def health(ctx: typer.Context):
    """Read worker lock and heartbeat without remote calls."""
    return {"schema_version": 1, "worker": ctx.obj["client"].agent().health()}

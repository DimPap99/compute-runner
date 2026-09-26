"""Agent commands emit one compact JSON object, without a global --json flag."""

from __future__ import annotations

import json
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from .agent import short

agent_app = typer.Typer(
    no_args_is_help=True, help="Bounded JSON API for agents. Scheduling stays in the worker."
)


def response(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except (ValueError, KeyError, RuntimeError, OSError, sqlite3.Error, yaml.YAMLError) as error:
            if isinstance(error, ValidationError):
                detail = error.errors(include_input=False, include_url=False)[0]
                message = f"Invalid {'.'.join(map(str, detail['loc']))}: {detail['msg']}"
            elif isinstance(error, KeyError) and error.args:
                message = str(error.args[0])  # str(KeyError) would add repr quotes
            else:
                message = str(error)
            typer.echo(json.dumps({"schema_version": 1, "error": short(message)}, ensure_ascii=False))
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
):
    """Submit a script, notebook, project YAML, or jobs: YAML atomically."""
    from .cli import load_specs, override_specs

    overrides = dict(
        entrypoint=entrypoint,
        module=module,
        gpu=gpu,
        internet=internet,
        accelerator=accelerator,
        timeout_seconds=timeout,
        args=arg,
    )
    specs = override_specs(load_specs(source), overrides)
    if dry_run:
        return ctx.obj["client"].agent().preview(specs)
    return ctx.obj["client"].agent().submit(specs, request_key=request_key)


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
def retry(ctx: typer.Context, job_id: str, request_key: Annotated[str, typer.Option()]):
    """Explicitly rerun saved code with a new request key, safe to replay."""
    return ctx.obj["client"].agent().retry(job_id, request_key=request_key)


@agent_app.command("cancel")
@response
def cancel(ctx: typer.Context, job_id: str):
    """Cancel a locally pending job. Active remote execution must be stopped on Kaggle."""
    return ctx.obj["client"].agent().cancel(job_id)


@agent_app.command("health")
@response
def health(ctx: typer.Context):
    """Read worker lock and heartbeat without contacting Kaggle."""
    return {"schema_version": 1, "worker": ctx.obj["client"].agent().health()}

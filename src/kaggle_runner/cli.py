"""Human-friendly CLI; --json provides machine-readable output."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .client import Client
from .models import Config, JobSpec
from .store import atomic_json, config_path
from . import service

app = typer.Typer(no_args_is_help=True, help="Queue, run and monitor private Kaggle workloads.")
worker_app = typer.Typer(no_args_is_help=True)
service_app = typer.Typer(no_args_is_help=True)
app.add_typer(worker_app, name="worker")
app.add_typer(service_app, name="service")
console = Console()


@app.callback()
def context(
    ctx: typer.Context,
    state_dir: Annotated[Path | None, typer.Option(help="Override persistent state directory")] = None,
    config_dir: Annotated[Path | None, typer.Option(help="Override configuration directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON")] = False,
):
    if config_dir:
        os.environ["KGR_CONFIG_DIR"] = str(config_dir.expanduser().resolve())
    ctx.obj = {"client": Client(state_dir=state_dir), "json": json_output}


def _client(ctx):
    return ctx.obj["client"]


def _job_dict(job):
    return job.model_dump(mode="json") | {"url": job.url, "remote_ref": job.remote_ref}


def _emit(ctx, value):
    if hasattr(value, "model_dump"):
        value = _job_dict(value)
    if ctx.obj["json"]:
        typer.echo(json.dumps(value, ensure_ascii=False))
    else:
        console.print_json(json.dumps(value, ensure_ascii=False))


def _id(ctx, value):
    matches = [job.id for job in _client(ctx).list() if job.id.startswith(value)]
    if len(matches) != 1:
        raise ValueError(f"Expected one matching job for {value}; found {len(matches)}")
    return matches[0]


def load_specs(path: Path):
    path = path.expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"} and path.is_file():
        data = yaml.safe_load(path.read_text())
        rows = data["jobs"] if isinstance(data, dict) and "jobs" in data else [data]
        if not isinstance(rows, list) or not rows:
            raise ValueError("A workload YAML must contain a job mapping or a nonempty jobs list")
        result = []
        for row in rows:
            spec = JobSpec.model_validate(row)
            if not spec.source.is_absolute():
                spec.source = path.parent / spec.source
            spec.inputs = {
                key: value if value.is_absolute() else path.parent / value
                for key, value in spec.inputs.items()
            }
            result.append(spec)
        return result
    return [JobSpec(source=path, name=path.stem)]


@app.command("init")
def initialize(
    ctx: typer.Context,
    owner: Annotated[str, typer.Option()],
    cpu_limit: int = 5,
    gpu_limit: int = 1,
    poll_seconds: float = 30,
):
    values = _client(ctx).config.model_dump()
    values.update(owner=owner, cpu_limit=cpu_limit, gpu_limit=gpu_limit, poll_seconds=poll_seconds)
    config = Config.model_validate(values)
    Client(config=config)  # Validate the account binding before saving configuration.
    atomic_json(config_path(), config.model_dump(mode="json"))
    _emit(ctx, dict(config=str(config_path()), owner=owner, state_dir=str(config.state_dir)))


@app.command()
def submit(
    ctx: typer.Context,
    source: Path,
    entrypoint: str | None = None,
    module: str | None = None,
    gpu: Annotated[bool | None, typer.Option("--gpu/--cpu")] = None,
    internet: Annotated[bool | None, typer.Option("--internet/--no-internet")] = None,
    accelerator: str | None = None,
    timeout: int | None = None,
    arg: Annotated[list[str] | None, typer.Option("--arg")] = None,
    dry_run: bool = False,
):
    specs = load_specs(source)
    overrides = dict(
        entrypoint=entrypoint,
        module=module,
        gpu=gpu,
        internet=internet,
        accelerator=accelerator,
        timeout_seconds=timeout,
        args=arg,
    )
    specs = [
        JobSpec.model_validate(spec.model_dump() | {k: v for k, v in overrides.items() if v is not None})
        for spec in specs
    ]
    if dry_run:
        _emit(ctx, [_client(ctx).preview(spec) for spec in specs])
    else:
        jobs = _client(ctx).submit_many(specs)
        if ctx.obj["json"]:
            _emit(ctx, [_job_dict(job) for job in jobs])
        else:
            for job in jobs:
                console.print(f"Queued {job.id} ({job.spec.name})")
        if not ctx.obj["json"] and not _client(ctx).worker_health()["running"]:
            console.print("Queued locally. Start processing with: kgr service start (or kgr worker run)")


@app.command("list")
def list_jobs(ctx: typer.Context, state: str | None = None):
    jobs = _client(ctx).list(states={state} if state else None)
    if ctx.obj["json"]:
        _emit(ctx, [_job_dict(job) for job in jobs])
        return
    table = Table("Job", "Name", "State", "Resource", "Outputs", "Waiting / error")
    for job in jobs:
        table.add_row(
            job.id[:12],
            job.spec.name,
            job.state,
            job.spec.accelerator or ("GPU" if job.spec.gpu else "CPU"),
            job.download_state,
            job.error or job.wait_reason or "",
        )
    console.print(table)


@app.command()
def status(ctx: typer.Context, job_id: str):
    _emit(ctx, _client(ctx).get(_id(ctx, job_id)))


@app.command()
def watch(ctx: typer.Context, job_id: str):
    job_id = _id(ctx, job_id)
    while True:
        job = _client(ctx).get(job_id)
        _emit(
            ctx,
            dict(
                id=job.id,
                state=job.state,
                remote_state=job.remote_state,
                url=job.url,
                elapsed_seconds=round((job.finished_at or time.time()) - job.created_at),
                wait_reason=job.wait_reason,
                error=job.error,
                download_state=job.download_state,
            ),
        )
        if job.terminal or job.state in {"blocked", "needs_attention"}:
            return
        time.sleep(_client(ctx).config.poll_seconds)


@app.command()
def wait(
    ctx: typer.Context,
    job_id: str,
    timeout: float | None = None,
    downloads: Annotated[bool, typer.Option("--downloads/--no-downloads")] = True,
):
    job = _client(ctx).wait(_id(ctx, job_id), timeout=timeout, downloads=downloads)
    _emit(ctx, job)
    if job.state != "succeeded":
        raise typer.Exit(1)


@app.command()
def logs(ctx: typer.Context, job_id: str, follow: bool = False):
    for chunk in _client(ctx).logs(_id(ctx, job_id), follow=follow):
        if ctx.obj["json"]:
            typer.echo(json.dumps({"data": chunk}))
        else:
            typer.echo(chunk, nl=False)
            sys.stdout.flush()


@app.command()
def download(ctx: typer.Context, job_id: str):
    _emit(ctx, _client(ctx).download(_id(ctx, job_id)))


@app.command()
def retry(ctx: typer.Context, job_id: str):
    _emit(ctx, _client(ctx).retry(_id(ctx, job_id)))


@app.command()
def cancel(ctx: typer.Context, job_id: str):
    _emit(ctx, _client(ctx).cancel(_id(ctx, job_id)))


@app.command()
def resolve(
    ctx: typer.Context,
    job_id: str,
    not_submitted: Annotated[
        bool, typer.Option(help="Assert you independently verified no remote run exists")
    ] = False,
):
    if not not_submitted:
        raise ValueError(
            "Inspect the notebook on Kaggle first; --not-submitted is an explicit operator assertion"
        )
    _emit(ctx, _client(ctx).resolve_not_submitted(_id(ctx, job_id)))


@app.command()
def quota(ctx: typer.Context):
    _emit(ctx, _client(ctx).quota())


@app.command()
def doctor(ctx: typer.Context, offline: bool = False):
    client = _client(ctx)
    info = dict(
        owner=client.config.owner,
        state_dir=str(client.config.state_dir),
        free_disk_bytes=shutil.disk_usage(client.config.state_dir).free,
        worker=client.worker_health(),
        python=sys.version.split()[0],
    )
    if not offline:
        info["quota"] = client.quota()
        info["active_runs"] = client.backend.active_runs()
    _emit(ctx, info)


@worker_app.command("run")
def worker_run(ctx: typer.Context, once: bool = False):
    if not _client(ctx).config.owner:
        raise ValueError("Run kgr init --owner YOUR_USERNAME first")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = _client(ctx).worker()
    worker.tick() if once else worker.run()


@worker_app.command("status")
def worker_status(ctx: typer.Context):
    _emit(ctx, _client(ctx).worker_health())


@service_app.command("install")
def service_install(ctx: typer.Context, start: Annotated[bool, typer.Option("--start/--no-start")] = True):
    if not _client(ctx).config.owner:
        raise ValueError("Run kgr init --owner YOUR_USERNAME first")
    # Persist a state-dir override so API clients and the service use the same queue.
    atomic_json(config_path(), _client(ctx).config.model_dump(mode="json"))
    _emit(ctx, {"unit": str(service.install(_client(ctx).config, start=start))})


@service_app.command("start")
def service_start():
    service.control("start")


@service_app.command("stop")
def service_stop():
    service.control("stop")


@service_app.command("restart")
def service_restart():
    service.control("restart")


@service_app.command("status")
def service_status(ctx: typer.Context):
    if not ctx.obj["json"]:
        service.control("status")
    _emit(ctx, _client(ctx).worker_health())


def main():
    try:
        app()
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        from .backend import safe_message

        typer.echo("Error: " + safe_message(error), err=True)
        raise SystemExit(1) from None

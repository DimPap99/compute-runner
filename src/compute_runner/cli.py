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
from .models import Account, Config, JobSpec
from .providers import safe_message
from .security import redacted_env_record
from .store import atomic_json, config_path
from . import service

app = typer.Typer(
    no_args_is_help=True,
    help="Queue, run and monitor compute workloads on your provider accounts. Kaggle is the only provider "
    "adapter today.",
)
worker_app = typer.Typer(no_args_is_help=True)
service_app = typer.Typer(no_args_is_help=True)
account_app = typer.Typer(no_args_is_help=True, help="Connect provider accounts; the first is the default.")
app.add_typer(worker_app, name="worker")
app.add_typer(service_app, name="service")
app.add_typer(account_app, name="account")
console = Console()


@app.callback()
def context(
    ctx: typer.Context,
    state_dir: Annotated[Path | None, typer.Option(help="Override persistent state directory")] = None,
    config_dir: Annotated[Path | None, typer.Option(help="Override configuration directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON")] = False,
):
    if config_dir:
        os.environ["COMPUTE_RUNNER_CONFIG_DIR"] = str(config_dir.expanduser().resolve())
    ctx.obj = {"client": Client(state_dir=state_dir), "json": json_output}


def _client(ctx):
    return ctx.obj["client"]


def _job_dict(job):
    return redacted_env_record(job.model_dump(mode="json")) | {
        "url": job.url,
        "remote_ref": job.remote_ref,
    }


def _emit(ctx, value):
    if hasattr(value, "model_dump"):
        value = _job_dict(value)
    if ctx.obj["json"]:
        typer.echo(json.dumps(value, ensure_ascii=False))
    else:
        console.print_json(json.dumps(value, ensure_ascii=False))


def _id(ctx, value):
    return _client(ctx).store.resolve_id(value)


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
            # Joining keeps absolute paths as they are.
            spec.source = path.parent / spec.source.expanduser()
            spec.inputs = {key: path.parent / value.expanduser() for key, value in spec.inputs.items()}
            result.append(spec)
        return result
    return [JobSpec(source=path, name=path.stem)]


def workload_specs(source: Path, *, timeout=None, arg=None, **overrides):
    """Load a workload and apply the submit commands' overrides to every job."""
    overrides |= dict(timeout_seconds=timeout, args=arg)
    values = {key: value for key, value in overrides.items() if value is not None}
    if overrides["gpu"] is False:
        if overrides["accelerator"] is not None:
            raise ValueError("--cpu cannot be combined with a GPU accelerator")
        values["accelerator"] = None
    return [JobSpec.model_validate(spec.model_dump() | values) for spec in load_specs(source)]


def _save(ctx, **changes):
    """Validate and persist configuration; fields that are not changed keep their current values."""
    changes = {key: value for key, value in changes.items() if value is not None}
    config = Config.model_validate(_client(ctx).config.model_dump() | changes)
    atomic_json(config_path(), config.model_dump(mode="json"))
    return config


@app.command("init")
def initialize(
    ctx: typer.Context,
    failover: Annotated[
        str | None,
        typer.Option(
            help="When a job cannot start on its account: off, ask (suggest another account) or auto "
            "(move it). Default ask; unchanged when omitted"
        ),
    ] = None,
    poll_seconds: Annotated[float | None, typer.Option(help="Default 30; unchanged when omitted")] = None,
    strict: Annotated[
        bool | None,
        typer.Option(
            "--strict/--no-strict",
            help="Broad log redaction and locked-down downloads. Default off; unchanged when omitted",
        ),
    ] = None,
):
    config = _save(ctx, failover=failover, poll_seconds=poll_seconds, strict=strict)
    _emit(
        ctx,
        dict(
            config=str(config_path()),
            state_dir=str(config.state_dir),
            failover=config.failover,
            strict=config.strict,
        ),
    )


@account_app.command("add")
def account_add(
    ctx: typer.Context,
    provider: str,
    user: str,
    credentials: Annotated[
        Path | None,
        typer.Option(help="Credentials file for this account only; default: the provider's usual location"),
    ] = None,
    cpu_limit: Annotated[int | None, typer.Option(help="Default 5; unchanged when omitted")] = None,
    gpu_limit: Annotated[int | None, typer.Option(help="Default 1; unchanged when omitted")] = None,
    default: Annotated[bool, typer.Option("--default", help="Prefer this account to the others")] = False,
):
    """Add or update an account. Only the credentials file's path is saved."""
    if credentials is not None:
        credentials = credentials.expanduser().resolve()
        if not credentials.is_file():
            raise ValueError(f"Credentials file not found: {credentials}")
    accounts = _client(ctx).config.accounts
    key = Account(provider=provider, user=user).id.casefold()
    index = next((i for i, a in enumerate(accounts) if a.id.casefold() == key), len(accounts))
    saved = accounts[index].model_dump() if index < len(accounts) else {}
    changes = dict(credentials=credentials, cpu_limit=cpu_limit, gpu_limit=gpu_limit)
    values = saved | dict(provider=provider, user=user) | {k: v for k, v in changes.items() if v is not None}
    account = Account.model_validate(values)
    others = [a.model_dump() for a in accounts if a.id.casefold() != key]
    others.insert(0 if default else index, account.model_dump())
    config = _save(ctx, accounts=others)
    _emit(ctx, dict(account=account.id, accounts=[a.id for a in config.accounts]))


@account_app.command("remove")
def account_remove(ctx: typer.Context, account_id: str):
    """Forget an account. Its unfinished jobs must be moved, cancelled or finished first."""
    client = _client(ctx)
    account = client.config.account(account_id)
    # Downloads of finished runs still need the account.
    busy = sum(
        job.account == account.id and (not job.terminal or job.download_state in {"pending", "downloading"})
        for job in client.list()
    )
    if busy:
        raise ValueError(f"{busy} unfinished jobs use {account.id}; move, cancel or finish them first")
    config = _save(ctx, accounts=[a.model_dump() for a in client.config.accounts if a.id != account.id])
    _emit(ctx, dict(removed=account.id, accounts=[a.id for a in config.accounts]))


@account_app.command("list")
def account_list(ctx: typer.Context):
    """Accounts in preference order with slots in use; no remote calls."""
    _emit(ctx, _client(ctx).agent().accounts())


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
    request_key: str | None = None,
    account: Annotated[str | None, typer.Option(help="Account ID such as kaggle:USER; default first")] = None,
):
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
        _emit(ctx, [_client(ctx).preview(spec) for spec in specs])
    else:
        jobs = _client(ctx).submit_many(specs, request_key=request_key, account=account)
        if ctx.obj["json"]:
            _emit(ctx, [_job_dict(job) for job in jobs])
        else:
            for job in jobs:
                console.print(f"Queued {job.id} ({job.spec.name})")
        if not ctx.obj["json"] and not _client(ctx).worker_health()["running"]:
            console.print("Queued locally. Start processing with: compute-runner service start (or compute-runner worker run)")


@app.command("list")
def list_jobs(ctx: typer.Context, state: str | None = None):
    jobs = _client(ctx).list(states={state} if state else None)
    if ctx.obj["json"]:
        _emit(ctx, [_job_dict(job) for job in jobs])
        return
    table = Table("Job", "Name", "State", "Account", "Resource", "Outputs", "Waiting / error")
    for job in jobs:
        table.add_row(
            job.id[:12],
            job.spec.name,
            job.state,
            job.account,
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
def retry(ctx: typer.Context, job_id: str, account: str | None = None):
    _emit(ctx, _client(ctx).retry(_id(ctx, job_id), account=account))


@app.command()
def move(ctx: typer.Context, job_id: str, account: Annotated[str, typer.Option()]):
    """Place a job that has not been submitted on another account."""
    _emit(ctx, _client(ctx).move(_id(ctx, job_id), account))


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
            "Inspect the run on its provider first; --not-submitted is an explicit operator assertion"
        )
    _emit(ctx, _client(ctx).resolve_not_submitted(_id(ctx, job_id)))


@app.command()
def quota(ctx: typer.Context, account: str | None = None):
    _emit(ctx, _client(ctx).quota(account))


@app.command()
def doctor(ctx: typer.Context, offline: bool = False):
    client = _client(ctx)
    info = dict(
        accounts=[account.id for account in client.config.accounts],
        failover=client.config.failover,
        state_dir=str(client.config.state_dir),
        strict=client.config.strict,
        free_disk_bytes=shutil.disk_usage(client.config.state_dir).free,
        worker=client.worker_health(),
        python=sys.version.split()[0],
    )
    if not offline:
        # Checked separately, so one broken account does not hide the others.
        info["remote"] = {}
        for account in client.config.accounts:
            provider = client.provider(account.id)
            try:
                info["remote"][account.id] = dict(quota=provider.quota(), active_runs=provider.active_runs())
            except ERRORS as error:
                info["remote"][account.id] = dict(error=safe_message(error))
    _emit(ctx, info)


@worker_app.command("run")
def worker_run(ctx: typer.Context, once: bool = False):
    _client(ctx).config.account()  # Fails with setup guidance when no account is configured.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = _client(ctx).worker()
    worker.tick() if once else worker.run()


@worker_app.command("status")
def worker_status(ctx: typer.Context):
    _emit(ctx, _client(ctx).worker_health())


@service_app.command("install")
def service_install(ctx: typer.Context, start: Annotated[bool, typer.Option("--start/--no-start")] = True):
    _client(ctx).config.account()  # Fails with setup guidance when no account is configured.
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
    except ERRORS as error:
        typer.echo("Error: " + safe_message(error), err=True)
        raise SystemExit(1) from None


# Import after load_specs is defined; the agent CLI reuses workload-file parsing.
from .agent_cli import ERRORS, agent_app  # noqa: E402

app.add_typer(agent_app, name="agent")

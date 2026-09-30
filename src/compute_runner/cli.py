"""Human-friendly CLI; --json provides machine-readable output."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import service
from .agent import ERRORS
from .agent_cli import agent_app
from .client import Client
from .credentials import account_secrets, credentials_path, kaggle_secrets, save_secrets
from .models import Account, Config
from .providers import connect, safe_message
from .security import redacted_env_record
from .store import atomic_json, config_path, load_config
from .workloads import workload_specs

app = typer.Typer(
    no_args_is_help=True,
    help="Queue, run and monitor compute workloads on your Kaggle accounts and SSH machines. "
    "LLM agents should use the bounded 'agent' commands.",
)
worker_app = typer.Typer(no_args_is_help=True, help="Run the scheduler in this terminal or inspect it.")
service_app = typer.Typer(no_args_is_help=True, help="Manage the worker as a systemd user service.")
account_app = typer.Typer(no_args_is_help=True, help="Connect provider accounts; the first is the default.")
app.add_typer(worker_app, name="worker")
app.add_typer(service_app, name="service")
app.add_typer(account_app, name="account")
app.add_typer(agent_app, name="agent")
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


def _save(ctx, *, unset=(), **changes):
    """Validate and persist configuration; fields that are not changed keep their current values.

    unset names fields restored to their defaults.
    """
    changes = {key: value for key, value in changes.items() if value is not None} | dict.fromkeys(unset)
    # The saved file, not this invocation's configuration: a one-off --state-dir must not persist.
    config = Config.model_validate(load_config().model_dump() | changes)
    atomic_json(config_path(), config.model_dump(mode="json"))
    if _client(ctx).worker_health()["running"]:
        # The worker reads configuration when it starts; stderr keeps --json output parseable.
        typer.echo("Restart the worker to apply this change: compute-runner service restart", err=True)
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
    results_dir: Annotated[
        str | None,
        typer.Option(
            help='Parent folder for experiment folders. Default results/ beside each workload\'s code; "" '
            "restores it; unchanged when omitted"
        ),
    ] = None,
    transfer: Annotated[
        bool | None,
        typer.Option(
            "--transfer/--no-transfer",
            help="Copy a dataset the job's account cannot read from an account that can. Default off; "
            "unchanged when omitted",
        ),
    ] = None,
):
    """Save worker-wide settings: failover, polling, strict mode, results folder and dataset copies."""
    config = _save(
        ctx,
        unset=("results_dir",) if results_dir == "" else (),
        failover=failover,
        poll_seconds=poll_seconds,
        strict=strict,
        results_dir=Path(results_dir).expanduser().absolute() if results_dir else None,
        transfer=transfer,
    )
    _emit(
        ctx,
        dict(
            config=str(config_path()),
            state_dir=str(config.state_dir),
            failover=config.failover,
            strict=config.strict,
            results_dir=str(config.results_dir or "results/ beside each workload's code"),
            transfer=config.transfer,
        ),
    )


def _read_file(path: Path | None, label: str) -> str | None:
    if path is None:
        return None
    path = path.expanduser()
    if not path.is_file():
        raise ValueError(f"{label} not found: {path.absolute()}")
    return path.read_text()


def _secret_input(label: str) -> str:
    """Typed without echo; the prompt goes to stderr, so --json output stays parseable."""
    value = typer.prompt(label, hide_input=True, err=True).strip()
    if not value:
        raise ValueError(f"{label} is empty")
    return value


def _account_secrets(provider, user, *, credentials, enter_key, key, password_file, enter_password):
    """Secrets given to account add, for the credentials file; None when none were given."""
    kaggle = {"--credentials": credentials, "--enter-key": enter_key or None}
    ssh = {"--key": key, "--password-file": password_file, "--enter-password": enter_password or None}
    for flags, owner in [(kaggle, "kaggle"), (ssh, "ssh")]:
        given = [flag for flag, value in flags.items() if value is not None]
        if given and provider != owner:
            raise ValueError(
                f"{given[0]} applies to {'Kaggle' if owner == 'kaggle' else 'SSH'} accounts only"
            )
        if len(given) > 1:
            raise ValueError(f"Choose one of {', '.join(flags)}")
    if provider == "kaggle":
        if enter_key:
            return kaggle_secrets(_secret_input("Kaggle API key or access token"))
        if credentials is None:
            return None
        secrets = kaggle_secrets(_read_file(credentials, "Credentials file"))
        if secrets.get("username", user).casefold() != user.casefold():
            raise ValueError(f"{credentials} holds the key of {secrets['username']}, not {user}")
        return secrets
    if key is not None:
        if not key.expanduser().is_file():
            raise ValueError(f"Key file not found: {key.expanduser().absolute()}")
        return {"key": str(key.expanduser().absolute())}
    if password_file is not None:
        return {"password": _read_file(password_file, "Password file").rstrip("\n")}
    if enter_password:
        return {"password": _secret_input(f"Password of {user}")}
    return None


@account_app.command("add")
def account_add(
    ctx: typer.Context,
    provider: Annotated[str, typer.Argument(help="kaggle or ssh")],
    user: Annotated[str, typer.Argument(help="Kaggle username, or a name you choose for an SSH machine")],
    credentials: Annotated[
        Path | None,
        typer.Option(help="Kaggle: a kaggle.json or access-token file to save in the credentials file"),
    ] = None,
    enter_key: Annotated[
        bool, typer.Option("--enter-key", help="Kaggle: type the API key or access token (not shown)")
    ] = False,
    cpu_limit: Annotated[int | None, typer.Option(help="Default 5; unchanged when omitted")] = None,
    gpu_limit: Annotated[
        int | None,
        typer.Option(help="Default 1 on Kaggle, 0 on SSH (GPUs the machine may use); unchanged when omitted"),
    ] = None,
    default: Annotated[bool, typer.Option("--default", help="Prefer this account to the others")] = False,
    host: Annotated[str | None, typer.Option(help="SSH: host name or address")] = None,
    port: Annotated[int | None, typer.Option(help="SSH: port; default 22")] = None,
    login: Annotated[str | None, typer.Option(help="SSH: user name on the machine")] = None,
    key: Annotated[
        Path | None,
        typer.Option(help="SSH: private key file, saved as a path; default: ssh-agent and ~/.ssh keys"),
    ] = None,
    password_file: Annotated[
        Path | None, typer.Option(help="SSH: a file holding the password, to save in the credentials file")
    ] = None,
    enter_password: Annotated[
        bool, typer.Option("--enter-password", help="SSH: type the password (not shown)")
    ] = False,
    workdir: Annotated[
        str | None,
        typer.Option(help="SSH: work directory on the machine, relative to home; default .compute-runner"),
    ] = None,
    python: Annotated[
        str | None, typer.Option(help="SSH: Python 3.9+ on the machine; default python3")
    ] = None,
    trust_new_host: Annotated[
        bool,
        typer.Option("--trust-new-host", help="SSH: accept the machine's host key if it is not known yet"),
    ] = False,
):
    """Add or update an account. Keys, passwords and tokens go to the credentials file only."""
    if trust_new_host and provider != "ssh":
        raise ValueError("--trust-new-host applies to SSH accounts only")
    secrets = _account_secrets(
        provider,
        user,
        credentials=credentials,
        enter_key=enter_key,
        key=key,
        password_file=password_file,
        enter_password=enter_password,
    )
    accounts = _client(ctx).config.accounts
    key_id = f"{provider}:{user}".casefold()
    index = next((i for i, a in enumerate(accounts) if a.id.casefold() == key_id), len(accounts))
    saved = accounts[index].model_dump() if index < len(accounts) else {}
    changes = dict(cpu_limit=cpu_limit, gpu_limit=gpu_limit)
    ssh = dict(host=host, port=port, username=login, workdir=workdir, python=python)
    ssh = {name: value for name, value in ssh.items() if value is not None}
    if ssh and provider != "ssh":
        flag = {"username": "login"}.get(name := next(iter(ssh)), name)
        raise ValueError(f"--{flag} applies to SSH accounts only")
    if provider == "ssh":
        changes["ssh"] = (saved.get("ssh") or {}) | ssh
    # An existing account keeps its saved ID, whatever the casing typed now; its jobs refer to it.
    values = dict(provider=provider, user=user) | saved | {k: v for k, v in changes.items() if v is not None}
    if secrets is not None:  # New secrets replace those an older configuration named by path.
        values["credentials"] = None
        if values.get("ssh"):
            values["ssh"] = values["ssh"] | dict(key=None, password_file=None)
    account = Account.model_validate(values)
    result = dict(account=account.id)
    if secrets is not None:
        result["credentials_file"] = str(save_secrets(account.id, secrets))
    others = [a.model_dump() for a in accounts if a.id.casefold() != key_id]
    others.insert(0 if default else index, account.model_dump())
    config = _save(ctx, accounts=others)
    result["accounts"] = [a.id for a in config.accounts]
    if trust_new_host:
        result["host_key"] = connect(account, config).trust_host()
    _emit(ctx, result)


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
    save_secrets(account.id, None)  # Its key, password or token is forgotten with it.
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
    param: Annotated[
        list[str] | None,
        typer.Option("--param", help="NAME=VALUE passed as --NAME VALUE and recorded with the results"),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Experiment name, and its results folder")] = None,
    input: Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            help="ALIAS=PATH or ALIAS=REFERENCE (kaggle:OWNER/SLUG, ssh:/PATH, job:ID); "
            "read as KGR_INPUT_ALIAS",
        ),
    ] = None,
    requirements: Annotated[
        str | None, typer.Option(help="Requirements file in the source folder to install; needs internet")
    ] = None,
    dry_run: bool = False,
    request_key: str | None = None,
    account: Annotated[str | None, typer.Option(help="Account ID such as kaggle:USER; default first")] = None,
):
    """Queue a script, notebook, project YAML or jobs: YAML; --dry-run lists the files to upload."""
    specs = workload_specs(
        source,
        entrypoint=entrypoint,
        module=module,
        gpu=gpu,
        internet=internet,
        accelerator=accelerator,
        timeout=timeout,
        arg=arg,
        param=param,
        name=name,
        input=input,
        requirements=requirements,
    )
    if dry_run:
        _emit(ctx, [_client(ctx).preview(spec, account) for spec in specs])
    else:
        jobs = _client(ctx).submit_many(specs, request_key=request_key, account=account)
        if ctx.obj["json"]:
            _emit(ctx, [_job_dict(job) for job in jobs])
        else:
            for job in jobs:
                console.print(f"Queued {job.id} ({job.spec.name}): results in {job.result_dir}")
        if not ctx.obj["json"] and not _client(ctx).worker_health()["running"]:
            console.print(
                "Queued locally. Start processing with: compute-runner service start "
                "(or compute-runner worker run)"
            )


@app.command("list")
def list_jobs(ctx: typer.Context, state: str | None = None):
    """List every job, or those in one state."""
    jobs = _client(ctx).list(states={state} if state else None)
    if ctx.obj["json"]:
        _emit(ctx, [_job_dict(job) for job in jobs])
        return
    table = Table("Job", "Name", "Run", "State", "Account", "Resource", "Outputs", "Waiting / error")
    for job in jobs:
        table.add_row(
            job.id[:12],
            job.spec.name,
            job.result_dir.name if job.run is not None else "",
            job.state,
            job.account,
            job.spec.accelerator or ("GPU" if job.spec.gpu else "CPU"),
            job.download_state,
            job.error or job.wait_reason or "",
        )
    console.print(table)


def _duration(seconds: int) -> str:
    """Compact time such as 45s, 12m, 3h 05m or 2d 4h."""
    seconds = max(0, seconds)
    if seconds < 3600:
        return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"
    hours = seconds // 3600
    return f"{hours}h {seconds // 60 % 60:02d}m" if hours < 48 else f"{hours // 24}d {hours % 24}h"


def _discovery_notes(ctx, checks: dict, live: bool):
    """Below a table: accounts whose runs started elsewhere are unknown, or were last read long ago."""
    notes = [
        f"{account}: last check failed: {check['error']}"
        if "error" in check
        else f"{account}: not checked yet"
        for account, check in checks.items()
        if "error" in check or check["checked_age_seconds"] is None
    ]
    ages = [
        check["checked_age_seconds"] for check in checks.values() if check["checked_age_seconds"] is not None
    ]
    # The worker checks accounts only while it has jobs to place, so an idle queue's check ages.
    if ages and max(ages) > _client(ctx).config.discovery_seconds:
        notes.append(f"Accounts last checked up to {_duration(max(ages))} ago")
    if notes and not live:
        notes.append("--live asks the providers now")
    for note in notes:
        console.print(note, style="dim", markup=False, highlight=False)


RESOURCES = {"cpu": "CPU", "gpu": "GPU", "unknown": "?"}
LIVE = typer.Option("--live", help="Ask every provider now instead of reading the worker's last check")


@app.command()
def running(
    ctx: typer.Context,
    resource: Annotated[
        Literal["all", "cpu", "gpu"], typer.Argument(help="Runs holding these slots")
    ] = "all",
    account: Annotated[str | None, typer.Option(help="Only this account")] = None,
    live: Annotated[bool, LIVE] = False,
):
    """Runs holding each account's slots: this queue's jobs and runs started elsewhere."""
    result = _client(ctx).agent().running(None if resource == "all" else resource, account=account, live=live)
    if ctx.obj["json"]:
        _emit(ctx, result)
        return
    if result["runs"]:
        table = Table("Account", "Location", "Notebook / run", "Name", "Type", "State", "Elapsed", "Job")
        # Only the reference and the name give way on a narrow terminal.
        for column in table.columns:
            column.no_wrap = column.header not in ("Notebook / run", "Name")
        for run in result["runs"]:
            ours = "job_id" in run
            cells = [
                run["account"],
                run["provider"],
                run["ref"],
                run["name"] if ours else "(external)",
                run.get("accelerator") or RESOURCES[run["resource"]],
                run.get("state", ""),
                _duration(run["elapsed_seconds"]) if ours else "",
                run["job_id"][:12] if ours else "",
            ]
            table.add_row(*map(Text, cells))
        console.print(table)
    counts = [
        f"{result['counts'][kind]} {RESOURCES[kind]}" for kind in ("gpu", "cpu") if result["counts"][kind]
    ]
    if result["counts"]["unknown"]:
        counts.append(f"{result['counts']['unknown']} of unknown type, counted as both")
    console.print(f"{result['total']} running" + (": " + ", ".join(counts) if counts else ""))
    _discovery_notes(ctx, result["discovery"], live)


@app.command()
def status(ctx: typer.Context, job_id: str):
    """Print a job's full record."""
    _emit(ctx, _client(ctx).get(_id(ctx, job_id)))


@app.command()
def watch(ctx: typer.Context, job_id: str):
    """Print a job's progress every polling interval until it settles."""
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
    """Wait for a job to settle; exits 1 unless it succeeded."""
    job = _client(ctx).wait(_id(ctx, job_id), timeout=timeout, downloads=downloads)
    _emit(ctx, job)
    if job.state != "succeeded":
        raise typer.Exit(1)


@app.command()
def logs(ctx: typer.Context, job_id: str, follow: bool = False):
    """Print a run's log, a snapshot while it runs, or stream it with --follow."""
    for chunk in _client(ctx).logs(_id(ctx, job_id), follow=follow):
        if ctx.obj["json"]:
            typer.echo(json.dumps({"data": chunk}))
        else:
            typer.echo(chunk, nl=False)
            sys.stdout.flush()


@app.command()
def download(ctx: typer.Context, job_id: str):
    """Download a finished run's outputs now; the worker also does this automatically."""
    _emit(ctx, _client(ctx).download(_id(ctx, job_id)))


@app.command()
def retry(ctx: typer.Context, job_id: str, account: str | None = None):
    """Rerun a finished or blocked job's saved code as a new job."""
    _emit(ctx, _client(ctx).retry(_id(ctx, job_id), account=account))


@app.command("continue")
def continue_run(ctx: typer.Context, job_id: str, account: str | None = None):
    """Resume a stopped resumable run from its verified checkpoint as the next run of its experiment."""
    _emit(ctx, _client(ctx).continue_run(_id(ctx, job_id), account=account))


@app.command()
def move(
    ctx: typer.Context,
    job_id: str,
    account: Annotated[str, typer.Option()],
    transfer: Annotated[
        bool, typer.Option("--transfer", help="Allow copying datasets the account cannot read")
    ] = False,
):
    """Place a job that has not been submitted on another account."""
    _emit(ctx, _client(ctx).move(_id(ctx, job_id), account, transfer=transfer))


@app.command()
def cancel(ctx: typer.Context, job_id: str):
    """Cancel a pending job locally, or ask its provider to stop a running one."""
    _emit(ctx, _client(ctx).cancel(_id(ctx, job_id)))


@app.command()
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
    _emit(ctx, _client(ctx).resolve_not_submitted(_id(ctx, job_id)))


@app.command()
def quota(ctx: typer.Context, account: str | None = None):
    """Query accelerator quota for one account, or every account."""
    _emit(ctx, _client(ctx).quota(account))


def _gpu_total(accounts: list[dict]) -> dict:
    """GPU slots and time over every account.

    free and quota_seconds add up the accounts checked successfully (None when there are none);
    complete says whether that is every account.
    """
    known = [account for account in accounts if account["gpu"]["free"] is not None]
    quotas = [account["gpu_quota_seconds"] for account in known if account["gpu_quota_limited"]]
    return dict(
        used=sum(account["gpu"]["used"] for account in accounts),
        limit=sum(account["gpu"]["limit"] for account in accounts),
        free=sum(account["gpu"]["free"] for account in known) if known else None,
        quota_seconds=sum(quotas) if quotas else None,
        complete=len(known) == len(accounts),
    )


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def _gpu_quota(account: dict) -> str:
    if not account["gpu_quota_limited"]:
        return "no limit" if account["gpu"]["limit"] else "-"
    seconds = account["gpu_quota_seconds"]
    return "?" if seconds is None else _hours(seconds)


@app.command()
def gpus(ctx: typer.Context, live: Annotated[bool, LIVE] = False):
    """GPU slots in use and free on each account, the GPU time left, and their totals."""
    result = _client(ctx).agent().accounts(live=live)
    accounts, total = result["accounts"], _gpu_total(result["accounts"])
    if ctx.obj["json"]:
        _emit(ctx, result | {"gpu_total": total})
        return
    table = Table("Account", "Location", "In use", "Free", "Quota left", "Checked")
    for account in accounts:
        gpu, age = account["gpu"], account["checked_age_seconds"]
        cells = [
            account["id"],
            account["provider"],
            f"{gpu['used']}/{gpu['limit']}",
            "?" if gpu["free"] is None else str(gpu["free"]),
            _gpu_quota(account),
            "never" if age is None else f"{_duration(age)} ago",
        ]
        table.add_row(*map(Text, cells))
    # Lower bounds while some account is unknown.
    bound = "" if total["complete"] else ">="
    if total["quota_seconds"] is not None:
        quota = bound + _hours(total["quota_seconds"])
    else:
        quota = "?" if any(account["gpu_quota_limited"] for account in accounts) else "no limit"
    table.add_section()
    table.add_row(
        "Total",
        "",
        f"{total['used']}/{total['limit']}",
        "?" if total["free"] is None else f"{bound}{total['free']}",
        quota,
        "",
        style="bold",
    )
    console.print(table)
    _discovery_notes(ctx, {account["id"]: account for account in accounts}, live)


def _credentials_source(account) -> str:
    """Where an account's login comes from; never the secret itself."""
    try:
        if saved := account_secrets(account.id):
            return "credentials file: " + ", ".join(sorted(saved))
    except ValueError as error:
        return f"error: {error}"
    older = account.credentials or (account.ssh and (account.ssh.key or account.ssh.password_file))
    if older:
        return f"file named in config.json: {older}"
    return (
        "Kaggle's default (~/.kaggle, KAGGLE_*)" if account.provider == "kaggle" else "ssh-agent and ~/.ssh"
    )


@app.command()
def doctor(ctx: typer.Context, offline: bool = False):
    """Check configuration, disk and worker, and each account's quota and active runs unless --offline."""
    client = _client(ctx)
    info = dict(
        accounts=[account.id for account in client.config.accounts],
        credentials_file=str(credentials_path()),
        credentials={account.id: _credentials_source(account) for account in client.config.accounts},
        failover=client.config.failover,
        transfer=client.config.transfer,
        results_dir=str(client.config.results_dir) if client.config.results_dir else None,
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
            try:
                info["remote"][account.id] = client.provider(account.id).diagnose()
            except ERRORS as error:
                info["remote"][account.id] = dict(error=safe_message(error))
    _emit(ctx, info)


@worker_app.command("run")
def worker_run(ctx: typer.Context, once: bool = False):
    """Run the scheduler in the foreground, or one cycle with --once."""
    _client(ctx).config.account()  # Fails with setup guidance when no account is configured.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = _client(ctx).worker()
    worker.tick() if once else worker.run()


@worker_app.command("status")
def worker_status(ctx: typer.Context):
    """Read worker lock and heartbeat."""
    _emit(ctx, _client(ctx).worker_health())


@service_app.command("install")
def service_install(ctx: typer.Context, start: Annotated[bool, typer.Option("--start/--no-start")] = True):
    """Install and enable the worker as a systemd user service."""
    _client(ctx).config.account()  # Fails with setup guidance when no account is configured.
    # Persist a state-dir override so API clients and the service use the same queue.
    atomic_json(config_path(), _client(ctx).config.model_dump(mode="json"))
    _emit(ctx, {"unit": str(service.install(_client(ctx).config, start=start))})


@service_app.command("start")
def service_start():
    """Start the worker service."""
    service.control("start")


@service_app.command("stop")
def service_stop():
    """Stop the worker service; queued jobs wait and remote runs continue."""
    service.control("stop")


@service_app.command("restart")
def service_restart():
    """Restart the worker service, for example to apply configuration changes."""
    service.control("restart")


@service_app.command("status")
def service_status(ctx: typer.Context):
    """Show systemd status and the worker heartbeat."""
    if not ctx.obj["json"]:
        service.control("status")
    _emit(ctx, _client(ctx).worker_health())


def main():
    try:
        app()
    except ERRORS as error:
        typer.echo("Error: " + safe_message(error), err=True)
        raise SystemExit(1) from None

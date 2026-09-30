"""Setting up: worker-wide settings, accounts and their secrets, quota and the setup check."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer

from ..agent import ERRORS
from ..credentials import account_secrets, credentials_path, kaggle_secrets, save_secrets
from ..models import Account
from ..providers import connect, safe_message
from ..store import config_path
from .common import client, output, save_settings
from .views import account_list

account_app = typer.Typer(no_args_is_help=True, help="Connect provider accounts; the first is the default.")
account_app.command("list")(account_list)
# The secret options of account add, by the provider they belong to.
SECRET_FLAGS = {
    "kaggle": ("--credentials", "--enter-key"),
    "ssh": ("--key", "--password-file", "--enter-password"),
}
PROVIDER_NAMES = {"kaggle": "Kaggle", "ssh": "SSH"}


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
    config = save_settings(
        ctx,
        unset=("results_dir",) if results_dir == "" else (),
        failover=failover,
        poll_seconds=poll_seconds,
        strict=strict,
        results_dir=Path(results_dir).expanduser().absolute() if results_dir else None,
        transfer=transfer,
    )
    output(ctx).emit(
        dict(
            config=str(config_path()),
            state_dir=str(config.state_dir),
            failover=config.failover,
            strict=config.strict,
            results_dir=str(config.results_dir or "results/ beside each workload's code"),
            transfer=config.transfer,
        )
    )


# Secrets --------------------------------------------------------------------------------------


def _read_file(path: Path, label: str) -> str:
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


def _check_secret_flags(provider: str, given: dict[str, object]) -> None:
    """At most one way to give secrets, and only one that belongs to the account's provider."""
    used = [flag for flag, value in given.items() if value not in (None, False)]
    for owner, flags in SECRET_FLAGS.items():
        mine = [flag for flag in used if flag in flags]
        if mine and provider != owner:
            raise ValueError(f"{mine[0]} applies to {PROVIDER_NAMES[owner]} accounts only")
        if len(mine) > 1:
            raise ValueError(f"Choose one of {', '.join(flags)}")


def _kaggle_secrets(user: str, credentials: Path | None, enter_key: bool) -> dict | None:
    if enter_key:
        return kaggle_secrets(_secret_input("Kaggle API key or access token"))
    if credentials is None:
        return None
    secrets = kaggle_secrets(_read_file(credentials, "Credentials file"))
    if secrets.get("username", user).casefold() != user.casefold():
        raise ValueError(f"{credentials} holds the key of {secrets['username']}, not {user}")
    return secrets


def _ssh_secrets(
    user: str, key: Path | None, password_file: Path | None, enter_password: bool
) -> dict | None:
    if key is not None:
        if not key.expanduser().is_file():
            raise ValueError(f"Key file not found: {key.expanduser().absolute()}")
        return {"key": str(key.expanduser().absolute())}
    if password_file is not None:
        return {"password": _read_file(password_file, "Password file").rstrip("\n")}
    if enter_password:
        return {"password": _secret_input(f"Password of {user}")}
    return None


# Accounts -------------------------------------------------------------------------------------


def _account_values(
    provider: str, user: str, saved: dict, *, limits: dict, ssh: dict, new_secrets: bool
) -> dict:
    """The account's settings: saved ones, with the options given now applied over them."""
    ssh = {name: value for name, value in ssh.items() if value is not None}
    if ssh and provider != "ssh":
        flag = {"username": "login"}.get(name := next(iter(ssh)), name)
        raise ValueError(f"--{flag} applies to SSH accounts only")
    changes = {name: value for name, value in limits.items() if value is not None}
    if provider == "ssh":
        changes["ssh"] = (saved.get("ssh") or {}) | ssh
    # An existing account keeps its saved ID, whatever the casing typed now; its jobs refer to it.
    values = dict(provider=provider, user=user) | saved | changes
    if new_secrets:  # New secrets replace those an older configuration named by path.
        values["credentials"] = None
        if values.get("ssh"):
            values["ssh"] = values["ssh"] | dict(key=None, password_file=None)
    return values


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
    flags = {
        "--credentials": credentials,
        "--enter-key": enter_key,
        "--key": key,
        "--password-file": password_file,
        "--enter-password": enter_password,
    }
    _check_secret_flags(provider, flags)
    secrets = (
        _kaggle_secrets(user, credentials, enter_key)
        if provider == "kaggle"
        else _ssh_secrets(user, key, password_file, enter_password)
    )
    accounts = client(ctx).config.accounts
    key_id = f"{provider}:{user}".casefold()
    index = next((i for i, a in enumerate(accounts) if a.id.casefold() == key_id), len(accounts))
    saved = accounts[index].model_dump() if index < len(accounts) else {}
    values = _account_values(
        provider,
        user,
        saved,
        limits=dict(cpu_limit=cpu_limit, gpu_limit=gpu_limit),
        ssh=dict(host=host, port=port, username=login, workdir=workdir, python=python),
        new_secrets=secrets is not None,
    )
    account = Account.model_validate(values)
    result = dict(account=account.id)
    if secrets is not None:
        result["credentials_file"] = str(save_secrets(account.id, secrets))
    others = [a.model_dump() for a in accounts if a.id.casefold() != key_id]
    others.insert(0 if default else index, account.model_dump())
    config = save_settings(ctx, accounts=others)
    result["accounts"] = [a.id for a in config.accounts]
    if trust_new_host:
        result["host_key"] = connect(account, config).trust_host()
    output(ctx).emit(result)


def account_remove(ctx: typer.Context, account_id: str):
    """Forget an account. Its unfinished jobs must be moved, cancelled or finished first."""
    found = client(ctx)
    account = found.config.account(account_id)
    # Downloads of finished runs still need the account.
    busy = sum(
        job.account == account.id and (not job.terminal or job.download_state in {"pending", "downloading"})
        for job in found.list()
    )
    if busy:
        raise ValueError(f"{busy} unfinished jobs use {account.id}; move, cancel or finish them first")
    config = save_settings(
        ctx, accounts=[a.model_dump() for a in found.config.accounts if a.id != account.id]
    )
    save_secrets(account.id, None)  # Its key, password or token is forgotten with it.
    output(ctx).emit(dict(removed=account.id, accounts=[a.id for a in config.accounts]))


account_app.command("add")(account_add)
account_app.command("remove")(account_remove)


# Checks ---------------------------------------------------------------------------------------


def quota(ctx: typer.Context, account: str | None = None):
    """Query accelerator quota for one account, or every account."""
    output(ctx).emit(client(ctx).quota(account))


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


def doctor(ctx: typer.Context, offline: bool = False):
    """Check configuration, disk and worker, and each account's quota and active runs unless --offline."""
    found = client(ctx)
    config = found.config
    info = dict(
        accounts=[account.id for account in config.accounts],
        credentials_file=str(credentials_path()),
        credentials={account.id: _credentials_source(account) for account in config.accounts},
        failover=config.failover,
        transfer=config.transfer,
        results_dir=str(config.results_dir) if config.results_dir else None,
        state_dir=str(config.state_dir),
        strict=config.strict,
        free_disk_bytes=shutil.disk_usage(config.state_dir).free,
        worker=found.worker_health(),
        python=sys.version.split()[0],
    )
    if not offline:
        info["remote"] = {account.id: _diagnose(found, account.id) for account in config.accounts}
    output(ctx).emit(info)


def _diagnose(found, account: str) -> dict:
    # Checked separately, so one broken account does not hide the others.
    try:
        return found.provider(account).diagnose()
    except ERRORS as error:
        return dict(error=safe_message(error))

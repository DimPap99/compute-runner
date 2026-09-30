"""What every command module shares: the invocation's client and output, and saving settings."""

from __future__ import annotations

import typer

from ..client import Client
from ..models import Config, checked_states
from ..store import atomic_json, config_path, load_config
from .output import Output


def client(ctx: typer.Context) -> Client:
    return ctx.obj["client"]


def output(ctx: typer.Context) -> Output:
    return ctx.obj["output"]


def full_id(ctx: typer.Context, value: str) -> str:
    """A complete job ID from an unambiguous prefix."""
    return client(ctx).store.resolve_id(value)


def selected(ctx: typer.Context, values, *, batch=None, account=None, states=None) -> list[str]:
    """The selected jobs: named ones (prefixes allowed), a batch's, or an account's, in some states.

    Exactly one of values, batch and account selects.
    """
    if sum(bool(item) for item in (values, batch, account)) != 1:
        raise ValueError("Select jobs by ID, --batch or --account")
    store = client(ctx).store
    named = store.resolve_ids(list(values)) if values else None
    account = account and client(ctx).config.account(account).id
    states = checked_states(states or None)
    _, jobs = store.page(job_ids=named, batch_id=batch, account=account, states=states, limit=-1)
    return [job.id for job in jobs]


def save_settings(ctx: typer.Context, *, unset=(), **changes) -> Config:
    """Validate and persist configuration; fields that are not changed keep their current values.

    unset names fields restored to their defaults.
    """
    changes = {key: value for key, value in changes.items() if value is not None} | dict.fromkeys(unset)
    # The saved file, not this invocation's configuration: a one-off --state-dir must not persist.
    config = Config.model_validate(load_config().model_dump() | changes)
    atomic_json(config_path(), config.model_dump(mode="json"))
    if client(ctx).worker_health()["running"]:
        # The worker reads configuration when it starts; stderr keeps --json output parseable.
        typer.echo("Restart the worker to apply this change: compute-runner service restart", err=True)
    return config

"""Running the scheduler: in this terminal, or as a systemd user service."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from .. import service
from ..store import atomic_json, config_path
from .common import client, output

worker_app = typer.Typer(no_args_is_help=True, help="Run the scheduler in this terminal or inspect it.")
service_app = typer.Typer(no_args_is_help=True, help="Manage the worker as a systemd user service.")


@worker_app.command("run")
def worker_run(ctx: typer.Context, once: bool = False):
    """Run the scheduler in the foreground, or one cycle with --once."""
    client(ctx).config.account()  # Fails with setup guidance when no account is configured.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = client(ctx).worker()
    worker.tick() if once else worker.run()


@worker_app.command("status")
def worker_status(ctx: typer.Context):
    """Read worker lock and heartbeat."""
    output(ctx).emit(client(ctx).worker_health())


@service_app.command("install")
def service_install(ctx: typer.Context, start: Annotated[bool, typer.Option("--start/--no-start")] = True):
    """Install and enable the worker as a systemd user service."""
    config = client(ctx).config
    config.account()  # Fails with setup guidance when no account is configured.
    # Persist a state-dir override so API clients and the service use the same queue.
    atomic_json(config_path(), config.model_dump(mode="json"))
    output(ctx).emit({"unit": str(service.install(config, start=start))})


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
    if not output(ctx).json:
        service.control("status")
    output(ctx).emit(client(ctx).worker_health())

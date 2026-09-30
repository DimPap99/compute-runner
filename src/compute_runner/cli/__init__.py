"""Human-friendly CLI; --json provides machine-readable output. LLM agents use the agent commands.

Commands live in modules by topic (jobs, views, accounts, worker, cleanup); this module
builds the application from them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from ..agent import ERRORS
from ..client import Client
from ..providers import safe_message
from . import accounts, cleanup, jobs, views
from .agent import agent_app
from .output import Output
from .worker import service_app, worker_app

app = typer.Typer(
    no_args_is_help=True,
    help="Queue, run and monitor compute workloads on your Kaggle accounts and SSH machines. "
    "LLM agents should use the bounded 'agent' commands.",
)


@app.callback()
def context(
    ctx: typer.Context,
    state_dir: Annotated[Path | None, typer.Option(help="Override persistent state directory")] = None,
    config_dir: Annotated[Path | None, typer.Option(help="Override configuration directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON")] = False,
):
    if config_dir:
        os.environ["COMPUTE_RUNNER_CONFIG_DIR"] = str(config_dir.expanduser().resolve())
    ctx.obj = {"client": Client(state_dir=state_dir), "json": json_output, "output": Output(json_output)}


COMMANDS = {
    "init": accounts.initialize,
    "submit": jobs.submit,
    "list": views.list_jobs,
    "running": views.running,
    "gpus": views.gpus,
    "status": jobs.status,
    "watch": jobs.watch,
    "wait": jobs.wait,
    "logs": jobs.logs,
    "download": jobs.download,
    "retry": jobs.retry,
    "continue": jobs.continue_run,
    "move": jobs.move,
    "cancel": jobs.cancel,
    "resolve": jobs.resolve,
    "quota": accounts.quota,
    "doctor": accounts.doctor,
    "cleanup": cleanup.cleanup,
}
for name, command in COMMANDS.items():
    app.command(name)(command)
app.add_typer(worker_app, name="worker")
app.add_typer(service_app, name="service")
app.add_typer(accounts.account_app, name="account")
app.add_typer(agent_app, name="agent")


def main():
    try:
        app()
    except ERRORS as error:
        typer.echo("Error: " + safe_message(error), err=True)
        raise SystemExit(1) from None

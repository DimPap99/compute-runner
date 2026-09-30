"""Options the human and agent commands share, declared once."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Annotated, Literal

import typer

from ..workloads import workload_specs

AccountOption = Annotated[str | None, typer.Option("--account", help="Account ID such as kaggle:USER")]
BatchOption = Annotated[str | None, typer.Option("--batch", help="A batch ID, from submit")]
StatesOption = Annotated[
    list[str] | None, typer.Option("--state", help="Only jobs in this state; repeatable")
]
LiveOption = Annotated[
    bool, typer.Option("--live", help="Ask every provider now instead of reading the worker's last check")
]
ResourceArgument = Annotated[
    Literal["all", "cpu", "gpu"], typer.Argument(help="Jobs or runs holding these slots")
]


def resource(value: str) -> str | None:
    """None for all, else cpu or gpu."""
    return None if value == "all" else value


# The options of every submit command, as workloads.workload_specs takes them.
WORKLOAD = {
    "entrypoint": Annotated[str | None, typer.Option(help="The script or notebook of a folder source")],
    "module": Annotated[str | None, typer.Option(help="A module of a folder source, run as python -m")],
    "gpu": Annotated[bool | None, typer.Option("--gpu/--cpu")],
    "internet": Annotated[bool | None, typer.Option("--internet/--no-internet")],
    "accelerator": Annotated[str | None, typer.Option(help="A provider accelerator ID; implies --gpu")],
    "timeout": Annotated[int | None, typer.Option(help="Seconds before the run is stopped")],
    "arg": Annotated[
        list[str] | None, typer.Option("--arg", help="An argument for the workload; repeatable")
    ],
    "param": Annotated[
        list[str] | None,
        typer.Option("--param", help="NAME=VALUE passed as --NAME VALUE and recorded with the results"),
    ],
    "name": Annotated[str | None, typer.Option(help="Experiment name, and its results folder")],
    "input": Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            help="ALIAS=PATH or ALIAS=REFERENCE (kaggle:OWNER/SLUG, ssh:/PATH, job:ID); "
            "read as KGR_INPUT_ALIAS",
        ),
    ],
    "requirements": Annotated[
        str | None, typer.Option(help="Requirements file in the source folder to install; needs internet")
    ],
}


def workload_options(command):
    """Give a submit command the workload options; it receives the loaded workload as specs.

    The command declares source (a Path argument) and specs (list[JobSpec]); the options in
    WORKLOAD appear between source and its own options, so every submit command takes the same.
    """
    signature = inspect.signature(command, eval_str=True)
    own = [parameter for name, parameter in signature.parameters.items() if name != "specs"]
    split = [parameter.name for parameter in own].index("source") + 1
    keyword = inspect.Parameter.KEYWORD_ONLY
    added = [
        inspect.Parameter(name, keyword, default=None, annotation=kind) for name, kind in WORKLOAD.items()
    ]
    rest = [parameter.replace(kind=keyword) for parameter in own[split:]]

    @functools.wraps(command)
    def wrapped(**options):
        workload = {name: options.pop(name) for name in WORKLOAD}
        return command(specs=workload_specs(Path(options["source"]), **workload), **options)

    wrapped.__signature__ = signature.replace(parameters=[*own[:split], *added, *rest])
    return wrapped

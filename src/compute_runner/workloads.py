"""Workload files and command-line overrides, shared by every front end."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import JobSpec, input_reference


def load_specs(path: Path) -> list[JobSpec]:
    """A script or notebook, a workload YAML with one job mapping, or a YAML jobs list."""
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
            spec.inputs = {
                key: value if input_reference(value) else path.parent / value.expanduser()
                for key, value in spec.inputs.items()
            }
            if spec.results_dir is not None:
                spec.results_dir = path.parent / spec.results_dir.expanduser()
            result.append(spec)
        return result
    return [JobSpec(source=path, name=path.stem)]


_INTEGER = re.compile(r"[-+]?(?:0|[1-9][0-9]*)")
_FLOAT = re.compile(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?=[eE]))(?:[eE][-+]?[0-9]+)?")


def param_value(text: str):
    """true and false as flags, decimal numbers (such as 0.01 or 1e-4) as numbers, anything else as typed.

    YAML 1.1 would read 010 as 8, 1:30 as 90 and yes as true, and pass those to the workload instead.
    """
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if _INTEGER.fullmatch(text):
        return int(text)
    return float(text) if _FLOAT.fullmatch(text) else text


def parse_params(items) -> dict:
    """NAME=VALUE options; see param_value for how values are read."""
    result = {}
    for item in items or []:
        name, separator, text = item.partition("=")
        if not separator:
            raise ValueError(f"--param takes NAME=VALUE, not {item}")
        result[name] = param_value(text)
    return result


def parse_inputs(items) -> dict:
    """ALIAS=VALUE options: a local path (relative to the current folder) or a reference."""
    result = {}
    for item in items or []:
        alias, separator, value = item.partition("=")
        if not separator or not alias or not value:
            raise ValueError(f"--input takes ALIAS=PATH or ALIAS=REFERENCE, not {item}")
        result[alias] = Path(value) if input_reference(value) else Path(value).expanduser().absolute()
    return result


def workload_specs(
    source: Path, *, timeout=None, arg=None, param=None, input=None, **overrides
) -> list[JobSpec]:
    """Load a workload and apply the submit commands' overrides to every job."""
    overrides |= dict(timeout_seconds=timeout, args=arg)
    values = {key: value for key, value in overrides.items() if value is not None}
    if overrides["gpu"] is False:
        if overrides["accelerator"] is not None:
            raise ValueError("--cpu cannot be combined with a GPU accelerator")
        values["accelerator"] = None
    params, inputs = parse_params(param), parse_inputs(input)
    return [
        JobSpec.model_validate(
            spec.model_dump() | values | {"params": spec.params | params, "inputs": spec.inputs | inputs}
        )
        for spec in load_specs(source)
    ]

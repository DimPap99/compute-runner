"""Workload files and command-line overrides, shared by every front end."""

from __future__ import annotations

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


def parse_params(items) -> dict:
    """NAME=VALUE options; values are read as YAML scalars, so 0.01 is a number and true a flag."""
    result = {}
    for item in items or []:
        name, separator, text = item.partition("=")
        if not separator:
            raise ValueError(f"--param takes NAME=VALUE, not {item}")
        value = yaml.safe_load(text) if text else ""
        result[name] = value if isinstance(value, (str, int, float, bool)) else text
    return result


def workload_specs(source: Path, *, timeout=None, arg=None, param=None, **overrides) -> list[JobSpec]:
    """Load a workload and apply the submit commands' overrides to every job."""
    overrides |= dict(timeout_seconds=timeout, args=arg)
    values = {key: value for key, value in overrides.items() if value is not None}
    if overrides["gpu"] is False:
        if overrides["accelerator"] is not None:
            raise ValueError("--cpu cannot be combined with a GPU accelerator")
        values["accelerator"] = None
    params = parse_params(param)
    return [
        JobSpec.model_validate(spec.model_dump() | values | {"params": spec.params | params})
        for spec in load_specs(source)
    ]

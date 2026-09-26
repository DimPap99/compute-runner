"""Workload files and command-line overrides, shared by every front end."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import JobSpec


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
            spec.inputs = {key: path.parent / value.expanduser() for key, value in spec.inputs.items()}
            result.append(spec)
        return result
    return [JobSpec(source=path, name=path.stem)]


def workload_specs(source: Path, *, timeout=None, arg=None, **overrides) -> list[JobSpec]:
    """Load a workload and apply the submit commands' overrides to every job."""
    overrides |= dict(timeout_seconds=timeout, args=arg)
    values = {key: value for key, value in overrides.items() if value is not None}
    if overrides["gpu"] is False:
        if overrides["accelerator"] is not None:
            raise ValueError("--cpu cannot be combined with a GPU accelerator")
        values["accelerator"] = None
    return [JobSpec.model_validate(spec.model_dump() | values) for spec in load_specs(source)]

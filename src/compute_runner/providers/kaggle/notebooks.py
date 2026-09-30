"""Launch notebooks: the private kernel that runs one attempt, built locally from its snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from ...models import JobRecord
from ...security import redacted_env_record
from ...store import atomic_json
from ..base import Provider
from ..launch import launch_config, write_launcher


def prepare_kernel(job: JobRecord, ref: str, folder: Path, state_dir: Path) -> Path:
    """Write the kernel's code and metadata into folder, ready for kernels_push."""
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = launch_config(job, state_dir, _inputs(job), source_ref=job.upload_refs.get("source"))
    code_file = write_launcher(job, config, folder, state_dir)
    atomic_json(folder / "kernel-metadata.json", _metadata(job, ref, code_file))
    atomic_json(folder / "provenance.json", redacted_env_record(job.model_dump(mode="json")))
    return folder


def _inputs(job: JobRecord) -> dict:
    """Where the runtime finds each input: an uploaded bundle, or a dataset Kaggle mounts directly."""
    inputs = {
        alias: dict(ref=job.upload_refs["input:" + alias], digest=bundle["digest"])
        for alias, bundle in Provider.bundled_inputs(job)
    }
    for alias in job.spec.dataset_inputs():
        inputs.setdefault(alias, dict(dataset=job.upload_refs["input:" + alias]))
    return inputs


def _metadata(job: JobRecord, ref: str, code_file: str) -> dict:
    datasets = [job.upload_refs.get("dataset:" + name, name) for name in job.spec.datasets]
    datasets += [value for key, value in job.upload_refs.items() if not key.startswith("dataset:")]
    metadata = dict(
        id=ref,
        title=ref.split("/")[1].replace("-", " "),
        code_file=code_file,
        language="python",
        kernel_type=job.snapshot["kind"],
        is_private=True,
        enable_gpu=job.spec.gpu,
        enable_tpu=False,
        enable_internet=job.spec.internet,
        dataset_sources=list(dict.fromkeys(datasets)),
        competition_sources=[],
        kernel_sources=[],
        model_sources=[],
    )
    if job.spec.accelerator:
        metadata["machine_shape"] = job.spec.accelerator
    return metadata


def render_log(raw):
    """Kaggle persisted logs may be a JSON array of stream events."""
    try:
        events = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if isinstance(events, list) and all(isinstance(event, dict) and "data" in event for event in events):
        return "".join(str(event["data"]) for event in events)
    return raw

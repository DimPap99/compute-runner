"""Construct a private kernel from immutable local snapshots."""

import base64
from pathlib import Path

import nbformat

from . import runtime
from .models import JobRecord
from .store import atomic_json


def prepare_kernel(job: JobRecord, state_dir: Path) -> Path:
    attempt = job.attempts[-1]
    folder = state_dir / "jobs" / job.id / f"attempt-{attempt.number}"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshot = job.snapshot
    source = snapshot["source"]
    payload = state_dir / "bundles" / source["digest"] / "files"
    config = dict(
        job_id=job.id,
        source_digest=source["digest"],
        source_files=source["files"],
        source_ref=job.upload_refs.get("source"),
        inline=None,
        inputs={
            alias: dict(ref=job.upload_refs["input:" + alias], digest=bundle["digest"])
            for alias, bundle in snapshot["inputs"].items()
        },
        module=snapshot["module"],
        entrypoint=snapshot["entrypoint"],
        args=job.spec.args,
        env=job.spec.env,
        requirements=job.spec.requirements,
    )
    if snapshot["single_file"]:
        config["inline"] = {
            name: base64.b64encode((payload / name).read_bytes()).decode() for name in source["files"]
        }
    bootstrap = Path(runtime.__file__).read_text() + "\n_KGR_CONFIG = " + repr(config) + "\n"
    if snapshot["kind"] == "notebook":
        notebook = nbformat.read(payload / snapshot["entrypoint"], as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap + "bootstrap(_KGR_CONFIG)\n"))
        code_file = "workload.ipynb"
        nbformat.write(notebook, folder / code_file)
    else:
        code_file = "workload.py"
        (folder / code_file).write_text(bootstrap + "run_script(_KGR_CONFIG)\n")
    metadata = dict(
        id=attempt.ref,
        title=attempt.ref.split("/")[1].replace("-", " "),
        code_file=code_file,
        language="python",
        kernel_type=snapshot["kind"],
        is_private=True,
        enable_gpu=job.spec.gpu,
        enable_tpu=False,
        enable_internet=job.spec.internet,
        dataset_sources=list(
            dict.fromkeys(
                [
                    *[job.upload_refs.get("dataset:" + ref, ref) for ref in job.spec.datasets],
                    *[value for key, value in job.upload_refs.items() if not key.startswith("dataset:")],
                ]
            )
        ),
        competition_sources=[],
        kernel_sources=[],
        model_sources=[],
    )
    if job.spec.accelerator:
        metadata["machine_shape"] = job.spec.accelerator
    atomic_json(folder / "kernel-metadata.json", metadata)
    atomic_json(folder / "provenance.json", job.model_dump(mode="json"))
    return folder

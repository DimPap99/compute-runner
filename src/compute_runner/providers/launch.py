"""Launch packages every provider shares: the runtime's configuration and the code that starts it."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import nbformat

from .. import runtime
from ..models import JobRecord


def inline_project(snapshot: dict, state_dir: Path) -> dict | None:
    """Embed a small script project within Kaggle's source-size limit.

    Keep every snapshotted file and its hash. Large projects retain dataset
    transport; code-only projects need no asynchronous dataset creation.
    """
    files = snapshot["source"].get("files", {})
    if (snapshot.get("single_file") or snapshot.get("kind") != "script" or not files
            or sum(record["size"] for record in files.values()) > 2_000_000):
        return None
    payload = state_dir / "bundles" / snapshot["source"]["digest"] / "files"
    embedded = {name: base64.b64encode(gzip.compress((payload / name).read_bytes(), mtime=0)).decode()
                for name in files}
    if len(json.dumps({"inline_gzip": embedded, "source_files": files})) > 700_000:
        return None
    return embedded


def launch_config(job: JobRecord, state_dir: Path, inputs: dict, **location) -> dict:
    """The runtime configuration for one attempt.

    inputs maps each alias to where the runtime finds it (see runtime.bootstrap); location says
    where the source bundle is, such as source_ref on Kaggle. A single-file source is embedded.
    """
    snapshot = job.snapshot
    source = snapshot["source"]
    config = dict(
        job_id=job.id,
        source_digest=source["digest"],
        inline=None,
        inputs=inputs,
        module=snapshot["module"],
        entrypoint=snapshot["entrypoint"],
        args=job.spec.command_args(),
        params=job.spec.params,
        env=job.spec.env,
        requirements=job.spec.requirements,
        **location,
    )
    if snapshot["single_file"]:
        # Directory bundles carry their own manifest; only embedded files need their checksums here.
        payload = state_dir / "bundles" / source["digest"] / "files"
        config["source_files"] = source["files"]
        if snapshot["kind"] == "notebook":
            # Notebook cells are also in the launcher. Compress the preserved
            # original so that embedding it does not double the kernel source
            # past Kaggle's 1 MB limit. The runtime still verifies its bytes.
            config["inline_gzip"] = {
                name: base64.b64encode(gzip.compress((payload / name).read_bytes(), mtime=0)).decode()
                for name in source["files"]
            }
        else:
            config["inline"] = {
                name: base64.b64encode((payload / name).read_bytes()).decode() for name in source["files"]
            }
    elif (embedded := inline_project(snapshot, state_dir)) is not None:
        config["source_files"] = source["files"]
        config["inline_gzip"] = embedded
    return config


def write_launcher(job: JobRecord, config: dict, folder: Path, state_dir: Path) -> str:
    """Write the runtime and its configuration into folder; return the file to run.

    A notebook gets the runtime as its first cell; anything else becomes workload.py.
    """
    bootstrap = Path(runtime.__file__).read_text() + "\n_KGR_CONFIG = " + repr(config) + "\n"
    if job.snapshot["kind"] == "notebook":
        payload = state_dir / "bundles" / job.snapshot["source"]["digest"] / "files"
        notebook = nbformat.read(payload / job.snapshot["entrypoint"], as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap + "bootstrap(_KGR_CONFIG)\n"))
        nbformat.write(notebook, folder / "workload.ipynb")
        return "workload.ipynb"
    (folder / "workload.py").write_text(bootstrap + "run_script(_KGR_CONFIG)\n")
    return "workload.py"

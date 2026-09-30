import base64
import hashlib
import random
import subprocess
import sys
from types import SimpleNamespace as Obj

import nbformat

from compute_runner.providers.launch import launch_config, write_launcher
from compute_runner.runtime import json_digest


def test_small_project_embeds_and_runs_with_verified_files(tmp_path):
    contents = {"task.py": b"from helper import VALUE\nassert VALUE == 42\n", "helper.py": b"VALUE = 42\n"}
    records = {
        name: dict(size=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in contents.items()
    }
    digest = json_digest(records)
    payload = tmp_path / "bundles" / digest / "files"
    payload.mkdir(parents=True)
    for name, data in contents.items():
        (payload / name).write_bytes(data)
    job = Obj(
        id="project",
        snapshot=dict(
            kind="script",
            single_file=False,
            module="task",
            entrypoint=None,
            source=dict(digest=digest, files=records),
        ),
        spec=Obj(command_args=lambda: [], params={}, env={}, requirements=None),
    )
    work, stage = tmp_path / "work", tmp_path / "stage"
    stage.mkdir()
    config = launch_config(job, tmp_path, {}, working_root=str(work), source_ref=None)
    assert config["inline_gzip"] and config["source_ref"] is None
    file = write_launcher(job, config, stage, tmp_path)
    assert (stage / file).stat().st_size < 1_000_000
    subprocess.run([sys.executable, str(stage / file)], check=True, capture_output=True)
    for name, data in contents.items():
        assert (work / "project" / name).read_bytes() == data


def test_large_notebook_fits_kaggle_limit_and_restores_exact_source(tmp_path):
    # An embedded code archive resembles the large notebooks used by real jobs.
    payload = base64.b64encode(random.Random(0).randbytes(350_000)).decode()
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f"payload = {payload!r}\n")])
    original = nbformat.writes(notebook).encode()
    digest = hashlib.sha256(original).hexdigest()
    files = tmp_path / "bundles" / digest / "files"
    files.mkdir(parents=True)
    (files / "notebook.ipynb").write_bytes(original)
    job = Obj(
        id="large-notebook",
        snapshot=dict(
            kind="notebook",
            single_file=True,
            module=None,
            entrypoint="notebook.ipynb",
            source=dict(
                digest=digest,
                files={
                    "notebook.ipynb": dict(sha256=digest, size=len(original)),
                },
            ),
        ),
        spec=Obj(command_args=lambda: [], params={}, env={}, requirements=None),
    )
    work = tmp_path / "work"
    config = launch_config(job, tmp_path, {}, working_root=str(work))
    stage = tmp_path / "stage"
    stage.mkdir()
    code_file = write_launcher(job, config, stage, tmp_path)
    assert (stage / code_file).stat().st_size < 1_000_000
    launched = nbformat.read(stage / code_file, as_version=4)
    subprocess.run(
        [sys.executable, "-"], input=launched.cells[0].source, text=True, check=True, capture_output=True
    )
    assert (work / "project" / "notebook.ipynb").read_bytes() == original
    assert launched.cells[1].source == notebook.cells[0].source

import shutil
import subprocess
import sys
import zipfile

import nbformat
import pytest

from compute_runner import JobSpec
from compute_runner.bundle import describe, inventory, snapshot, snapshot_bundle
from compute_runner.runtime import _find_bundle, unpack_bundle

from conftest import staged_launcher


def test_snapshot_is_immutable_and_reused(setup):
    client, _, spec = setup
    first = client.submit(spec)
    second = client.submit(spec)
    assert first.snapshot["source"]["digest"] == second.snapshot["source"]["digest"]
    spec.source.write_text("raise RuntimeError('changed')")
    saved = client.config.state_dir / "bundles" / first.snapshot["source"]["digest"] / "files" / "hello.py"
    assert saved.read_text() == "print('hello')\n"
    assert client.submit(spec).snapshot["source"]["digest"] != first.snapshot["source"]["digest"]


def test_credential_cache_and_ignore_exclusions(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for name in [
        "main.py",
        ".env",
        ".env.production",
        ".netrc",
        ".npmrc",
        "KAGGLE.JSON",
        "key.PEM",
        "state.tfstate.backup",
        "skip.txt",
        "keep.txt",
    ]:
        (project / name).write_text("example")
    (project / ".gitignore").write_text("skip.txt\n")
    (project / ".kgrignore").write_text("keep.txt\n")
    (project / ".venv").mkdir()
    (project / ".venv" / "secret").write_text("example")
    _, files = inventory(project, [])
    assert {p.name for p in files} == {"main.py", ".gitignore", ".kgrignore"}


def test_snapshot_rejects_detected_credential_without_echoing_it(tmp_path):
    source = tmp_path / "main.py"
    credential = "KAGGLE_KEY=" + "a" * 32
    source.write_text("print('before')\n" + credential + "\n")
    with pytest.raises(ValueError, match="Detected Kaggle API key") as error:
        snapshot_bundle(*inventory(source, []), tmp_path / "bundles")
    assert credential not in str(error.value)


def test_job_env_rejects_secret_names_and_values(tmp_path):
    source = tmp_path / "main.py"
    source.write_text("pass")
    with pytest.raises(ValueError, match="looks secret"):
        JobSpec(source=source, env={"API_TOKEN": "anything"})
    with pytest.raises(ValueError, match="detected Kaggle token"):
        JobSpec(source=source, env={"VALUE": "KGAT_" + "a" * 24})


def test_rejects_symlink_and_missing_entrypoint(tmp_path):
    (tmp_path / "main.py").write_text("pass")
    (tmp_path / "link.py").symlink_to(tmp_path / "main.py")
    with pytest.raises(ValueError, match="Symlink"):
        inventory(tmp_path, [])
    with pytest.raises(ValueError, match="absent"):
        describe(JobSpec(source=tmp_path, entrypoint="missing.py", exclude=["link.py"]))


def test_supporting_notebooks_need_not_be_python(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("pass")
    kernelspec = {"language": "R", "name": "ir", "display_name": "R"}
    nbformat.write(nbformat.v4.new_notebook(metadata={"kernelspec": kernelspec}), project / "analysis.ipynb")
    (project / "broken.ipynb").write_text("not json")
    saved = snapshot(JobSpec(source=project, entrypoint="main.py"), tmp_path / "state")
    assert {"analysis.ipynb", "broken.ipynb"} <= set(saved["source"]["files"])
    with pytest.raises(ValueError, match="Only Python"):
        describe(JobSpec(source=project, entrypoint="analysis.ipynb"))


def test_input_folders_skip_gitignore_but_honor_kgrignore(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / ".gitignore").write_text("train.csv\n")
    (data / ".kgrignore").write_text("scratch.tmp\n")
    for name in ("train.csv", "scratch.tmp"):
        (data / name).write_text("example")
    script = tmp_path / "main.py"
    script.write_text("pass")
    files = describe(JobSpec(source=script, inputs={"data": data}))["inputs"]["data"]["files"]
    assert "train.csv" in files and "scratch.tmp" not in files


def test_notebook_outputs_cleared_without_editing_original(tmp_path):
    source = tmp_path / "test.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print(42)",
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="private old output")],
                execution_count=1,
            )
        ]
    )
    nbformat.write(notebook, source)
    bundle = snapshot_bundle(*inventory(source, []), tmp_path / "bundles", notebooks=True)
    saved = nbformat.read(tmp_path / "bundles" / bundle["digest"] / "files" / source.name, as_version=4)
    assert saved.cells[0].outputs == []
    assert nbformat.read(source, as_version=4).cells[0].outputs


def test_script_launcher_executes_locally(setup, tmp_path):
    client, backend, spec = setup
    spec.source.write_text(
        "import os\nfrom pathlib import Path\n"
        'Path(os.environ["KGR_OUTPUT_DIR"], "result.txt").write_text("ok")\n'
    )
    job = client.submit(spec)
    script = staged_launcher(backend, job)
    script.write_text(script.read_text().replace("/kaggle/working", str(tmp_path / "working")))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "working/outputs/result.txt").read_text() == "ok"


def test_folder_module_launcher_and_data_input(setup, tmp_path):
    client, backend, _ = setup
    project = tmp_path / "project"
    project.mkdir()
    (project / "pkg").mkdir()
    (project / "pkg/__init__.py").write_text("")
    (project / "pkg/helper.py").write_text("VALUE = 42")
    (project / "settings.txt").write_text("setting")
    (project / "pkg/main.py").write_text("""import os
from pathlib import Path
from pkg.helper import VALUE
assert Path('settings.txt').read_text() == 'setting'
assert Path(os.environ['KGR_INPUT_DATA'], 'data.txt').read_text() == 'input'
Path(os.environ['KGR_OUTPUT_DIR'], 'result.txt').write_text(str(VALUE))
""")
    data = tmp_path / "data"
    data.mkdir()
    (data / "data.txt").write_text("input")
    job = client.submit(JobSpec(source=project, module="pkg.main", inputs={"data": data}))
    job.upload_refs["source"] = backend.ensure_bundle(job.snapshot["source"])
    job.upload_refs["input:data"] = backend.ensure_bundle(job.snapshot["inputs"]["data"])
    for key, bundle in [("source", job.snapshot["source"]), ("input:data", job.snapshot["inputs"]["data"])]:
        mount = tmp_path / "input" / job.upload_refs[key].split("/")[1]
        shutil.copytree(client.config.state_dir / "bundles" / bundle["digest"] / "files", mount)
    script = staged_launcher(backend, job)
    script.write_text(
        script.read_text()
        .replace("/kaggle/working", str(tmp_path / "working"))
        .replace("/kaggle/input", str(tmp_path / "input"))
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "working/outputs/result.txt").read_text() == "42"


def test_archive_and_expanded_lookup(tmp_path):
    source = tmp_path / "hello.py"
    source.write_text("print(42)")
    bundle = snapshot_bundle(*inventory(source, []), tmp_path / "bundles")
    mount = tmp_path / "input/test-bundle"
    mount.mkdir(parents=True)
    archive = tmp_path / "bundles" / bundle["digest"] / "payload.zip"
    shutil.copyfile(archive, mount / "payload.zip")
    found = _find_bundle("tester/test-bundle/1", bundle["digest"], input_root=tmp_path / "input")
    assert found[0] == "archive"
    target = tmp_path / "extracted"
    target.mkdir()
    unpack_bundle(archive, bundle["digest"], target)
    assert (target / "hello.py").read_text() == "print(42)"
    shutil.copytree(tmp_path / "bundles" / bundle["digest"] / "files", mount, dirs_exist_ok=True)
    assert (
        _find_bundle("tester/test-bundle", bundle["digest"], input_root=tmp_path / "input")[0] == "directory"
    )
    (mount / "hello.py").write_text("corrupted")
    with pytest.raises(ValueError, match="invalid|checksum"):
        _find_bundle("tester/test-bundle", bundle["digest"], input_root=tmp_path / "input")


def test_rejects_tampered_archive(tmp_path):
    source = tmp_path / "hello.py"
    source.write_text("original")
    bundle = snapshot_bundle(*inventory(source, []), tmp_path / "bundles")
    malicious = tmp_path / "bad.zip"
    saved = tmp_path / "bundles" / bundle["digest"] / "files"
    with zipfile.ZipFile(malicious, "w") as z:
        z.write(saved / "kgr-manifest.json", "kgr-manifest.json")
        z.writestr("hello.py", "modified")
        z.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unexpected"):
        unpack_bundle(malicious, bundle["digest"], tmp_path / "out")


def test_offline_requirements_validation():
    with pytest.raises(ValueError, match="internet"):
        JobSpec(source=".", module="main", requirements="requirements.txt")


def test_state_directory_cannot_be_bundled(setup):
    client, _, spec = setup
    with pytest.raises(ValueError, match="state directory"):
        client.submit(JobSpec(source=spec.source.parent, entrypoint=spec.source.name))

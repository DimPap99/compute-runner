"""Run folders are fixed at submission and filled by the worker; agents only read them."""

import hashlib
import json
import re
import sqlite3
import subprocess
from contextlib import contextmanager
import sys
from pathlib import Path

import pytest

from compute_runner import Client, JobSpec
from compute_runner.client import resume_required
from compute_runner.workloads import load_specs, workload_specs
from conftest import due, staged_launcher


def finish(client, backend, state="COMPLETE"):
    for job in client.list():
        if job.remote_ref in backend.remote:
            backend.remote[job.remote_ref] = dict(state=state, error=None)
            due(client, job.id)
    client.worker().tick()


def test_run_folders_are_numbered_at_submission_beside_the_code(setup, tmp_path):
    client, backend, spec = setup
    first = client.submit(spec.model_copy(update={"name": "hello world", "params": {"lr": 0.01, "seed": 1}}))
    batch = client.submit_many([spec.model_copy(update={"name": "hello world"})] * 2)
    runs = [first, *batch]
    experiment = tmp_path / "results/hello-world"
    assert [job.result_dir.parent for job in runs] == [experiment] * 3
    assert [job.run for job in runs] == [1, 2, 3]
    assert re.fullmatch(r"001_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", first.result_dir.name)
    assert [job.result_dir.name[:4] for job in batch] == ["002_", "003_"]
    assert client.get(first.id).result_dir == first.result_dir
    # Submission reserves each run folder empty; the worker fills it.
    runs_on_disk = sorted(path for path in experiment.iterdir() if not path.name.startswith("."))
    assert runs_on_disk == [job.result_dir for job in runs] and not any(any(p.iterdir()) for p in runs_on_disk)

    client.worker().tick()
    record = json.loads((first.result_dir / "job.json").read_text())
    assert record["job_id"] == first.id and record["params"] == {"lr": 0.01, "seed": 1}
    assert record["run"] == 1 and record["state"] == "remote_queued" and record["attempts"][0]["url"]
    assert record["command"]["args"] == ["--lr", "0.01", "--seed", "1"]
    index = (experiment / "runs.md").read_text()
    assert "lr=0.01, seed=1" in index and first.id[:12] in index
    assert index.count("| remote_queued |") == 3

    finish(client, backend)
    assert json.loads((first.result_dir / "job.json").read_text())["state"] == "succeeded"
    assert (first.result_dir / "outputs/result.json").read_text() == '{"ok": true}'
    assert (first.result_dir / "run.log").read_text() == "remote completed\n"
    rows = (experiment / "runs.md").read_text()
    assert rows.count("| succeeded |") == 3 and rows.count("| complete |") == 3


def test_records_are_written_once_per_change_across_worker_restarts(setup):
    client, _, spec = setup
    job = client.submit(spec)
    client.worker().tick()
    record = job.result_dir / "job.json"
    record.write_text("stale")
    # A new worker resumes from the saved cursor; unchanged jobs are not rewritten.
    client.worker().tick()
    assert record.read_text() == "stale" and int(client.store.meta("views_cursor")) > 0
    client.cancel(job.id)
    client.worker().tick()
    assert json.loads(record.read_text())["job_id"] == job.id


def test_numbers_continue_after_folders_already_on_disk(setup, tmp_path):
    client, _, spec = setup
    (tmp_path / "results/workload/007_2020-01-01_00-00-00").mkdir(parents=True)
    (tmp_path / "results/workload/notes").mkdir()
    assert client.submit(spec).run == 8


def test_queues_of_different_state_directories_never_share_a_run_number(setup, tmp_path):
    client, backend, spec = setup
    other = Client(
        config=client.config.model_copy(update={"state_dir": tmp_path / "other-state"}),
        providers={"kaggle:tester": backend},
    )
    numbers = [client.submit(spec).run, other.submit(spec).run, client.submit(spec).run]
    assert numbers == [1, 2, 3]


def test_a_batch_that_fails_to_commit_leaves_no_run_folders(setup, tmp_path, monkeypatch):
    client, _, spec = setup

    def fail(*args):
        raise RuntimeError("disk full")

    monkeypatch.setattr(client.store, "_event", fail)
    with pytest.raises(RuntimeError):
        client.submit_many([spec, spec])
    assert [p.name for p in (tmp_path / "results/workload").iterdir()] == [".lock"]
    monkeypatch.undo()

    # A failure of the commit itself, after every statement ran, also removes them.
    connection = client.store.connection

    @contextmanager
    def failing_commit():
        with connection() as db:
            yield db
            raise sqlite3.OperationalError("disk I/O error")

    # Only add_batch opens a connection once the request-key lookup is out of the way.
    monkeypatch.setattr(client.store, "request", lambda *args: None)
    monkeypatch.setattr(client.store, "connection", failing_commit)
    with pytest.raises(sqlite3.OperationalError):
        client.submit(spec)
    assert [p.name for p in (tmp_path / "results/workload").iterdir()] == [".lock"]
    monkeypatch.undo()
    assert client.submit(spec).run == 1


def test_results_folder_is_the_workloads_then_the_configured_one(setup, tmp_path):
    client, _, spec = setup
    client.config.results_dir = tmp_path / "configured"
    assert client.submit(spec).result_dir.parent == tmp_path / "configured/workload"
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp/sweep.yaml").write_text(
        "name: sweep\nsource: ../hello.py\nresults_dir: runs\ninputs: {prior: 'job:abc/model'}\n"
    )
    [loaded] = load_specs(tmp_path / "exp/sweep.yaml")
    assert loaded.results_dir == tmp_path / "exp/runs" and str(loaded.inputs["prior"]) == "job:abc/model"
    loaded.inputs = {}
    assert client.submit(loaded).result_dir.parent == tmp_path / "exp/runs/sweep"


def test_results_inside_a_project_are_not_uploaded_with_it(setup, tmp_path):
    client, _, _ = setup
    project = tmp_path / "project"
    (project / "results/train/001_2020-01-01_00-00-00").mkdir(parents=True)
    (project / "results/train/001_2020-01-01_00-00-00/model.bin").write_text("weights")
    (project / "train.py").write_text("print(1)\n")
    preview = client.preview(JobSpec(source=project, entrypoint="train.py", name="train"))
    assert preview["files"] == ["train.py"] and preview["experiment_dir"] == str(project / "results/train")
    job = client.submit(JobSpec(source=project, entrypoint="train.py", name="train"))
    assert list(job.snapshot["source"]["files"]) == ["train.py"] and job.run == 2
    with pytest.raises(ValueError, match="source folder itself"):
        client.submit(JobSpec(source=project, entrypoint="train.py", results_dir=project))
    # Named like the source folder, the experiment folder would be the source itself.
    with pytest.raises(ValueError, match="source folder itself"):
        client.submit(JobSpec(source=project, entrypoint="train.py", name="project", results_dir=tmp_path))


def test_params_reach_the_command_line_and_environment(setup, tmp_path):
    client, backend, spec = setup
    (tmp_path / "hello.py").write_text("import os, sys\nprint(sys.argv[1:])\nprint(os.environ['KGR_PARAMS_JSON'])\n")
    params = {"lr": 0.01, "amp": True, "debug": False, "tag": "a b"}
    job = client.submit(spec.model_copy(update={"args": ["--epochs", "2"], "params": params}))
    script = staged_launcher(backend, job)
    script.write_text(script.read_text().replace("/kaggle/working", str(tmp_path / "working")))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "['--epochs', '2', '--lr', '0.01', '--amp', '--tag', 'a b']" in result.stdout
    assert json.dumps(params) in result.stdout


def test_param_options_are_read_like_yaml_and_screened(tmp_path):
    (tmp_path / "t.py").write_text("")

    def specs(*params):
        options = dict(entrypoint=None, module=None, gpu=None, internet=None, accelerator=None)
        return workload_specs(tmp_path / "t.py", param=list(params), **options)

    [spec] = specs("lr=0.01", "seed=3", "amp=true", "arch=resnet", "empty=", "day=2020-01-01")
    assert spec.params == {"lr": 0.01, "seed": 3, "amp": True, "arch": "resnet", "empty": "", "day": "2020-01-01"}
    with pytest.raises(ValueError, match="NAME=VALUE"):
        specs("lr")
    with pytest.raises(ValueError, match="parameter api_key looks secret"):
        specs("api_key=abc")
    with pytest.raises(ValueError, match="parameter hf-token looks secret"):
        specs("hf-token=abc")
    with pytest.raises(ValueError, match="Invalid parameter name"):
        specs("--lr=1")


def test_earlier_request_keys_still_replay(setup):
    client, _, spec = setup
    intent = client._intent(client._normalize(spec))
    assert "params" not in intent and "results_dir" not in intent


def test_inputs_can_name_another_jobs_outputs(setup):
    client, backend, spec = setup
    backend.output_files = {"outputs/features/a.npy": "A", "outputs/metrics.json": "{}"}
    prep = client.submit(spec.model_copy(update={"name": "prep"}))
    uses = spec.model_copy(update={"name": "train", "inputs": {"features": Path(f"job:{prep.id[:8]}/features")}})
    with pytest.raises(ValueError, match="not downloaded yet"):
        client.submit(uses)
    client.worker().tick()
    finish(client, backend)
    train = client.submit(uses)
    assert str(train.spec.inputs["features"]) == f"job:{prep.id}/features"
    assert list(train.snapshot["inputs"]["features"]["files"]) == ["a.npy"]
    with pytest.raises(ValueError, match="has no output missing"):
        client.submit(spec.model_copy(update={"inputs": {"x": Path(f"job:{prep.id}/missing")}}))


def checkpoints(backend, digest=None):
    backend.output_files = {
        "outputs/checkpoints/checkpoint-step-000000000005.pt": "older",
        "outputs/checkpoints/checkpoint-step-000000000010.pt": "weights",
        "outputs/checkpoints/latest.json": json.dumps(
            dict(
                schema_version=1,
                file="checkpoint-step-000000000010.pt",
                sha256=digest or hashlib.sha256(b"weights").hexdigest(),
                global_step=10,
            )
        ),
    }


def test_continue_resumes_from_the_verified_checkpoint_as_the_next_run(two_accounts):
    client, home, other, spec = two_accounts
    checkpoints(home)
    agent = client.agent()
    first = agent.submit([spec.model_copy(update={"name": "train", "args": ["--resume", "auto"]})], request_key="t1")
    first = first["jobs"][0]
    with pytest.raises(ValueError, match="after it stops"):
        agent.continue_run(first["id"], request_key="t2")
    client.worker().tick()
    finish(client, home, state="CANCEL_ACKNOWLEDGED")

    resumed = agent.continue_run(first["id"][:8], request_key="t2", account="kaggle:other")
    again = agent.continue_run(first["id"], request_key="t2", account="kaggle:other")
    assert again["replayed"] and again["jobs"][0]["id"] == resumed["jobs"][0]["id"]
    job = client.get(resumed["jobs"][0]["id"])
    assert job.parent_id == first["id"] and job.account == "kaggle:other"
    assert job.result_dir.parent == Path(first["run_dir"]).parent and job.run == 2
    assert job.spec.args == ["--resume", "required"]
    assert str(job.spec.inputs["resume"]) == f"job:{first['id']}/checkpoints"
    # Only the manifest and the checkpoint it names are uploaded.
    resume = job.snapshot["inputs"]["resume"]["files"]
    assert sorted(resume) == ["checkpoint-step-000000000010.pt", "latest.json"]
    client.worker().tick()
    assert len(other.pushes) == 1
    assert "| 001 |" in (job.result_dir.parent / "runs.md").read_text()


def test_continue_replaces_an_input_named_resume_in_any_casing(setup, tmp_path):
    client, backend, spec = setup
    checkpoints(backend)
    (tmp_path / "old").mkdir()
    (tmp_path / "old/weights.txt").write_text("old")
    job = client.submit(spec.model_copy(update={"inputs": {"RESUME": tmp_path / "old"}}))
    client.worker().tick()
    finish(client, backend, state="ERROR")
    resumed = client.continue_run(job.id)
    assert list(resumed.spec.inputs) == ["resume"] and list(resumed.snapshot["inputs"]) == ["resume"]


def test_continue_refuses_a_checkpoint_that_does_not_verify(setup):
    client, backend, spec = setup
    checkpoints(backend, digest="0" * 64)
    job = client.submit(spec)
    client.worker().tick()
    finish(client, backend, state="ERROR")
    with pytest.raises(ValueError, match="does not match latest.json"):
        client.continue_run(job.id)
    backend.output_files = {}
    other = client.submit(spec)
    client.worker().tick()
    finish(client, backend, state="ERROR")
    with pytest.raises(ValueError, match="No readable checkpoint manifest"):
        client.continue_run(other.id)


@pytest.mark.parametrize(
    "args, params, expected_args, expected_params",
    [
        (["--resume", "auto", "--x"], {}, ["--resume", "required", "--x"], {}),
        (["--resume=never"], {}, ["--resume=required"], {}),
        ([], {}, ["--resume", "required"], {}),
        (["--x", "--resume"], {}, ["--x", "--resume", "required"], {}),
        ([], {"resume": "auto"}, [], {"resume": "required"}),
    ],
)
def test_continuations_always_require_the_checkpoint(tmp_path, args, params, expected_args, expected_params):
    spec = resume_required(JobSpec(source=tmp_path, entrypoint="t.py", args=args, params=params))
    assert spec.args == expected_args and spec.params == expected_params


def test_jobs_from_before_run_folders_keep_their_layout(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    legacy = client.config.state_dir / "results" / job.id
    client.store.update(job.id, run=None, result_dir=legacy)
    client.worker().tick()
    finish(client, backend)
    assert (legacy / "outputs/outputs/result.json").is_file() and (legacy / "provenance.json").is_file()
    listed = client.agent().outputs(job.id)
    assert listed["root"] == str(legacy / "outputs") and listed["files"][0]["path"] == "outputs/result.json"
    assert not (legacy / "job.json").exists()

"""End-to-end scenarios for an LLM driving the queue through `kgr agent` and AgentClient."""

import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace as Obj

import nbformat
import pytest
import requests
import urllib3
from typer.testing import CliRunner

from kaggle_runner import JobSpec
from kaggle_runner.backend import READ_TIMEOUT, KaggleBackend, RemoteError, download_outputs
from kaggle_runner.cli import app
from kaggle_runner.launcher import prepare_kernel
from kaggle_runner.models import Attempt
from kaggle_runner.worker import source_copies
from conftest import due


def agent_cli(code=0):
    runner = CliRunner()

    def command(*args):
        result = runner.invoke(app, ["agent", *map(str, args)])
        assert result.exit_code == code, (result.output, result.exception)
        return json.loads(result.stdout)

    return command


def finish(client, backend, state="COMPLETE", error=None):
    for job in client.list():
        if job.remote_ref in backend.remote:
            backend.remote[job.remote_ref] = dict(state=state, error=error)
            due(client, job.id)
    client.worker().tick()


@pytest.fixture
def workload(tmp_path):
    """A CPU script, a GPU project and a notebook, as an LLM would write them."""
    (tmp_path / "prep.py").write_text("print('prep')\n")
    project = tmp_path / "project"
    (project / "demo").mkdir(parents=True)
    (project / "demo/__init__.py").write_text("")
    (project / "demo/train.py").write_text("print('train')\n")
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell("print('nb')"))
    nbformat.write(notebook, tmp_path / "explore.ipynb")
    config = tmp_path / "batch.yaml"
    config.write_text(
        "jobs:\n"
        "  - {name: prep, source: prep.py, timeout_seconds: 600}\n"
        "  - {name: train, source: project, module: demo.train, accelerator: NvidiaTeslaT4,"
        " internet: true, args: ['--epochs', '2']}\n"
        "  - {name: explore, source: explore.ipynb}\n"
    )
    return config


def test_llm_submit_monitor_wait_and_read_results(setup, workload, monkeypatch):
    client, backend, _ = setup
    monkeypatch.setattr("kaggle_runner.cli.Client", lambda **_: client)
    monkeypatch.chdir(workload.parent)
    command = agent_cli()
    preview = command("submit", workload, "--request-key", "exp-1", "--dry-run")
    assert preview["total"] == 3 and preview["gpu_jobs"] == 1 and client.list() == []

    batch = command("submit", workload, "--request-key", "exp-1")
    batch_id = batch["batch_id"]
    assert [j["name"] for j in batch["jobs"]] == ["prep", "train", "explore"]
    assert [j["resource"] for j in batch["jobs"]] == ["cpu", "NvidiaTeslaT4", "cpu"]
    # An interrupted call is replayed, not duplicated.
    assert command("submit", workload, "--request-key", "exp-1")["replayed"]

    client.worker().tick()
    kinds = sorted(push["kernel_type"] for push in backend.pushes)
    assert kinds == ["notebook", "script", "script"]
    gpu = next(push for push in backend.pushes if push["enable_gpu"])
    assert gpu["machine_shape"] == "NvidiaTeslaT4" and gpu["enable_internet"]

    changes = command("changes", "--batch", batch_id, "--after", 0)
    assert {j["state"] for j in changes["jobs"]} == {"remote_queued"}
    cursor = changes["cursor"]
    assert command("changes", "--batch", batch_id, "--after", cursor)["jobs"] == []

    # Running: logs are a live snapshot and are re-read on every call.
    train_id = batch["jobs"][1]["id"]
    assert command("logs", train_id)["live"]
    backend.live_text = "epoch 1\nepoch 2\n"
    assert command("logs", train_id)["text"].endswith("epoch 2\n") and backend.live_calls == 2

    # Nothing has finished, so a short wait times out without an error exit.
    waited = command("wait", "--batch", batch_id, "--timeout", 0)
    assert waited["timed_out"] and not waited["settled"]

    finish(client, backend)
    waited = command("wait", "--batch", batch_id, "--timeout", 5)
    assert waited["settled"] and waited["counts"] == {"succeeded": 3}
    assert all(j["outputs_ready"] for j in waited["jobs"])
    changed = command("changes", "--batch", batch_id, "--after", cursor)
    assert {j["state"] for j in changed["jobs"]} == {"succeeded"}

    outputs = command("outputs", train_id[:10])
    assert outputs["outputs_ready"] and outputs["files"] == [{"path": "outputs/result.json", "bytes": 12}]
    result = json.loads(
        (client.config.state_dir / "results" / train_id / "outputs/outputs/result.json").read_text()
    )
    assert result == {"ok": True} and outputs["root"].endswith(f"{train_id}/outputs")
    assert outputs["log_path"].endswith(f"{train_id}/run.log")

    # A finished job's persisted log replaces the live snapshot once, then serves from cache.
    finished_log = command("logs", train_id)
    assert finished_log["fetched"] and not finished_log["live"] and finished_log["text"] == "example log\n"
    assert not command("logs", train_id)["fetched"]


def test_failed_run_is_diagnosable_and_retry_is_explicit_and_idempotent(setup):
    client, backend, spec = setup
    agent = client.agent()
    job_id = agent.submit([spec], request_key="run-1")["jobs"][0]["id"]
    client.worker().tick()
    finish(client, backend, state="ERROR", error="Traceback: ZeroDivisionError")
    status = agent.wait([job_id], timeout=5)
    job = status["jobs"][0]
    assert status["settled"] and job["state"] == "failed" and "ZeroDivisionError" in job["error"]
    assert job["outputs_ready"]  # Logs and partial outputs of failed runs are still collected.
    client.worker().tick()
    assert len(backend.pushes) == 1  # Failures are never rerun automatically.

    retried = agent.retry(job_id[:8], request_key="run-1-retry")
    assert agent.retry(job_id, request_key="run-1-retry")["replayed"]
    new = retried["jobs"][0]
    assert new["parent_id"] == job_id and new["state"] == "queued"
    client.worker().tick()
    assert len(backend.pushes) == 2


def test_gpu_waits_for_quota_while_cpu_proceeds(setup):
    client, backend, spec = setup
    backend.gpu_seconds = 0
    agent = client.agent()
    batch = agent.submit([spec.model_copy(update={"gpu": True}), spec], request_key="mixed")
    client.worker().tick()
    status = agent.status(batch_id=batch["batch_id"])
    gpu, cpu = status["jobs"]
    assert gpu["state"] == "queued" and gpu["reason"] == "Waiting for available GPU quota"
    assert cpu["state"] == "remote_queued" and len(backend.pushes) == 1


def test_wait_returns_on_download_failure_instead_of_hanging(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    client.worker().tick()
    backend.download_error = RemoteError("output listing unavailable")
    finish(client, backend)
    assert client.get(job.id).download_state == "error"
    assert client.wait(job.id, timeout=1).download_state == "error"
    status = client.agent().wait([job.id], timeout=1)
    assert status["settled"] and "unavailable" in status["jobs"][0]["download_error"]


def test_agent_wait_detects_a_stopped_worker(setup, monkeypatch):
    client, _, spec = setup
    job = client.submit(spec)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(RuntimeError, match="No worker is running"):
        client.agent().wait([job.id], timeout=600)
    assert 30 <= clock[0] < 40


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "either"),
        ({"job_ids": ["x"], "batch_id": "y"}, "either"),
        ({"batch_id": "b", "timeout": -1}, "timeout"),
    ],
)
def test_agent_wait_validates_selection(setup, kwargs, message):
    client, _, _ = setup
    with pytest.raises(ValueError, match=message):
        client.agent().wait(**kwargs)


def test_outputs_pagination_and_pending_downloads(setup):
    client, backend, spec = setup
    backend.output_files = {f"outputs/file{i:02}.txt": "x" * i for i in range(5)}
    agent = client.agent()
    job_id = agent.submit([spec], request_key="many")["jobs"][0]["id"]
    pending = agent.outputs(job_id)
    assert pending["total"] == 0 and pending["downloads"] == "pending" and not pending["outputs_ready"]
    client.worker().tick()
    finish(client, backend)
    first = agent.outputs(job_id, limit=2)
    assert [f["path"] for f in first["files"]] == ["outputs/file00.txt", "outputs/file01.txt"]
    assert first["total"] == 5 and first["next_offset"] == 2
    last = agent.outputs(job_id, limit=2, offset=4)
    assert last["files"] == [{"path": "outputs/file04.txt", "bytes": 4}] and last["next_offset"] is None


def test_runtime_outputs_exclude_source_copies_but_keep_new_files(setup, tmp_path):
    """Run a generated kernel locally, then download its working directory as Kaggle would."""
    client, backend, _ = setup
    project = tmp_path / "src"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg/__init__.py").write_text("")
    (project / "pkg/helper.py").write_text("VALUE = 7\n")
    (project / "pkg/main.py").write_text(
        "import os\nfrom pathlib import Path\nfrom pkg.helper import VALUE\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'metrics.json').write_text(str(VALUE))\n"
        "Path('checkpoint.bin').write_text('weights')\n"
    )
    job = client.submit(JobSpec(source=project, module="pkg.main"))
    job.attempts = [Attempt(number=1, ref="tester/kgr-scenario")]
    job.upload_refs["source"] = backend.ensure_bundle(job.snapshot["source"])
    mount = tmp_path / "input" / job.upload_refs["source"].split("/")[1]
    shutil.copytree(client.config.state_dir / "bundles" / job.snapshot["source"]["digest"] / "files", mount)
    script = prepare_kernel(job, client.config.state_dir) / "workload.py"
    working = tmp_path / "working"
    script.write_text(
        script.read_text()
        .replace("/kaggle/working", str(working))
        .replace("/kaggle/input", str(tmp_path / "input"))
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    remote = sorted(p.relative_to(working).as_posix() for p in working.rglob("*") if p.is_file())
    assert "project/pkg/helper.py" in remote and any("__pycache__" in name for name in remote)
    pages = [Obj(files=[Obj(file_name=n, url=n) for n in remote], log="done")]

    class Response:
        def __init__(self, name):
            self.data = (working / name).read_bytes()
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, **kwargs):
            yield self.data

    destination = tmp_path / "result"
    receipts = download_outputs(
        pages, destination, skip=source_copies(job), get=lambda url, **k: Response(url)
    )
    assert sorted(receipts) == ["outputs/metrics.json", "project/checkpoint.bin"]
    assert (destination / "outputs/outputs/metrics.json").read_text() == "7"


class StreamApi:
    def __init__(self, events, error=None):
        self.events, self.error, self.timeouts, self.closed = events, error, [], False

    def kernels_logs_stream(self, ref):
        try:
            for event in self.events:
                self.timeouts.append(READ_TIMEOUT.get())
                yield event
            if self.error:
                raise self.error
        finally:
            self.closed = True


def read_timeout():
    return requests.ConnectionError(urllib3.exceptions.ReadTimeoutError(None, None, "idle"))


def backend_with(api, tmp_path):
    backend = KaggleBackend("tester", tmp_path)
    backend._api = api
    return backend


def test_live_log_snapshot_ends_when_stream_goes_idle(tmp_path):
    api = StreamApi([{"data": "step 1\n"}, {"stream_name": "meta"}, {"data": "step 2\n"}], read_timeout())
    assert backend_with(api, tmp_path).live_log("tester/k", idle_seconds=3) == "step 1\nstep 2\n"
    assert api.timeouts == [3, 3, 3] and api.closed and READ_TIMEOUT.get() == 90


def test_live_log_before_any_output(tmp_path):
    assert backend_with(StreamApi([], read_timeout()), tmp_path).live_log("tester/k") == ""
    with pytest.raises(RemoteError):
        backend_with(StreamApi([], requests.ConnectionError("refused")), tmp_path).live_log("tester/k")
    assert READ_TIMEOUT.get() == 90


def test_live_log_is_bounded_for_chatty_runs(tmp_path, monkeypatch):
    clock = iter(range(100))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    api = StreamApi({"data": f"{i}\n"} for i in range(10**6))
    text = backend_with(api, tmp_path).live_log("tester/k", max_seconds=5)
    assert text == "0\n1\n2\n3\n4\n" and api.closed


def test_log_tail_ignores_carriage_return_progress_bars(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    client.worker().tick()
    finish(client, backend)
    backend.logs = lambda ref, follow=False: iter(
        ["".join(f"\r{i}%" for i in range(100)) + "\nloss=0.1\nError: boom\n"]
    )
    text = client.agent().logs(job.id, tail=3)["text"]
    assert text.startswith("\r0%") and text.endswith("loss=0.1\nError: boom\n")


def test_push_reference_case_differences_are_accepted(setup):
    client, backend, spec = setup
    original = backend.push

    def push(folder, **options):
        result = original(folder, **options)
        return result | {"ref": result["ref"].upper()}

    backend.push = push
    job = client.submit(spec)
    client.worker().tick()
    assert client.get(job.id).state == "remote_queued"
    assert client.get(job.id).attempts[-1].state == "accepted"


def test_structured_errors_for_common_llm_mistakes(setup, tmp_path, monkeypatch):
    client, _, _ = setup
    monkeypatch.setattr("kaggle_runner.cli.Client", lambda **_: client)
    monkeypatch.chdir(tmp_path)
    fail = agent_cli(code=1)
    assert fail("changes", "--batch", "nope")["error"] == "No batch nope"
    assert "Expected one matching job" in fail("outputs", "nope")["error"]
    broken = tmp_path / "broken.ipynb"
    broken.write_text(
        json.dumps({"cells": [{"cell_type": "code"}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    )
    assert (
        "Invalid notebook broken.ipynb" in fail("submit", broken, "--request-key", "nb", "--dry-run")["error"]
    )
    assert fail("wait", "--timeout", 1)["error"] == "Select either job IDs or a batch"
    assert client.list() == []


def discovery_backend(tmp_path, kernels, running, gpus=()):
    """Listing mirrors the live service: newest run first, every notebook reported as CPU."""
    backend = backend_with(
        Obj(kernels_list_with_response=lambda **_: Obj(kernels=kernels, next_page_token=None)), tmp_path
    )
    calls = {"status": [], "get_kernel": []}

    def status(ref):
        calls["status"].append(ref)
        return {"state": "RUNNING" if ref in running else "COMPLETE", "error": ""}

    def kernels_call(method, request, ref=None):
        assert method == "get_kernel"
        calls["get_kernel"].append(ref)
        return Obj(metadata=Obj(enable_gpu=ref in gpus))

    backend.status, backend._kernels = status, kernels_call
    return backend, calls


def test_account_discovery_stops_at_runs_too_old_to_be_active(tmp_path):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # the service returns naive UTC
    kernels = [
        Obj(ref=f"tester/recent-{i}", last_run_time=now - timedelta(hours=i), enable_gpu=False)
        for i in range(5)
    ]
    kernels += [
        Obj(ref=f"tester/old-{i}", last_run_time=now - timedelta(days=2 + i), enable_gpu=False)
        for i in range(65)
    ]
    backend, calls = discovery_backend(tmp_path, kernels, running={"tester/recent-0", "tester/recent-1"})
    assert backend.active_runs() == {"tester/recent-0": "cpu", "tester/recent-1": "cpu"}
    assert calls["status"] == [f"tester/recent-{i}" for i in range(5)]


def test_external_gpu_runs_are_detected_once_per_run(tmp_path):
    from datetime import datetime

    now = datetime.now()
    kernels = [Obj(ref=f"tester/k{i}", last_run_time=now, enable_gpu=False) for i in range(3)]
    running = {"tester/k0", "tester/k1"}
    backend, calls = discovery_backend(tmp_path, kernels, running, gpus={"tester/k1"})
    assert backend.active_runs() == {"tester/k0": "cpu", "tester/k1": "gpu"}
    assert backend.active_runs() == {"tester/k0": "cpu", "tester/k1": "gpu"}
    assert calls["get_kernel"] == ["tester/k0", "tester/k1"]  # cached across discovery cycles
    running.discard("tester/k1")
    assert backend.active_runs() == {"tester/k0": "cpu"} and set(backend._resources) == {("tester/k0", now)}


def test_remote_cancel_uses_the_session_id_logged_by_this_job(tmp_path):
    backend = KaggleBackend("tester", tmp_path)
    sent = []
    backend._kernels = lambda method, request, ref=None: (
        sent.append((method, request.kernel_session_id)) or Obj(error_message="")
    )
    backend.live_log = lambda ref: (
        "KGR workload other session 1\nKGR workload job1 session 352993764\ntick 3\n"
    )
    backend.cancel("tester/kgr-x", "job1")
    assert sent == [("cancel_kernel_session", 352993764)]
    backend.live_log = lambda ref: "KGR workload other session 1\n"
    with pytest.raises(ValueError, match="No session ID"):
        backend.cancel("tester/kgr-x", "job1")
    backend.live_log = lambda ref: "KGR workload job1 session 5\n"
    backend._kernels = lambda *args, **kwargs: Obj(error_message="Session already finished")
    with pytest.raises(RemoteError, match="already finished"):
        backend.cancel("tester/kgr-x", "job1")


def test_runtime_logs_the_session_id_before_any_setup(setup, tmp_path, monkeypatch):
    client, _, spec = setup
    job = client.submit(spec)
    job.attempts = [Attempt(number=1, ref="tester/kgr-session")]
    script = prepare_kernel(job, client.config.state_dir) / "workload.py"
    script.write_text(script.read_text().replace("/kaggle/working", str(tmp_path / "working")))
    env = dict(os.environ, KAGGLE_CONTAINER_NAME="kaggle_QLKaLNVAIohcv9mw-352993764-webtier")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"KGR workload {job.id} session 352993764"

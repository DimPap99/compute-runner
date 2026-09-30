"""The human and agent commands for looking across jobs and accounts, and for acting on several jobs."""

import json
import time

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from compute_runner.cli import app
from compute_runner.cli import views as views_module
from compute_runner.worker import DISCOVERY

from conftest import due


@pytest.fixture
def run(setup, monkeypatch):
    """Invoke the CLI on the setup fixture's queue; returns (output, exit code)."""
    client, backend, spec = setup
    monkeypatch.setattr("compute_runner.cli.Client", lambda **_: client)
    monkeypatch.setenv("COLUMNS", "200")

    def invoke(*args, input=None, code=0):
        result = CliRunner().invoke(app, list(args), input=input)
        assert result.exit_code == code, (result.output, result.exception)
        return result.output

    return invoke


def test_both_submit_commands_take_the_same_workload_options():
    command = get_command(app)

    def options(*path):
        found = command
        for name in path:
            found = found.commands[name]
        return {param.name for param in found.params}

    shared = {
        "entrypoint",
        "module",
        "gpu",
        "internet",
        "accelerator",
        "timeout",
        "arg",
        "param",
        "name",
        "input",
    }
    assert shared | {"requirements"} <= options("submit") & options("agent", "submit")


def test_list_filters_and_shows_names_as_written(setup, run):
    client, backend, spec = setup
    client.submit(spec.model_copy(update={"name": "a[/b] [bold]x"}))
    client.submit(spec.model_copy(update={"name": "trainer", "gpu": True}))
    done = client.submit(spec.model_copy(update={"name": "done"}))
    client.cancel(done.id)
    assert "a[/b] [bold]x" in run("list")
    listed = json.loads(run("--json", "list", "gpu"))
    assert [job["spec"]["name"] for job in listed] == ["trainer"]
    assert "done" not in run("list", "--active") and "done" in run("list", "--state", "cancelled")
    newest = json.loads(run("--json", "list", "--limit", "2"))
    assert [job["spec"]["name"] for job in newest] == ["trainer", "done"]
    assert "States must be a nonempty list of" in str(
        CliRunner().invoke(app, ["list", "--state", "nope"]).exception
    )


def test_account_list_and_gpus_show_totals_quota_reset_and_devices(setup, run):
    client, backend, spec = setup
    client.submit(spec.model_copy(update={"gpu": True}))
    client.worker().tick()
    saved = client.config.state_dir / DISCOVERY
    found = json.loads(saved.read_text())
    found["kaggle:tester"] |= {"gpu_refresh_at": time.time() + 2 * 86400 + 60, "devices": ["Tesla T4"]}
    saved.write_text(json.dumps(found))
    accounts = run("account", "list")
    assert "Total" in accounts and "1/1" in accounts and "27.8h" in accounts and "Failover: ask" in accounts
    gpus = run("gpus")
    assert "in 2d 0h" in gpus and "Tesla T4" in gpus


def test_running_watch_redraws_until_interrupted(setup, run, monkeypatch):
    client, backend, spec = setup
    client.submit(spec.model_copy(update={"name": "watched"}))
    client.worker().tick()
    redraws = []

    def sleep(seconds):
        redraws.append(seconds)
        if len(redraws) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(views_module.time, "sleep", sleep)
    output = run("running", "--watch", "--interval", "5")
    assert "watched" in output and redraws == [5, 5]


def test_logs_tail_prints_only_the_last_lines(setup, run):
    client, backend, spec = setup
    job = client.submit(spec)
    client.worker().tick()
    backend.remote[client.get(job.id).remote_ref]["state"] = "COMPLETE"
    backend.logs = lambda ref, follow=False: iter(["one\ntwo\nthree\n"])
    due(client, job.id)
    client.worker().tick()
    assert run("logs", job.id[:8], "--tail", "2") == "two\nthree\n"


def test_several_jobs_are_cancelled_after_confirmation_and_failed_ones_rerun(setup, run):
    client, backend, spec = setup
    batch = client.submit_batch([spec] * 3, request_key="three")
    ids = [job.id for job in batch.jobs]
    assert "Aborted" in run("cancel", "--batch", batch.id, input="n\n", code=1)
    assert all(client.get(job_id).state == "queued" for job_id in ids)
    assert "Cancelled 3 of 3 jobs" in run("cancel", "--batch", batch.id, "--yes")
    rerun = json.loads(run("--json", "retry", "--batch", batch.id, "--state", "cancelled"))
    assert sorted(job["parent_id"] for job in rerun) == sorted(ids)


def test_cleanup_reports_and_deletes_after_confirmation(setup, run):
    client, backend, spec = setup
    backend.artifacts = lambda: []
    job = client.submit(spec)
    client.worker().tick()
    client.store.update(
        job.id, state="succeeded", finished_at=time.time() - 30 * 86400, download_state="complete"
    )
    assert f"jobs/{job.id}" in run("cleanup") and "--delete removes them" in run("cleanup")
    run("cleanup", "--delete", input="n\n", code=1)
    assert (client.config.state_dir / "jobs" / job.id).exists()
    assert "Deleted 1 items" in run("cleanup", "--delete", "--yes")
    assert not (client.config.state_dir / "jobs" / job.id).exists()


def test_agent_commands_for_the_whole_picture(setup, run):
    client, backend, spec = setup
    batch = client.submit_batch([spec, spec.model_copy(update={"gpu": True})], request_key="two")
    client.worker().tick()

    def agent(*args, code=0):
        return json.loads(run("agent", *args, code=code))

    assert agent("overview")["running"]["counts"] == {"cpu": 1, "gpu": 1, "unknown": 0}
    assert agent("status", "--resource", "gpu")["total"] == 1
    assert agent("running", "--resource", "cpu")["total"] == 1
    assert agent("cleanup")["totals"]["reclaimable"]["count"] == 0
    assert agent("cancel", "--batch", batch.id)["cancelled"] == 2
    assert "Select either job IDs or a batch" in agent("cancel", code=1)["error"]

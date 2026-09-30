"""Views across accounts: the runs holding each account's slots, and the GPUs free on each."""

import json
import time

from rich.console import Console
from typer.testing import CliRunner

from compute_runner.cli import app
from compute_runner.providers import RemoteError
from compute_runner.worker import DISCOVERY


def cli(client, monkeypatch, *args):
    monkeypatch.setattr("compute_runner.cli.Client", lambda **_: client)
    monkeypatch.setattr("compute_runner.cli.console", Console(width=200))
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, (result.output, result.exception)
    return result.output


def test_running_lists_this_queues_runs_then_others_once_each_in_account_order(two_accounts):
    client, home, other, spec = two_accounts
    gpu = client.submit(spec.model_copy(update={"gpu": True, "name": "train"}))
    cpu = client.submit(spec.model_copy(update={"name": "eda"}))
    client.worker().tick()
    gpu_ref = client.get(gpu.id).remote_ref
    # Kaggle's listing includes this queue's own runs; each appears once, as this queue's job.
    home.external = {gpu_ref.upper(): "gpu"}
    other.external = {"other/notebook": "cpu", "other/mystery": "unknown"}
    agent = client.agent()
    result = agent.running(live=True)
    assert [(run["account"], run.get("job_id"), run["resource"]) for run in result["runs"]] == [
        ("kaggle:tester", gpu.id, "gpu"),
        ("kaggle:tester", cpu.id, "cpu"),
        ("kaggle:other", None, "cpu"),
        ("kaggle:other", None, "unknown"),
    ]
    assert result["counts"] == {"cpu": 2, "gpu": 1, "unknown": 1}
    assert result["runs"][0]["provider"] == "kaggle" and result["runs"][0]["state"] == "remote_queued"
    # A run of unknown resource holds both pools, as the worker counts it.
    assert [run["ref"] for run in agent.running("gpu", live=True)["runs"]] == [gpu_ref, "other/mystery"]
    assert agent.running(account="KAGGLE:OTHER", live=True)["account"] == "kaggle:other"
    # Without live, only what the worker has checked: it never needed kaggle:other.
    local = agent.running()
    assert local["total"] == 2 and local["discovery"]["kaggle:other"] == {"checked_age_seconds": None}


def test_free_slots_are_unknown_until_checked_and_none_once_gpu_time_is_spent(two_accounts):
    client, home, other, spec = two_accounts
    client.submit(spec.model_copy(update={"gpu": True}))
    client.worker().tick()
    first, second = client.agent().accounts()["accounts"]
    assert first["gpu"] == {"used": 1, "limit": 1, "free": 0} and first["cpu"]["free"] == 5
    assert second["gpu"]["free"] is None and second["cpu"]["free"] is None
    home.discovery_error = RemoteError("HTTP 429: slow down", "rate_limit")
    other.gpu_seconds = 0
    first, second = client.agent().accounts(live=True)["accounts"]
    assert first["gpu"]["free"] is None and first["error"] == "HTTP 429: slow down"
    assert second["gpu"] == {"used": 0, "limit": 1, "free": 0} and second["cpu"]["free"] == 5


def test_gpus_table_totals_every_account_and_marks_lower_bounds(two_accounts, monkeypatch):
    client, home, other, spec = two_accounts
    client.submit(spec.model_copy(update={"gpu": True}))
    client.worker().tick()
    totals = json.loads(cli(client, monkeypatch, "--json", "gpus", "--live"))["totals"]
    assert totals["gpu"] == {"used": 1, "limit": 2, "free": 1}
    assert totals["gpu_quota_seconds"] == 200000 and totals["complete"]
    table = cli(client, monkeypatch, "gpus")
    assert "kaggle:other" in table and "never" in table and ">=0" in table and ">=27.8h" in table
    assert "kaggle:other: not checked yet" in table and "--live asks the providers now" in table


def test_running_table_shows_names_and_errors_as_written_and_says_when_discovery_is_old(
    two_accounts, monkeypatch
):
    client, home, other, spec = two_accounts
    client.submit(spec.model_copy(update={"name": "[bold]sweep[/bold]", "gpu": True}))
    client.worker().tick()
    saved = client.config.state_dir / DISCOVERY
    found = json.loads(saved.read_text())
    found["kaggle:tester"]["checked_at"] = time.time() - 7200
    found["kaggle:other"] = {"error": "[Errno 111] Connection refused", "checked_at": None}
    saved.write_text(json.dumps(found))
    output = cli(client, monkeypatch, "running", "gpu")
    assert "[bold]sweep[/bold]" in output and "1 running: 1 GPU" in output
    assert "kaggle:other: last check failed: [Errno 111] Connection refused" in output
    assert "Accounts last checked up to 2h 00m ago" in output
    assert "0 running" in cli(client, monkeypatch, "running", "cpu", "--account", "kaggle:tester")


def test_discovery_saved_by_an_older_worker_still_loads(tmp_path):
    from compute_runner.worker import DiscoveryFile

    (tmp_path / DISCOVERY).write_text(
        json.dumps(
            {
                "kaggle:tester": {"runs": {"tester/k": "gpu"}, "checked_at": 5, "gpu_seconds": 7, "later": 1},
                "kaggle:other": "not a discovery",
            }
        )
    )
    found = DiscoveryFile(tmp_path).load()
    assert found["kaggle:tester"].runs == {"tester/k": "gpu"} and found["kaggle:tester"].known
    assert found["kaggle:tester"].devices == [] and found["kaggle:tester"].gpu_refresh_at is None
    assert not found["kaggle:other"].known

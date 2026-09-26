import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from typer.testing import CliRunner

from kaggle_runner import AgentClient, Client, JobSpec
from kaggle_runner.cli import app
from kaggle_runner.store import Store


def test_request_replay_survives_restart_missing_source_and_completion(setup):
    client, backend, spec = setup
    agent = AgentClient(client)
    first = agent.submit([spec, spec], request_key="experiment-v1")
    client.worker().tick()
    assert len(backend.pushes) == 2
    for ref in backend.remote:
        backend.remote[ref]["state"] = "COMPLETE"
    for job in client.list():
        client.store.update(job.id, next_action_at=0)
    client.worker().tick()
    spec.source.unlink()
    restarted = Client(config=client.config, backend=backend).agent()
    replay = restarted.submit([spec, spec], request_key="experiment-v1")
    assert replay["batch_id"] == first["batch_id"] and replay["replayed"]
    assert replay["counts"] == {"succeeded": 2}
    assert all(j["outputs_ready"] for j in replay["jobs"])
    assert len(client.list()) == 2
    client.worker().tick()
    assert len(backend.pushes) == 2


def test_key_conflict_does_not_change_original_request(setup):
    client, _, spec = setup
    first = client.submit(spec, request_key="intent/1")
    with pytest.raises(ValueError, match="different request"):
        client.submit(spec.model_copy(update={"gpu": True}), request_key="intent/1")
    spec.source.write_text("print('changed code')")
    assert client.submit(spec, request_key="intent/1").snapshot == first.snapshot
    assert client.submit(spec, request_key="intent/2").snapshot != first.snapshot
    assert len(client.list()) == 2


def test_concurrent_replays_commit_one_batch(setup):
    client, backend, spec = setup
    barrier = Barrier(6)

    def submit(_):
        other = Client(config=client.config, backend=backend)
        barrier.wait()
        return other.submit_batch([spec] * 3, request_key="parallel")

    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(submit, range(6)))
    assert len({b.id for b in batches}) == 1
    assert sum(not b.replayed for b in batches) == 1
    assert len(client.list()) == 3
    assert len({tuple(j.id for j in b.jobs) for b in batches}) == 1


def test_batch_snapshot_failure_is_atomic_and_key_remains_usable(setup, tmp_path):
    client, _, spec = setup
    missing = JobSpec(source=tmp_path / "missing.py")
    with pytest.raises((OSError, ValueError)):
        client.submit_batch([spec, missing], request_key="atomic")
    assert client.list() == []
    assert client.agent().changes()["cursor"] == 0
    missing.source.write_text("print(1)")
    batch = client.submit_batch([spec, missing], request_key="atomic")
    assert len(batch.jobs) == 2 and not batch.replayed


def test_batch_database_failure_rolls_back_receipt_jobs_and_events(setup, monkeypatch):
    client, _, spec = setup
    original = client.store._insert_job
    calls = 0

    def failing(db, job):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        original(db, job)

    monkeypatch.setattr(client.store, "_insert_job", failing)
    with pytest.raises(OSError, match="disk full"):
        client.submit_batch([spec] * 2, request_key="atomic")
    assert client.list() == []
    assert client.agent().changes()["cursor"] == 0
    monkeypatch.setattr(client.store, "_insert_job", original)
    batch = client.submit_batch([spec] * 2, request_key="atomic")
    assert not batch.replayed and len(batch.jobs) == 2


def test_retry_is_idempotent_and_has_parent(setup):
    client, _, spec = setup
    job = client.submit(spec, request_key="original")
    agent = client.agent()
    with pytest.raises(ValueError, match="terminal or blocked"):
        agent.retry(job.id, request_key="retry")
    agent.cancel(job.id)
    assert agent.cancel(job.id)["jobs"][0]["state"] == "cancelled"
    first = agent.retry(job.id, request_key="retry")
    replay = agent.retry(job.id, request_key="retry")
    assert first["batch_id"] == replay["batch_id"] and replay["replayed"]
    assert first["jobs"][0]["parent_id"] == job.id
    assert len(client.list()) == 2
    with pytest.raises(ValueError, match="different request"):
        agent.retry(job.id, request_key="original")


def test_status_is_bounded_and_omits_manifests_and_settings(setup):
    client, _, spec = setup
    spec = spec.model_copy(update={"env": {"VALUE": "not-for-agent-output"}, "name": "n" * 10000})
    batch = client.submit_batch([spec] * 23)
    client.store.update(batch.jobs[3].id, state="failed", error="e" * 5000)
    agent = client.agent()
    page = agent.status(batch_id=batch.id, limit=7)
    assert page["counts"] == {"queued": 22, "failed": 1}
    assert page["total"] == 23 and len(page["jobs"]) == 7 and page["next_offset"] == 7
    ids = []
    offset = 0
    while offset is not None:
        page = agent.status(batch_id=batch.id, limit=7, offset=offset)
        ids.extend(j["id"] for j in page["jobs"])
        text = json.dumps(page)
        assert len(text) < 6000
        assert all(
            term not in text for term in ("not-for-agent-output", '"snapshot"', '"sha256"', '"source"')
        )
        offset = page["next_offset"]
    assert ids == [job.id for job in batch.jobs]
    filtered = agent.status([batch.jobs[3].id[:12]], states=["failed"])
    assert len(filtered["jobs"][0]["error"]) <= 400


def test_batch_pages_preserve_input_order_when_clock_moves_backwards(setup):
    client, _, spec = setup
    batch = client.submit_batch([spec] * 3)
    for index, job in enumerate(batch.jobs):
        client.store.update(job.id, created_at=100 - index)
        with client.store.connection() as db:
            db.execute("UPDATE jobs SET created=? WHERE id=?", (100 - index, job.id))
    page = client.agent().status(batch_id=batch.id)
    assert [j["id"] for j in page["jobs"]] == [j.id for j in batch.jobs]
    assert [j["batch_index"] for j in page["jobs"]] == [0, 1, 2]


def test_change_cursor_coalesces_paginates_and_ignores_poll_noise(setup):
    client, _, spec = setup
    batch = client.submit_batch([spec] * 5)
    agent = client.agent()
    for _ in range(3):
        for job in batch.jobs:
            client.store.update(job.id, last_polled_at=123, next_action_at=456)
    first = agent.changes(limit=2)
    assert first["has_more"] and len(first["jobs"]) == 2
    seen = {j["id"] for j in first["jobs"]}
    cursor = first["cursor"]
    # A changed job can appear again between pages; it cannot be missed.
    changed_id = first["jobs"][0]["id"]
    client.store.update(changed_id, state="running")
    client.store.update(changed_id, state="failed", download_state="error", download_error="broken output")
    latest = {}
    while True:
        page = agent.changes(after=cursor, limit=2)
        assert page["cursor"] >= cursor
        cursor = page["cursor"]
        seen.update(j["id"] for j in page["jobs"])
        latest.update({j["id"]: j for j in page["jobs"]})
        if not page["has_more"]:
            break
    assert seen == {j.id for j in batch.jobs}
    assert latest[changed_id]["state"] == "failed"
    assert latest[changed_id]["download_error"] == "broken output"
    assert agent.changes(after=cursor)["jobs"] == []
    client.store.update(changed_id, download_error="different output error")
    assert len(agent.changes(after=cursor)["jobs"]) == 1
    with pytest.raises(ValueError, match="ahead"):
        agent.changes(after=10**9)


def test_filtered_cursor_advances_past_other_batches_without_losing_later_events(setup):
    client, _, spec = setup
    first = client.submit_batch([spec])
    other = client.submit_batch([spec])
    agent = client.agent()
    page = agent.changes(batch_id=first.id)
    assert [j["id"] for j in page["jobs"]] == [first.jobs[0].id]
    client.store.update(other.jobs[0].id, state="running")
    empty = agent.changes(after=page["cursor"], batch_id=first.id)
    assert not empty["jobs"] and empty["cursor"] > page["cursor"]
    client.store.update(first.jobs[0].id, state="running")
    assert agent.changes(after=empty["cursor"], batch_id=first.id)["jobs"][0]["state"] == "running"


def test_old_queue_migrates_and_legacy_events_are_visible(setup):
    client, _, spec = setup
    job = client.submit(spec)
    # Recreate the v1 schema around a real persisted record.
    with sqlite3.connect(client.store.db) as db:
        db.executescript(
            "DROP TABLE batch_jobs; DROP TABLE batches; UPDATE meta SET value='1' WHERE key='schema_version';"
        )
    migrated = Store(client.config.state_dir)
    assert migrated.get(job.id).snapshot == job.snapshot
    summary = Client(config=client.config).agent().changes()["jobs"][0]
    assert summary["id"] == job.id and summary["batch_id"] is None
    with migrated.connection() as db:
        assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "2"


def test_bounded_utf8_logs_are_cached_and_failed_refresh_keeps_old_copy(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    client.worker().tick()
    calls = []
    data = "".join(f"line {i}: Ελληνικά 🌍\n" for i in range(200))

    def logs(ref, follow=False):
        assert not follow
        calls.append(ref)
        yield data[:100]
        yield data[100:]

    backend.logs = logs
    agent = client.agent()
    result = agent.logs(job.id, tail=3, max_bytes=101)
    assert result["bytes"] <= 101 and len(result["text"].splitlines()) <= 3
    assert result["truncated"] and result["fetched"]
    assert result["text"].endswith("Ελληνικά 🌍\n")
    path = Path(result["path"])
    assert path.read_text() == data and path.stat().st_mode & 0o777 == 0o600
    assert not agent.logs(job.id)["fetched"] and len(calls) == 1

    def broken(*args, **kwargs):
        yield "partial replacement"
        raise RuntimeError("network failure")

    backend.logs = broken
    with pytest.raises(RuntimeError, match="network failure"):
        agent.logs(job.id, refresh=True)
    assert path.read_text() == data
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("status", {"limit": 101}),
        ("status", {"offset": -1}),
        ("status", {"states": ["bogus"]}),
        ("status", {"job_ids": "abc"}),
        ("changes", {"after": -1}),
        ("changes", {"limit": 0}),
        ("logs", {"job_id": "abc", "tail": 501}),
        ("logs", {"job_id": "abc", "max_bytes": 65537}),
    ],
)
def test_agent_bounds_fail_before_remote_access(setup, method, kwargs):
    client, _, _ = setup
    with pytest.raises(ValueError):
        getattr(client.agent(), method)(**kwargs)


def test_cli_agent_batch_workflow_and_structured_errors(setup, tmp_path, monkeypatch):
    client, backend, spec = setup
    monkeypatch.setattr("kaggle_runner.cli.Client", lambda **_: client)
    config = tmp_path / "batch.yaml"
    config.write_text(
        "jobs:\n  - source: hello.py\n  - source: hello.py\n    gpu: true\n    internet: true\n"
    )
    runner = CliRunner()

    def command(args, code=0):
        result = runner.invoke(app, ["agent", *args])
        assert result.exit_code == code, (result.output, result.exception)
        return json.loads(result.stdout)

    preview = command(["submit", str(config), "--request-key", "cli", "--dry-run"])
    assert preview["total"] == 2 and preview["gpu_jobs"] == 1 and client.list() == []
    first = command(["submit", str(config), "--request-key", "cli"])
    assert first["total"] == 2 and not first["replayed"]
    assert command(["submit", str(config), "--request-key", "cli"])["replayed"]
    assert command(["status", "--batch", first["batch_id"]])["total"] == 2
    job_id = first["jobs"][0]["id"]
    assert command(["status", job_id[:12]])["total"] == 1
    command(["submit", str(config), "--request-key", "cli", "--cpu"], code=1)
    changed = command(["changes", "--batch", first["batch_id"], "--limit", "1"])
    assert changed["has_more"]
    client.worker().tick()
    assert len(backend.pushes) == 2
    assert command(["logs", job_id])["text"] == "example log\n"
    assert "error" in command(["cancel", job_id], code=1)
    assert "error" in command(["status", "--limit", "101"], code=1)
    assert "error" in command(["changes", "--batch", "unknown"], code=1)
    assert "error" in command(["submit", str(config), "--request-key", "bad key"], code=1)
    assert len(client.list()) == 2


def test_core_submission_keys_are_optional_but_agent_keys_required(setup):
    client, _, spec = setup
    assert client.submit(spec).id != client.submit(spec).id
    with pytest.raises(ValueError, match="required"):
        client.agent().submit([spec], request_key=None)


def test_status_and_changes_are_local_only(setup):
    client, _, spec = setup
    client.submit(spec)
    client._backend = None
    agent = client.agent()
    assert agent.status()["total"] == 1
    assert agent.changes()["jobs"]
    assert client._backend is None


def test_cli_reports_invalid_yaml_and_database_failures_as_json(setup, tmp_path, monkeypatch):
    client, _, _ = setup
    monkeypatch.setattr("kaggle_runner.cli.Client", lambda **_: client)
    broken = tmp_path / "broken.yaml"
    broken.write_text("jobs: [\n")
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "submit", str(broken), "--request-key", "broken"])
    assert result.exit_code == 1 and json.loads(result.stdout)["error"]

    def locked(**kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(client.store, "page", locked)
    result = runner.invoke(app, ["agent", "status"])
    assert result.exit_code == 1 and json.loads(result.stdout)["error"] == "database is locked"


@pytest.mark.parametrize("agent_mode", [False, True])
def test_cpu_override_clears_gpu_accelerator(setup, tmp_path, monkeypatch, agent_mode):
    client, _, _ = setup
    monkeypatch.setattr("kaggle_runner.cli.Client", lambda **_: client)
    config = tmp_path / "gpu.yaml"
    config.write_text("source: hello.py\ngpu: true\naccelerator: NvidiaTeslaT4\n")
    command = ["agent"] if agent_mode else ["--json"]
    runner = CliRunner()
    result = runner.invoke(app, [*command, "submit", str(config), "--request-key", "cpu", "--cpu"])
    assert result.exit_code == 0, result.output
    job = client.list()[0]
    assert not job.spec.gpu and job.spec.accelerator is None
    result = runner.invoke(
        app,
        [
            *command,
            "submit",
            str(config),
            "--request-key",
            "contradiction",
            "--cpu",
            "--accelerator",
            "NvidiaTeslaT4",
        ],
    )
    assert result.exit_code != 0 and len(client.list()) == 1

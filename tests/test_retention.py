"""What the queue keeps: slim records, one event per job, and, when asked, retention."""

import json
import sqlite3
import time

import pytest

from compute_runner.providers import Artifact
from compute_runner.store import Store

from conftest import due


def finished_long_ago(client, job_id):
    client.store.update(
        job_id, state="succeeded", finished_at=time.time() - 30 * 86400, download_state="complete"
    )


def test_opening_an_older_queue_slims_its_records_and_events_once(setup, tmp_path):
    client, _, spec = setup
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("a")
    job = client.submit(spec.model_copy(update={"inputs": {"data": data}}))
    saved = job.snapshot["inputs"]["data"]
    assert set(saved) == {"digest", "bytes"}
    # As older versions wrote them: every input's file list, and every event.
    fat = json.loads(job.model_dump_json())
    fat["snapshot"]["inputs"]["data"]["files"] = {"a.txt": {"sha256": "0" * 64, "size": 1}}
    # A record this version cannot load is slimmed too.
    unreadable = dict(fat, id="f" * 32, spec=dict(fat["spec"], params={"lr": None}))
    with sqlite3.connect(client.store.db) as db:
        db.execute("UPDATE jobs SET record=? WHERE id=?", (json.dumps(fat), job.id))
        db.execute(
            "INSERT INTO jobs VALUES (?,?,?,?)", (unreadable["id"], 0, "failed", json.dumps(unreadable))
        )
        db.executemany(
            "INSERT INTO events(job_id,timestamp,state,detail) VALUES(?,?,?,?)",
            [(job.id, 0, "queued", "")] * 3,
        )
        db.execute("DELETE FROM meta WHERE key='compacted'")
    store = Store(client.config.state_dir)
    assert store.get(job.id).snapshot["inputs"]["data"] == saved
    with sqlite3.connect(store.db) as db:
        records = [json.loads(row[0]) for row in db.execute("SELECT record FROM jobs")]
        assert [record["snapshot"]["inputs"]["data"] for record in records] == [saved, saved]
        assert db.execute("SELECT COUNT(*) FROM events WHERE job_id=?", (job.id,)).fetchone() == (1,)
        assert db.execute("SELECT value FROM meta WHERE key='compacted'").fetchone() == ("1",)


def test_only_each_jobs_latest_event_is_kept_and_cursors_outlive_removed_jobs(setup):
    client, _, spec = setup
    first, second, running = client.submit_many([spec, spec, spec])
    for job in (first, second):
        client.store.update(job.id, state="cancelled", finished_at=time.time())
    with sqlite3.connect(client.store.db) as db:
        assert db.execute("SELECT COUNT(*) FROM events").fetchone() == (3,)
    cursor, _, _ = client.store.changes()
    # The newest event was the removed job's; the cursor still names a point in this queue.
    assert client.store.forget([second.id, running.id]) == [second.id]
    assert client.store.changes(after=cursor) == (cursor, False, [])
    assert [job.id for job in client.list()] == [first.id, running.id]


def test_a_job_waiting_for_dataset_processing_records_no_event_each_cycle(setup, tmp_path):
    client, backend, spec = setup
    backend.upload_ready = False
    data = tmp_path / "data.txt"
    data.write_text("input")
    job = client.submit(spec.model_copy(update={"inputs": {"data": data}}))
    client.worker().tick()
    assert client.get(job.id).wait_reason == "Waiting for dataset processing"
    cursor, _, _ = client.store.changes()
    for _ in range(3):
        due(client, job.id)
        client.worker().tick()
    assert client.store.changes(after=cursor)[2] == []
    backend.upload_ready = True
    due(client, job.id)
    client.worker().tick()
    assert [changed.id for changed in client.store.changes(after=cursor)[2]] == [job.id]


def test_retention_deletes_old_local_leftovers_and_records_once_a_day(setup):
    client, backend, spec = setup
    client.config.retention_days = 7
    notebooks = []
    backend.artifacts = lambda: notebooks

    def never(artifact):
        pytest.fail("Retention must not delete remote items")

    backend.delete_artifact = never
    kept, gone = client.submit_many([spec, spec])
    client.worker().tick()
    for job in (kept, gone):
        finished_long_ago(client, job.id)
    # kept's notebook is still on the account, so its record must outlast it.
    ref = client.get(kept.id).remote_ref
    notebooks.append(Artifact("notebook", ref, attempt=ref))
    client.worker().tick()
    assert len(client.list()) == 2  # The first cycle already made today's pass.
    client.store.set_meta("retention_at", time.time() - 86400)
    client.worker().tick()
    assert [job.id for job in client.list()] == [kept.id]
    assert not any((client.config.state_dir / "jobs").iterdir())

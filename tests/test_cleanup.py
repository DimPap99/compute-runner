"""Cleanup removes only what this runner made for jobs that are finished, downloaded and old."""

import time

from compute_runner.providers import Artifact


def finished_long_ago(client, job_id, days=30, downloads="complete"):
    old = time.time() - days * 86400
    client.store.update(job_id, state="succeeded", finished_at=old, download_state=downloads)


def launched(client, spec, tmp_path, name):
    data = tmp_path / f"{name}.txt"
    data.write_text(name)
    job = client.submit(spec.model_copy(update={"name": name, "inputs": {"data": data}}))
    client.worker().tick()
    return client.get(job.id)


def test_cleanup_reports_what_may_go_and_deletes_only_that(setup, tmp_path):
    client, backend, spec = setup
    old = launched(client, spec, tmp_path, "old")
    busy = launched(client, spec, tmp_path, "busy")
    finished_long_ago(client, old.id)
    digest = old.snapshot["inputs"]["data"]["digest"]
    remote = [
        Artifact("notebook", old.remote_ref, attempt=old.remote_ref),
        Artifact("notebook", busy.remote_ref, attempt=busy.remote_ref),
        Artifact(
            "notebook", "tester/kgr-elsewhere-0123456789ab-a1", attempt="tester/kgr-elsewhere-0123456789ab-a1"
        ),
        Artifact("dataset", f"tester/kgr-b-{digest[:40]}", bytes=5, digest=digest[:40]),
    ]
    deleted = []
    backend.artifacts = lambda: remote
    backend.delete_artifact = deleted.append
    report = client.cleanup()
    verdicts = {item["name"]: (item["verdict"], item["reason"]) for item in report["items"]}
    assert set(verdicts) == {old.remote_ref, f"tester/kgr-b-{digest[:40]}", f"jobs/{old.id}"}
    assert report["totals"]["unknown"]["count"] == 1
    # The busy job's notebook, its staging folder and every snapshot stay.
    assert report["totals"]["kept"]["count"] >= 4
    report = client.cleanup(delete=True)
    assert [artifact.name for artifact in deleted] == [old.remote_ref, f"tester/kgr-b-{digest[:40]}"]
    assert report["deleted"]["deleted"] == 3 and report["deleted"]["failed"] == {}
    assert not (client.config.state_dir / "jobs" / old.id).exists()
    assert (client.config.state_dir / "jobs" / busy.id).exists()
    assert (client.config.state_dir / "bundles" / digest).exists()


def test_snapshots_go_only_when_asked_and_recent_or_undownloaded_runs_stay(setup, tmp_path):
    client, backend, spec = setup
    backend.artifacts = lambda: []
    recent = launched(client, spec, tmp_path, "recent")
    pending = launched(client, spec, tmp_path, "pending")
    finished_long_ago(client, recent.id, days=1)
    finished_long_ago(client, pending.id, downloads="error")
    kept = {item["name"] for item in client.cleanup()["items"]}
    assert kept == set()
    finished_long_ago(client, recent.id)
    report = client.cleanup(include_snapshots=True, accounts=[])
    names = {item["name"] for item in report["items"]}
    recent_digest = recent.snapshot["inputs"]["data"]["digest"]
    assert f"bundles/{recent_digest}" in names and f"jobs/{recent.id}" in names
    assert not any(pending.id in name for name in names)
    assert report["by_location"] == {"local": report["totals"]["reclaimable"]}


def test_one_unreachable_account_is_reported_and_the_rest_still_listed(two_accounts):
    client, home, other, spec = two_accounts

    def offline():
        raise RuntimeError("Cannot reach Kaggle")

    home.artifacts = offline
    other.artifacts = lambda: [
        Artifact("notebook", "other/kgr-x-0123456789ab-a1", attempt="other/kgr-x-0123456789ab-a1")
    ]
    report = client.cleanup(local=False)
    assert report["errors"] == {"kaggle:tester": "Cannot reach Kaggle"}
    assert report["totals"]["unknown"]["count"] == 1

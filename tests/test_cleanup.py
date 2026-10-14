"""Cleanup removes only what this runner made for jobs that are finished, downloaded and old."""

import time

import pytest

from compute_runner.bundle import bundle_payload
from compute_runner.cleanup import Cleanup
from compute_runner.providers import Artifact


def finished_long_ago(client, job_id, days=30, downloads="complete"):
    old = time.time() - days * 86400
    client.store.update(job_id, state="succeeded", finished_at=old, download_state=downloads)


def record(items, job_id):
    return next(item for item in items if item.kind == "record" and item.name == job_id)


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
    # The old job's record too: every account was checked, and its remote items may go with it.
    assert set(verdicts) == {old.remote_ref, f"tester/kgr-b-{digest[:40]}", f"jobs/{old.id}", old.id}
    assert report["totals"]["unknown"]["count"] == 1
    # The busy job's notebook, staging folder and record, and every snapshot, stay.
    assert report["totals"]["kept"]["count"] >= 5
    report = client.cleanup(delete=True)
    assert [artifact.name for artifact in deleted] == [old.remote_ref, f"tester/kgr-b-{digest[:40]}"]
    assert report["deleted"]["deleted"] == 4 and report["deleted"]["failed"] == {}
    assert not (client.config.state_dir / "jobs" / old.id).exists()
    assert (client.config.state_dir / "jobs" / busy.id).exists()
    assert (client.config.state_dir / "bundles" / digest).exists()
    assert [job.id for job in client.list()] == [busy.id]


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


def test_a_record_goes_last_and_only_with_its_remote_items(two_accounts, tmp_path):
    client, home, other, spec = two_accounts
    old = launched(client, spec, tmp_path, "old")
    finished_long_ago(client, old.id)
    home.artifacts = lambda: [Artifact("notebook", old.remote_ref, attempt=old.remote_ref)]
    other.artifacts = lambda: []
    checks_all = "records go only in a cleanup that checks every account"
    assert record(Cleanup(client, accounts=["kaggle:tester"]).items(), old.id).reason == checks_all

    def offline():
        raise RuntimeError("Cannot reach Kaggle")

    other.artifacts = offline
    assert record(Cleanup(client).items(), old.id).reason == checks_all
    other.artifacts = lambda: []

    def refuse(artifact):
        raise RuntimeError("Kaggle is busy")

    home.delete_artifact = refuse
    planner = Cleanup(client)
    items = planner.items()
    assert record(items, old.id).verdict == "reclaimable"
    assert list(planner.delete(items)["failed"]) == [f"kaggle:tester {old.remote_ref}"]
    # Its notebook stayed, so its record stays: without it, the notebook would be nobody's.
    assert client.get(old.id).state == "succeeded"
    home.delete_artifact = lambda artifact: None
    planner = Cleanup(client)
    assert planner.delete(planner.items())["deleted"] == 2
    with pytest.raises(KeyError):
        client.get(old.id)


def test_a_record_of_an_account_no_longer_configured_stays(two_accounts, tmp_path):
    client, home, other, spec = two_accounts
    job = client.submit(spec, account="kaggle:other")
    finished_long_ago(client, job.id)
    client.config.accounts = client.config.accounts[:1]
    home.artifacts = lambda: []
    assert record(Cleanup(client).items(), job.id).reason == "its account kaggle:other is not configured"


def test_removed_records_keep_their_runs_md_row_and_refuse_their_request_key(setup):
    client, backend, spec = setup
    backend.artifacts = lambda: []
    first = client.submit(spec, request_key="exp-1")
    client.worker().tick()
    finished_long_ago(client, first.id)
    client.worker().tick()
    client.cleanup(delete=True)
    assert client.list() == []
    # Replaying would otherwise return an empty batch, or tempt a caller to run the work again.
    with pytest.raises(ValueError, match="removed by cleanup; use a new key"):
        client.submit(spec, request_key="exp-1")
    second = client.submit(spec)
    client.worker().tick()
    assert second.run == 2
    index = (second.result_dir.parent / "runs.md").read_text()
    assert f"| {first.result_dir.name} | succeeded |" in index and f"| {second.result_dir.name} |" in index


def test_unpacked_copies_earlier_versions_kept_may_always_go(setup):
    client, backend, spec = setup
    backend.artifacts = lambda: []
    job = client.submit(spec)
    bundle = bundle_payload(client.config.state_dir, job.snapshot["source"]["digest"]).parent
    (bundle / "files").mkdir()
    (bundle / "files/hello.py").write_text("print('hello')\n")
    items = {item.name: item for item in Cleanup(client).items()}
    unpacked = items[f"bundles/{bundle.name}/files"]
    assert (unpacked.kind, unpacked.verdict, unpacked.bytes) == ("unpacked", "reclaimable", 15)
    # The snapshot itself stays for its queued job, counted without the copy.
    snapshot = items[f"bundles/{bundle.name}"]
    assert snapshot.verdict == "kept" and snapshot.bytes == (bundle / "payload.zip").stat().st_size
    client.cleanup(delete=True)
    assert [path.name for path in bundle.iterdir()] == ["payload.zip"]

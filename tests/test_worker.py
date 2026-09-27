import time

import pytest

from compute_runner import Account, JobSpec
from compute_runner.providers import RemoteError
from compute_runner.worker import outstanding
from conftest import due


def test_five_slots_sixth_waits_and_starts_after_completion(setup):
    client, backend, spec = setup
    jobs = client.submit_many([spec] * 6)
    worker = client.worker()
    worker.tick()
    assert len(backend.pushes) == 5
    assert client.get(jobs[5].id).state == "queued"
    first = client.get(jobs[0].id)
    backend.remote[first.remote_ref] = dict(state="COMPLETE", error=None)
    due(client, first.id)
    worker.tick()
    assert len(backend.pushes) == 6
    assert client.get(first.id).state == "succeeded"
    assert client.get(first.id).download_state == "complete"


def test_existing_account_runs_reduce_capacity(setup):
    client, backend, spec = setup
    backend.external = {f"tester/external-{i}": "cpu" for i in range(4)}
    client.submit_many([spec] * 3)
    client.worker().tick()
    assert len(backend.pushes) == 1


def test_account_inventory_matches_own_runs_ignoring_case(setup):
    client, backend, spec = setup
    client.submit_many([spec] * 2)
    worker = client.worker()
    worker.tick()
    backend.external = {ref.upper(): "cpu" for ref in backend.remote}
    worker.discovery["kaggle:tester"].checked_at = None  # Discover again now.
    client.submit_many([spec] * 3)
    worker.tick()
    assert len(backend.pushes) == 5


def test_notebook_slug_never_ends_with_a_hyphen(setup):
    client, backend, spec = setup
    client.submit(spec.model_copy(update={"name": "my experiment 1 v2"}))
    client.worker().tick()
    assert "--" not in backend.pushes[0]["id"]


def test_gpu_block_does_not_block_cpu_and_recovers(setup):
    client, backend, spec = setup
    backend.gpu_seconds = 0
    gpu = client.submit(spec.model_copy(update={"gpu": True}))
    client.submit(spec)
    worker = client.worker()
    worker.tick()
    assert len(backend.pushes) == 1 and not backend.pushes[0]["enable_gpu"]
    assert client.get(gpu.id).state == "queued"
    backend.gpu_seconds = 3600
    worker.discovery["kaggle:tester"].checked_at = None  # Quota is read with the next discovery.
    worker.tick()
    assert len(backend.pushes) == 2 and backend.pushes[1]["enable_gpu"]


def test_unknown_discovery_reserves_capacity(setup):
    client, backend, spec = setup
    backend.discovery_error = RemoteError("offline")
    client.submit(spec)
    client.worker().tick()
    assert not backend.pushes


def test_crash_after_acceptance_recovers_without_duplicate(setup):
    client, backend, spec = setup
    job = client.submit(spec)

    def crash():
        raise SystemExit("simulated worker death")

    backend.on_push = crash
    with pytest.raises(SystemExit):
        client.worker().tick()
    assert client.get(job.id).state == "submitting"
    backend.on_push = None
    client.worker().tick()
    assert client.get(job.id).state == "remote_queued"
    assert len(backend.pushes) == 1


def test_unknown_response_becomes_attention_and_never_retries(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    backend.push_error = TimeoutError("connection dropped")
    worker = client.worker()
    worker.tick()
    saved = client.get(job.id)
    saved.attempts[-1].started_at = time.time() - 10
    client.store.update(job.id, attempts=[a.model_dump() for a in saved.attempts], next_action_at=0)
    worker.tick()
    assert client.get(job.id).state == "needs_attention"
    with pytest.raises(ValueError, match="may still exist"):
        client.retry(job.id)
    due(client, job.id)
    worker.tick()
    assert len(backend.pushes) == 1


def test_capacity_rejection_backoff_and_fresh_slug(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    backend.push_error = RemoteError(
        "Maximum batch CPU session count of 5 reached", "capacity", definitive=True
    )
    worker = client.worker()
    worker.tick()
    assert client.get(job.id).state == "queued"
    assert client.get(job.id).remote_ref is None  # A rejected attempt created no notebook.
    worker.tick()
    assert len(backend.pushes) == 1
    backend.push_error = None
    due(client, job.id)
    worker.tick()
    assert len(backend.pushes) == 2
    assert backend.pushes[0]["id"] != backend.pushes[1]["id"]


def test_failed_execution_not_rerun_and_explicit_retry_uses_snapshot(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    worker = client.worker()
    worker.tick()
    backend.remote[client.get(job.id).remote_ref] = dict(state="ERROR", error="bad code")
    due(client, job.id)
    worker.tick()
    worker.tick()
    assert client.get(job.id).state == "failed"
    assert len(backend.pushes) == 1
    spec.source.unlink()
    new = client.retry(job.id)
    assert new.id != job.id and new.snapshot == job.snapshot
    worker.tick()
    assert len(backend.pushes) == 2


def test_download_failure_does_not_rerun_job(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    worker = client.worker()
    worker.tick()
    backend.remote[client.get(job.id).remote_ref] = dict(state="COMPLETE", error=None)
    backend.download_error = OSError("disk full")
    due(client, job.id)
    worker.tick()
    assert client.get(job.id).state == "succeeded"
    assert client.get(job.id).download_state == "error"
    backend.download_error = None
    due(client, job.id)
    worker.tick()
    assert client.get(job.id).download_state == "complete"
    assert len(backend.pushes) == 1


def test_download_retries_back_off(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    worker = client.worker()
    worker.tick()
    backend.remote[client.get(job.id).remote_ref] = dict(state="COMPLETE", error=None)
    backend.download_error = OSError("storage unavailable")
    delays = []
    for _ in range(3):
        due(client, job.id)
        worker.tick()
        failed = client.get(job.id)
        delays.append(round(failed.download_retry_at - failed.updated_at))
    assert delays == [60, 120, 240] and failed.download_failures == 3


def test_dataset_processing_precedes_submission(setup, tmp_path):
    client, backend, spec = setup
    data = tmp_path / "data"
    data.mkdir()
    (data / "file.txt").write_text("input")
    job = client.submit(spec.model_copy(update={"inputs": {"data": data}, "datasets": ["tester/existing"]}))
    backend.upload_ready = False
    worker = client.worker()
    worker.tick()
    assert client.get(job.id).state == "preparing"
    assert not backend.pushes
    backend.upload_ready = True
    due(client, job.id)
    worker.tick()
    assert len(backend.pushes) == 1
    assert "tester/existing/7" in backend.pushes[0]["dataset_sources"]


def test_resource_metadata_explicit_and_runtime_forwarded(setup):
    client, backend, spec = setup
    client.submit(
        JobSpec(source=spec.source, gpu=True, accelerator="NvidiaTeslaT4", internet=True, timeout_seconds=120)
    )
    client.worker().tick()
    metadata = backend.pushes[0]
    assert metadata["enable_gpu"] is True and metadata["enable_internet"] is True
    assert metadata["is_private"] is True and metadata["enable_tpu"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["timeout_seconds"] == 120


def test_pending_cancel_is_local_and_active_cancel_stops_the_remote_run(setup):
    client, backend, spec = setup
    queued = client.submit(spec)
    client.cancel(queued.id)
    client.worker().tick()
    assert not backend.pushes
    active = client.submit(spec)
    client.worker().tick()
    ref = client.get(active.id).remote_ref
    assert client.cancel(active.id).wait_reason == "Cancellation requested on kaggle:tester"
    assert backend.cancelled == [ref]
    client.worker().tick()
    assert client.get(active.id).wait_reason == "Cancellation requested on kaggle:tester"
    backend.remote[ref]["state"] = "CANCEL_ACKNOWLEDGED"
    due(client, active.id)
    client.worker().tick()
    cancelled = client.get(active.id)
    assert cancelled.state == "cancelled" and cancelled.download_state == "complete"
    with pytest.raises(ValueError, match="already finished"):
        client.cancel(active.id)


def test_cancelling_a_run_still_queued_on_the_provider_ends_it_at_once(setup):
    client, backend, spec = setup
    backend.delete_queued = True
    job = client.submit(spec)
    client.worker().tick()
    cancelled = client.cancel(job.id)
    assert cancelled.state == "cancelled" and cancelled.download_state == "disabled"
    assert cancelled.wait_reason == "Cancelled before it started on kaggle:tester"
    # The worker has nothing left to poll, and a retry is allowed.
    client.worker().tick()
    assert client.get(job.id).state == "cancelled"
    assert client.retry(job.id).state == "queued"


def test_cancel_refuses_unconfirmed_submissions(setup):
    client, backend, spec = setup
    backend.push_error = RemoteError("timeout", "uncertain")
    job = client.submit(spec)
    client.worker().tick()
    assert client.get(job.id).attempts[-1].state == "uncertain"
    with pytest.raises(ValueError, match="unconfirmed"):
        client.cancel(job.id)
    assert backend.cancelled == []


def test_only_one_worker_holds_lock(setup):
    client, _, _ = setup
    with client.store.worker_lock():
        with pytest.raises(RuntimeError, match="already owns"):
            client.worker().tick()


def test_job_on_an_account_the_worker_does_not_know_waits_for_a_restart(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    # The worker started before the account was added.
    accounts, client.config.accounts = client.config.accounts, [Account(user="someoneelse")]
    client.worker().tick()
    waiting = client.get(job.id)
    assert waiting.state == "queued"
    assert "kaggle:tester is unknown to the running worker" in waiting.wait_reason
    assert not backend.pushes
    client.config.accounts = accounts
    client.worker().tick()
    assert client.get(job.id).state == "remote_queued"


def test_fifo_waits_for_inputs_but_other_resource_pool_progresses(setup, tmp_path):
    client, backend, spec = setup
    data = tmp_path / "inputs"
    data.mkdir()
    (data / "input.txt").write_text("value")
    first = client.submit(spec.model_copy(update={"inputs": {"data": data}}))
    client.submit(spec)
    client.submit(spec.model_copy(update={"gpu": True}))
    backend.upload_ready = False
    worker = client.worker()
    worker.tick()
    assert client.get(first.id).state == "preparing"
    assert len(backend.pushes) == 1 and backend.pushes[0]["enable_gpu"]
    worker.tick()
    assert len(backend.pushes) == 1


def test_concurrent_clients_keep_every_submission(setup):
    from concurrent.futures import ThreadPoolExecutor
    from compute_runner import Client

    client, backend, spec = setup

    def submit(_):
        return Client(config=client.config, providers={"kaggle:tester": backend}).submit(spec).id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(submit, range(12)))
    assert len(set(ids)) == 12 and len(client.list()) == 12


def test_cancel_during_preparation_prevents_push(setup, tmp_path):
    client, backend, spec = setup
    data = tmp_path / "input.txt"
    data.write_text("input")
    job = client.submit(spec.model_copy(update={"inputs": {"data": data}}))
    original = backend.ensure_bundle

    def upload(bundle):
        client.cancel(job.id)
        return original(bundle)

    backend.ensure_bundle = upload
    client.worker().tick()
    assert client.get(job.id).state == "cancelled"
    assert not backend.pushes


def test_unresolvable_auth_error_becomes_attention(setup):
    client, backend, spec = setup
    job = client.submit(spec)
    backend.push_error = TimeoutError("dropped")
    worker = client.worker()
    worker.tick()
    saved = client.get(job.id)
    saved.attempts[-1].started_at = time.time() - 10
    client.store.update(job.id, attempts=[a.model_dump() for a in saved.attempts], next_action_at=0)

    def denied(ref):
        raise RemoteError("forbidden", "auth")

    backend.status = denied
    worker.tick()
    assert client.get(job.id).state == "needs_attention"
    assert len(backend.pushes) == 1


def _accepted_run_disappears(client, backend, spec):
    job = client.submit(spec)
    client.worker().tick()
    job = client.get(job.id)
    del backend.remote[job.remote_ref]
    job.attempts[-1].started_at -= 60  # Past the reconciliation window.
    client.store.update(job.id, attempts=job.attempts, next_action_at=0)
    return job


def test_a_deleted_run_can_be_resolved_and_frees_its_slot(setup):
    client, backend, spec = setup
    job = _accepted_run_disappears(client, backend, spec)
    client.worker().tick()
    assert client.get(job.id).state == "needs_attention"
    resolved = client.resolve_not_submitted(job.id)
    assert resolved.state == "blocked" and not outstanding(resolved)
    retried = client.retry(job.id)
    client.worker().tick()
    assert client.get(retried.id).state == "remote_queued"


def test_polling_does_not_undo_a_resolution_made_during_the_remote_call(setup):
    client, backend, spec = setup
    job = _accepted_run_disappears(client, backend, spec)
    client.worker().tick()
    client.store.update(job.id, next_action_at=0)
    missing = backend.status

    def resolved_meanwhile(ref):
        client.resolve_not_submitted(job.id)
        return missing(ref)

    backend.status = resolved_meanwhile
    client.worker().tick()
    assert client.get(job.id).state == "blocked"


def test_a_definitive_transient_launch_failure_is_retried_not_blocked(setup):
    # SSH reports a lost upload this way: nothing started, and trying again may work.
    client, backend, spec = setup
    backend.push_error = RemoteError("upload interrupted", "transient", definitive=True)
    job = client.submit(spec)
    client.worker().tick()
    job = client.get(job.id)
    assert job.state == "queued" and job.attempts[-1].state == "rejected"
    backend.push_error = None
    due(client, job.id)
    client.worker().tick()
    assert client.get(job.id).state == "remote_queued"


def test_following_a_job_before_its_launch_waits_for_it(setup, monkeypatch):
    # submit, then logs --follow at once: the job is still queued locally.
    import threading

    client, backend, spec = setup
    monkeypatch.setattr(client, "worker_health", lambda: {"running": True})
    job = client.submit(spec)
    launch = threading.Timer(0.5, client.worker().tick)
    launch.start()
    assert "".join(client.logs(job.id, follow=True)) == "example log\n"
    launch.join()
    cancelled = client.submit(spec)
    client.cancel(cancelled.id)
    with pytest.raises(ValueError, match=r"not been submitted yet \(cancelled\)"):
        list(client.logs(cancelled.id, follow=True))

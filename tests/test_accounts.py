"""Accounts, placement and failover between them."""

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from compute_runner import Account, Config
from compute_runner.cli import app
from compute_runner.providers import RemoteError
from compute_runner.providers.kaggle import KaggleProvider, _api_class
from conftest import due


def test_legacy_configuration_becomes_the_default_account(tmp_path):
    config = Config.model_validate(
        {"owner": "alice", "cpu_limit": 2, "gpu_limit": 0, "strict": True, "state_dir": str(tmp_path)}
    )
    assert config.accounts == [Account(user="alice", cpu_limit=2, gpu_limit=0)]
    assert config.account().id == "kaggle:alice" and config.strict and config.failover == "ask"
    assert Config.model_validate({"owner": ""}).accounts == []
    with pytest.raises(ValueError, match="configured: kaggle:alice"):
        config.account("kaggle:bob")


def test_saved_jobs_from_single_account_versions_keep_working(setup):
    client, backend, spec = setup
    job = client.submit(spec, request_key="before-upgrade")
    client.worker().tick()
    ref = client.get(job.id).remote_ref
    # Rewrite the record as the previous version stored it.
    with sqlite3.connect(client.store.db) as db:
        record = json.loads(db.execute("SELECT record FROM jobs WHERE id=?", (job.id,)).fetchone()[0])
        record["owner"] = record.pop("account").split(":")[1]
        for attempt in record["attempts"]:
            del attempt["account"], attempt["url"]
        db.execute("UPDATE jobs SET record=? WHERE id=?", (json.dumps(record), job.id))
    upgraded = client.get(job.id)
    assert upgraded.account == "kaggle:tester" and upgraded.attempts[0].account == "kaggle:tester"
    assert upgraded.url == f"https://www.kaggle.com/code/{ref}"
    assert client.submit(spec, request_key="before-upgrade").id == job.id
    backend.remote[ref] = dict(state="COMPLETE", error=None)
    due(client, job.id)
    client.worker().tick()
    assert client.get(job.id).state == "succeeded"


@pytest.mark.parametrize("policy", ["off", "ask", "auto"])
def test_exhausted_gpu_quota_follows_the_failover_policy(two_accounts, policy):
    client, home, other, spec = two_accounts
    client.config.failover = policy
    home.gpu_seconds = 0
    job = client.submit(spec.model_copy(update={"gpu": True}))
    client.worker().tick()
    job = client.get(job.id)
    assert not home.pushes
    if policy == "auto":
        assert job.account == "kaggle:other" and job.state == "remote_queued" and len(other.pushes) == 1
        assert job.url.startswith("https://www.kaggle.com/code/other/kgr-")
        assert job.attempts[0].account == "kaggle:other"
        return
    assert job.state == "queued"
    assert job.wait_reason == "Waiting for available GPU quota on kaggle:tester"
    assert job.suggested_account == ("kaggle:other" if policy == "ask" else None)
    summary = client.agent().status([job.id])["jobs"][0]
    assert summary["account"] == "kaggle:tester"
    assert summary.get("suggested_account") == job.suggested_account
    # The user approves; the agent moves the job and the worker starts it there.
    assert client.agent().move([job.id], account="kaggle:other")["moved"] == 1
    client.worker().tick()
    moved = client.get(job.id)
    assert moved.account == "kaggle:other" and moved.state == "remote_queued" and moved.suggested_account is None


@pytest.mark.parametrize("policy", ["ask", "auto"])
def test_launch_rejected_for_capacity_fails_over_although_discovery_saw_free_slots(two_accounts, policy):
    client, home, other, spec = two_accounts
    client.config.failover = policy
    home.push_error = RemoteError("Maximum batch CPU session count of 5 reached.", "capacity", definitive=True)
    job = client.submit(spec)
    worker = client.worker()
    worker.tick()
    assert client.get(job.id).wait_reason == "Waiting to retry rejected submission"
    worker.tick()  # Still backing off on the home account.
    job = client.get(job.id)
    if policy == "ask":
        assert job.state == "queued" and job.suggested_account == "kaggle:other" and not other.pushes
    else:
        assert job.account == "kaggle:other" and job.state == "remote_queued" and len(other.pushes) == 1


def test_busy_slots_move_only_as_many_jobs_as_the_other_account_can_start(two_accounts, tmp_path):
    client, home, other, spec = two_accounts
    client.config.failover = "auto"
    for account in client.config.accounts:
        account.cpu_limit = 1
    # Moved jobs stay in preparation there, so they must still count against its slots.
    other.upload_ready = False
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hi')\n")
    jobs = client.submit_many([spec.model_copy(update={"source": project, "entrypoint": "main.py"})] * 3)
    client.worker().tick()
    placed = [client.get(job.id) for job in jobs]
    assert [(job.account, job.state) for job in placed] == [
        ("kaggle:tester", "remote_queued"),
        ("kaggle:other", "preparing"),
        ("kaggle:tester", "queued"),
    ]
    assert placed[2].suggested_account is None and "CPU capacity on kaggle:tester" in placed[2].wait_reason
    with sqlite3.connect(client.store.db) as db:
        events = [row[0] for row in db.execute("SELECT detail FROM events WHERE job_id=?", (jobs[1].id,))]
    assert "Moved from kaggle:tester: Waiting for CPU capacity on kaggle:tester" in events


def test_moved_job_uploads_its_inputs_to_the_new_account(two_accounts, tmp_path):
    client, home, other, spec = two_accounts
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hi')\n")
    home.push_error = RemoteError("Invalid machine shape", "invalid", definitive=True)
    job = client.submit(spec.model_copy(update={"source": project, "entrypoint": "main.py"}))
    client.worker().tick()
    blocked = client.get(job.id)
    assert blocked.state == "blocked" and blocked.upload_refs["source"].startswith("tester/")
    moved = client.move(job.id, "kaggle:other")
    assert moved.state == "queued" and moved.upload_refs == {} and moved.wait_reason.startswith("Moved from")
    client.worker().tick()
    job = client.get(job.id)
    assert job.state == "remote_queued" and job.upload_refs["source"].startswith("other/")
    assert other.pushes[0]["dataset_sources"] == [job.upload_refs["source"]]
    assert [attempt.account for attempt in job.attempts] == ["kaggle:tester", "kaggle:other"]


def test_submitted_and_finished_jobs_cannot_move(two_accounts):
    client, home, other, spec = two_accounts
    job = client.submit(spec)
    client.worker().tick()
    with pytest.raises(ValueError, match="not been submitted"):
        client.move(job.id, "kaggle:other")
    result = client.agent().move([job.id], account="kaggle:other")
    assert result["moved"] == 0 and result["jobs"][0]["account"] == "kaggle:tester"
    with pytest.raises(ValueError, match="Unknown account"):
        client.agent().move([job.id], account="kaggle:nobody")


def test_move_during_preparation_stops_the_old_account_launch(two_accounts, tmp_path):
    client, home, other, spec = two_accounts
    data = tmp_path / "input.txt"
    data.write_text("input")
    job = client.submit(spec.model_copy(update={"inputs": {"data": data}}))
    original = home.ensure_bundle

    def upload(bundle):
        client.move(job.id, "kaggle:other")
        return original(bundle)

    home.ensure_bundle = upload
    client.worker().tick()
    assert not home.pushes
    assert client.get(job.id).account == "kaggle:other" and client.get(job.id).upload_refs == {}


def test_explicit_accounts_on_submit_and_retry(two_accounts):
    client, home, other, spec = two_accounts
    job = client.submit(spec, request_key="on-other", account="kaggle:other")
    assert job.account == "kaggle:other"
    assert client.submit(spec, request_key="on-other", account="KAGGLE:OTHER").id == job.id
    with pytest.raises(ValueError, match="different request"):
        client.submit(spec, request_key="on-other")
    client.worker().tick()
    other.remote[client.get(job.id).remote_ref] = dict(state="ERROR", error="boom")
    due(client, job.id)
    client.worker().tick()
    assert client.retry(job.id).account == "kaggle:other"
    assert client.retry(job.id, account="kaggle:tester").account == "kaggle:tester"


def test_agent_accounts_report_policy_slots_and_last_quota(two_accounts):
    client, home, other, spec = two_accounts
    client.submit(spec.model_copy(update={"gpu": True}))
    client.worker().tick()
    result = client.agent().accounts()
    assert result["failover"] == "ask" and result["default"] == "kaggle:tester"
    first, second = result["accounts"]
    assert first["id"] == "kaggle:tester" and first["provider"] == "kaggle"
    assert first["gpu"] == {"used": 1, "limit": 1} and first["cpu"] == {"used": 0, "limit": 5}
    assert first["gpu_quota_seconds"] == 100000 and first["checked_age_seconds"] == 0
    assert second["gpu"]["used"] == 0 and second["checked_age_seconds"] is None


def test_account_commands_keep_order_and_protect_unfinished_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()

    def run(*args):
        result = runner.invoke(app, ["--json", *args])
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
        return json.loads(result.output)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Credentials file not found"):
        run("account", "add", "kaggle", "other", "--credentials", str(missing))
    run("account", "add", "kaggle", "tester")
    assert run("account", "add", "kaggle", "other", "--default")["accounts"] == ["kaggle:other", "kaggle:tester"]
    # Updating an account typed in other casing keeps the ID its jobs refer to.
    assert run("account", "add", "kaggle", "TESTER", "--cpu-limit", "3")["account"] == "kaggle:tester"
    script = tmp_path / "hello.py"
    script.write_text("print(42)")
    job_id = run("submit", str(script))[0]["id"]
    with pytest.raises(ValueError, match="1 unfinished jobs use kaggle:other"):
        run("account", "remove", "kaggle:other")
    assert run("move", job_id, "--account", "kaggle:tester")["account"] == "kaggle:tester"
    assert run("account", "remove", "kaggle:other")["accounts"] == ["kaggle:tester"]
    listed = runner.invoke(app, ["agent", "accounts"])
    assert json.loads(listed.output)["default"] == "kaggle:tester"


def credentials_api(path, monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    api = _api_class(path)()
    api.authenticate()
    return api


def test_account_credentials_file_ignores_ambient_kaggle_settings(tmp_path, monkeypatch):
    path = tmp_path / "kaggle.json"
    path.write_text(json.dumps({"username": "tester", "key": "0" * 32}))
    api = credentials_api(
        path, monkeypatch, KAGGLE_USERNAME="intruder", KAGGLE_KEY="1" * 32, KAGGLE_API_TOKEN="KGAT_ambient_value"
    )
    assert api.config_values["username"] == "tester" and api.config_values["key"] == "0" * 32
    # The SDK transport would otherwise send the ambient token instead of this account's key.
    with api.build_kaggle_client() as client:
        assert client._http_client._session.auth == ("tester", "0" * 32)
    provider = KaggleProvider(Account(user="someoneelse", credentials=path), tmp_path)
    with pytest.raises(RemoteError, match="authenticate as tester") as error:
        provider.api
    assert error.value.kind == "auth"


def test_account_token_file_wins_over_the_environment_token(tmp_path, monkeypatch):
    from kaggle.api.kaggle_api_extended import KaggleApi

    owners = {"KGAT_file_token_value": "tester", "KGAT_environment_value": "intruder"}
    monkeypatch.setattr(KaggleApi, "_introspect_token", lambda self, token: owners.get(token))
    path = tmp_path / "access_token"
    path.write_text("KGAT_file_token_value\n")
    api = credentials_api(path, monkeypatch, KAGGLE_API_TOKEN="KGAT_environment_value")
    assert api.config_values["username"] == "tester"
    assert api.config_values["token"] == "KGAT_file_token_value"
    with api.build_kaggle_client() as client:
        assert client._http_client._session.auth.token == "KGAT_file_token_value"


def test_auto_failover_does_not_bounce_a_job_between_full_accounts(two_accounts):
    client, home, other, spec = two_accounts
    client.config.failover = "auto"
    client.config.retry_seconds = 60
    home.push_error = other.push_error = RemoteError("Maximum CPU session count reached", "capacity", definitive=True)
    job = client.submit(spec)
    worker = client.worker()
    for _ in range(4):
        worker.tick()
    job = client.get(job.id)
    # Rejected on tester, moved once, rejected on other; it retries there instead of moving back.
    assert [attempt.account for attempt in job.attempts] == ["kaggle:tester", "kaggle:other"]
    assert job.account == "kaggle:other" and job.state == "queued" and job.suggested_account is None


def test_a_job_retrying_uploads_keeps_its_account_and_does_not_push_others_away(two_accounts, tmp_path):
    client, home, other, spec = two_accounts
    client.config.failover = "auto"
    client.config.retry_seconds = 60
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hi')\n")
    data = tmp_path / "data.txt"
    data.write_text("input")
    uploaded = home.ensure_bundle

    def second_upload_fails(bundle):
        if home.uploads:
            raise RemoteError("Connection reset")
        return uploaded(bundle)

    home.ensure_bundle = second_upload_fails
    first = client.submit(spec.model_copy(update={"source": project, "entrypoint": "main.py", "inputs": {"data": data}}))
    later = client.submit_many([spec] * 2)
    worker = client.worker()
    worker.tick()
    worker.tick()
    first = client.get(first.id)
    assert first.state == "preparing" and first.account == "kaggle:tester" and "input:data" in first.upload_refs
    for job in map(client.get, [job.id for job in later]):
        assert job.account == "kaggle:tester" and job.state == "queued" and job.suggested_account is None
        assert job.wait_reason == "Queued behind a job preparing on kaggle:tester"
    assert not other.pushes and not other.uploads


def test_a_new_failover_suggestion_is_reported_as_a_change(two_accounts):
    client, home, other, spec = two_accounts
    home.gpu_seconds = other.gpu_seconds = 0
    job = client.submit(spec.model_copy(update={"gpu": True}))
    worker = client.worker()
    worker.tick()
    cursor = client.agent().changes()["cursor"]
    other.gpu_seconds = 3600
    worker.discovery["kaggle:other"].checked_at = None
    worker.tick()
    changed = client.agent().changes(after=cursor)["jobs"]
    assert [(item["id"], item["suggested_account"]) for item in changed] == [(job.id, "kaggle:other")]


def test_waiting_jobs_do_not_query_other_accounts_every_cycle(two_accounts):
    client, home, other, spec = two_accounts
    calls = []
    quota = other.quota
    other.quota = lambda: calls.append(1) or quota()
    home.gpu_seconds = 0
    client.submit(spec.model_copy(update={"gpu": True}))
    worker = client.worker()
    for _ in range(5):
        worker.tick()
    assert len(calls) == 1


def test_suggestions_see_slots_taken_earlier_in_the_same_cycle(two_accounts):
    client, home, other, spec = two_accounts
    for account in client.config.accounts:
        account.cpu_limit = 1
    home.external = {"tester/own-notebook": "cpu"}
    first = client.submit(spec)
    client.submit(spec, account="kaggle:other")
    last = client.submit(spec)
    client.worker().tick()
    assert client.get(first.id).suggested_account == "kaggle:other"
    # The job native to kaggle:other took its only slot, so nothing else is suggested there.
    assert len(other.pushes) == 1 and client.get(last.id).suggested_account is None


def test_moving_a_job_to_its_own_account_is_refused(two_accounts):
    client, home, other, spec = two_accounts
    job = client.submit(spec)
    with pytest.raises(ValueError, match="already on kaggle:tester"):
        client.move(job.id, "kaggle:tester")
    assert client.agent().move([job.id], account="kaggle:tester")["moved"] == 0


def test_downloads_for_a_removed_account_wait_until_it_returns(two_accounts):
    client, home, other, spec = two_accounts
    job = client.submit(spec, account="kaggle:other")
    worker = client.worker()
    worker.tick()
    other.remote[client.get(job.id).remote_ref] = dict(state="COMPLETE", error=None)
    other.download_error = RemoteError("Output listing unavailable")
    due(client, job.id)
    worker.tick()
    assert client.get(job.id).download_state == "error" and other.download_calls == 1
    accounts, client.config.accounts = client.config.accounts, client.config.accounts[:1]
    due(client, job.id)
    worker.tick()
    assert other.download_calls == 1
    client.config.accounts, other.download_error = accounts, None
    worker.tick()
    assert client.get(job.id).download_state == "complete"

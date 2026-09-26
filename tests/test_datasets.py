"""Datasets across accounts: attached where readable, copied only when allowed, found by alias."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as Obj

import pytest
import requests

from compute_runner import Account, Client, JobSpec
from compute_runner.models import input_reference
from compute_runner.providers import RemoteError
from compute_runner.providers.kaggle import KaggleProvider
from conftest import FakeProvider, make_config


def launcher(client, job):
    return client.config.state_dir / "jobs" / job.id / "attempt-1" / "workload.py"


def test_input_references_are_told_apart_from_paths():
    assert input_reference("kaggle:alice/data/3") == ("kaggle", "alice/data/3")
    assert input_reference("job:abc/checkpoints") == ("job", "abc/checkpoints")
    assert input_reference("data/kaggle:x") is None and input_reference("kaggle:/abs") is None


def test_aliased_dataset_is_attached_and_found_under_its_alias(setup, tmp_path):
    client, backend, spec = setup
    (tmp_path / "hello.py").write_text("import os\nprint('TRAIN', os.environ['KGR_INPUT_TRAIN'])\n")
    job = client.submit(spec.model_copy(update={"inputs": {"train": Path("kaggle:tester/cifar")}}))
    client.worker().tick()
    assert "tester/cifar/7" in backend.pushes[0]["dataset_sources"]
    mount = tmp_path / "input/datasets/tester/cifar"
    mount.mkdir(parents=True)
    script = launcher(client, job)
    script.write_text(
        script.read_text()
        .replace("/kaggle/working", str(tmp_path / "working"))
        .replace("/kaggle/input", str(tmp_path / "input"))
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"TRAIN {mount}" in result.stdout


def test_invalid_dataset_inputs_are_rejected_at_submission(setup):
    client, _, spec = setup
    with pytest.raises(ValueError, match="Invalid Kaggle dataset reference"):
        client.submit(spec.model_copy(update={"inputs": {"data": Path("kaggle:not a ref")}}))


def test_unreadable_dataset_is_copied_only_when_allowed(two_accounts):
    client, home, other, spec = two_accounts
    home.unreadable = {"other/private"}
    job = client.submit(spec.model_copy(update={"inputs": {"data": Path("kaggle:other/private")}}))
    client.worker().tick()
    blocked = client.get(job.id)
    assert blocked.state == "blocked" and "kaggle:other can" in blocked.error and "--transfer" in blocked.error
    assert home.pushes == [] and other.fetched == []

    moved = client.agent().move([job.id], account="kaggle:tester", transfer=True)
    assert moved["moved"] == 1 and moved["jobs"][0]["state"] == "queued"
    client.worker().tick()
    assert other.fetched == ["other/private/7"]
    job = client.get(job.id)
    copy = job.transfers["data"]
    assert copy["source"] == "kaggle:other/private/7" and job.state == "remote_queued"
    assert f"tester/kgr-b-{copy['digest'][:40]}/1" in home.pushes[0]["dataset_sources"]
    # The launch verifies the copy against its digest, like any bundle.
    assert copy["digest"] in launcher(client, job).read_text()

    # Allowed for every job, the same dataset version is copied once.
    client.config.transfer = True
    second = client.submit(spec.model_copy(update={"name": "b", "inputs": {"data": Path("kaggle:other/private")}}))
    client.worker().tick()
    assert other.fetched == ["other/private/7"] and client.get(second.id).state == "remote_queued"


@pytest.mark.parametrize("update", [{"inputs": {"data": Path("kaggle:nobody/data")}}, {"datasets": ["nobody/data/2"]}])
def test_dataset_no_account_can_find_blocks_before_launch(two_accounts, update):
    client, home, other, spec = two_accounts
    home.unreadable = other.unreadable = {"nobody/data"}
    job = client.submit(spec.model_copy(update=update))
    client.worker().tick()
    summary = client.agent().status([job.id])["jobs"][0]
    assert summary["state"] == "blocked" and home.pushes == [] and other.fetched == []
    assert "No connected account can find dataset nobody/data" in summary["error"]
    assert summary["reason"].startswith("Check the dataset reference")
    # Once an account may read it, a retry runs.
    home.unreadable = set()
    retried = client.retry(job.id)
    client.worker().tick()
    assert client.get(retried.id).state == "remote_queued"


def failing(error):
    def resolve_dataset(ref):
        raise error

    return resolve_dataset


def test_an_account_that_could_not_be_asked_keeps_the_job_retrying(two_accounts):
    client, home, other, spec = two_accounts
    home.unreadable = other.unreadable = {"nobody/data"}
    other.resolve_dataset = failing(RemoteError("timed out", "transient"))
    job = client.submit(spec.model_copy(update={"inputs": {"data": Path("kaggle:nobody/data")}}))
    client.worker().tick()
    job = client.get(job.id)
    assert job.state == "preparing" and job.wait_reason == "Upload retry pending"
    assert "Could not check whether kaggle:other can read" in job.error and home.pushes == []
    # Bad keys are a definitive answer: that account cannot read it.
    other.resolve_dataset = failing(RemoteError("401", "auth", definitive=True))
    client.store.update(job.id, next_action_at=0)
    client.worker().tick()
    assert client.get(job.id).state == "blocked"


@pytest.mark.parametrize("definitive", [True, False])
def test_an_account_whose_credentials_do_not_work_cannot_read(two_accounts, definitive):
    # Local failures (a missing credentials file, keys for another user) are not definitive.
    client, home, other, spec = two_accounts
    home.unreadable = {"nobody/data"}
    other.resolve_dataset = failing(RemoteError("authentication unavailable", "auth", definitive=definitive))
    job = client.submit(spec.model_copy(update={"inputs": {"data": Path("kaggle:nobody/data")}}))
    client.worker().tick()
    job = client.get(job.id)
    assert job.state == "blocked" and "No connected account can find" in job.error


def test_unaliased_datasets_are_never_copied(two_accounts):
    client, home, other, spec = two_accounts
    client.config.transfer = True
    home.unreadable = {"other/private"}
    job = client.submit(spec.model_copy(update={"datasets": ["other/private"]}))
    client.worker().tick()
    job = client.get(job.id)
    assert job.state == "blocked" and "alias" in job.error and other.fetched == []


def test_failover_offers_an_account_that_needs_a_copy(two_accounts):
    client, home, other, spec = two_accounts
    client.config.failover = "ask"
    home.gpu_seconds = 0
    other.unreadable = {"tester/private"}
    job = client.submit(spec.model_copy(update={"gpu": True, "inputs": {"data": Path("kaggle:tester/private")}}))
    client.worker().tick()
    summary = client.agent().status([job.id])["jobs"][0]
    assert summary["suggested_account"] == "kaggle:other" and summary["suggested_transfer"] is True

    client.config.failover = "auto"
    client.worker().tick()
    assert client.get(job.id).account == "kaggle:tester" and other.pushes == []

    client.config.transfer = True
    client.worker().tick()
    job = client.get(job.id)
    assert job.account == "kaggle:other" and job.state == "remote_queued"
    assert home.fetched == ["tester/private/7"] and len(other.pushes) == 1


def test_cancelling_clears_a_suggested_copy(two_accounts):
    client, home, other, spec = two_accounts
    client.config.failover = "ask"
    home.gpu_seconds = 0
    other.unreadable = {"tester/private"}
    job = client.submit(spec.model_copy(update={"gpu": True, "inputs": {"data": Path("kaggle:tester/private")}}))
    client.worker().tick()
    assert client.get(job.id).suggested_transfer
    client.cancel(job.id)
    summary = client.agent().status([job.id])["jobs"][0]
    assert "suggested_account" not in summary and "suggested_transfer" not in summary


def test_failover_prefers_an_account_that_can_read_the_data(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    config = make_config(tmp_path, "tester", "second", "third", failover="ask")
    backends = {user: FakeProvider(config.state_dir, user) for user in ("tester", "second", "third")}
    client = Client(config=config, providers={f"kaggle:{user}": b for user, b in backends.items()})
    backends["tester"].gpu_seconds = 0
    backends["second"].unreadable = {"tester/private"}
    script = tmp_path / "train.py"
    script.write_text("print(1)\n")
    job = client.submit(JobSpec(source=script, gpu=True, inputs={"data": Path("kaggle:tester/private")}))
    client.worker().tick()
    job = client.get(job.id)
    assert job.suggested_account == "kaggle:third" and not job.suggested_transfer


def response(code):
    value = requests.Response()
    value.status_code = code
    value._content = b"{}"
    return value


def test_kaggle_checks_access_even_for_pinned_datasets(tmp_path):
    calls = []

    def status(ref, format=None):
        calls.append(ref)
        owner = ref.split("/")[0]
        if owner in {"them", "gone"}:
            raise requests.HTTPError("no", response=response(403 if owner == "them" else 404))
        if owner == "down":
            raise requests.HTTPError("unavailable", response=response(503))
        return json.dumps({"current_version_number": 4})

    backend = KaggleProvider(Account(user="tester"), tmp_path)
    backend._api = Obj(dataset_status=status)
    assert backend.resolve_dataset("me/data/2") == "me/data/2" and calls == ["me/data"]
    assert backend.resolve_dataset("me/data") == "me/data/4"
    # A version past the latest does not exist.
    assert backend.resolve_dataset("me/data/4") == "me/data/4" and backend.resolve_dataset("me/data/5") is None
    assert backend.resolve_dataset("them/data/1") is None and backend.resolve_dataset("gone/data") is None
    with pytest.raises(RemoteError) as error:
        backend.resolve_dataset("down/data")
    assert error.value.kind == "transient"


def test_mounts_are_found_whatever_the_owners_casing(tmp_path):
    # Kaggle mounts /kaggle/input/datasets/OWNER/SLUG in lowercase; account names keep their casing.
    import zipfile

    from compute_runner.bundle import snapshot_bundle
    from compute_runner.runtime import _find_bundle, _find_dataset

    inputs = tmp_path / "input"
    mount = inputs / "datasets/dimpap99/data"
    mount.mkdir(parents=True)
    assert _find_dataset("DimPap99/Data/3", input_root=inputs) == mount

    (tmp_path / "src").mkdir()
    (tmp_path / "src/train.py").write_text("print(1)\n")
    bundle = snapshot_bundle(tmp_path / "src", ["train.py"], tmp_path / "bundles")
    unzipped = inputs / f"datasets/dimpap99/kgr-b-{bundle['digest'][:40]}"
    with zipfile.ZipFile(tmp_path / "bundles" / bundle["digest"] / "payload.zip") as archive:
        archive.extractall(unzipped)
    kind, location, _ = _find_bundle(f"DimPap99/kgr-b-{bundle['digest'][:40]}/1", bundle["digest"], input_root=inputs)
    assert (kind, location) == ("directory", unzipped)

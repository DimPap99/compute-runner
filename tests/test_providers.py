"""What every provider shares, and the Kaggle adapter's diagnostics and cleanup hooks."""

import json
import sys
import threading
from datetime import datetime
from types import SimpleNamespace as Obj

import pytest
import requests

from compute_runner.providers import Artifact, Provider
from compute_runner.providers.kaggle.client import quiet


def test_quiet_restores_stdout_when_threads_overlap():
    original = sys.stdout
    start = threading.Barrier(4)

    def authenticate_repeatedly():
        start.wait()
        for _ in range(50):
            with quiet():
                print("Kaggle banner")

    threads = [threading.Thread(target=authenticate_repeatedly) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sys.stdout is original


def project_job(client, spec, tmp_path, **update):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "main.py").write_text("print('hi')\n")
    return client.submit(spec.model_copy(update={"source": project, "entrypoint": "main.py", **update}))


def test_small_projects_travel_inside_the_kaggle_notebook_but_are_bundles_elsewhere(setup, tmp_path):
    client, backend, spec = setup
    data = tmp_path / "data.txt"
    data.write_text("input")
    job = project_job(client, spec, tmp_path, inputs={"data": data})
    assert list(backend.bundles_for(job)) == ["input:data"]
    assert list(Provider.bundles_for(backend, job)) == ["input:data", "source"]


def test_input_status_reports_each_dataset_and_falls_back_to_the_owned_listing(setup, tmp_path):
    client, backend, spec = setup
    data = tmp_path / "data.txt"
    data.write_text("input")
    job = client.submit(spec.model_copy(update={"inputs": {"data": data, "weights": "kaggle:owner/weights"}}))
    bundle_ref = f"tester/kgr-b-{job.snapshot['inputs']['data']['digest'][:40]}"

    def dataset_status(ref, format=None):
        if ref == bundle_ref:
            raise requests.HTTPError("403 Forbidden")
        return "ready" if format is None else json.dumps({"current_version_number": 3})

    listed = Obj(ref=bundle_ref, id=7, title="kgr b", last_updated=None, is_private=True, total_bytes=5)
    backend._api = Obj(
        dataset_status=dataset_status, dataset_list=lambda mine, page: [listed] if page == 1 else []
    )
    rows = {row["alias"]: row for row in backend.input_status(job)}
    assert rows["weights"] == {
        "alias": "weights",
        "ref": "owner/weights",
        "status": "ready",
        "version": {"current_version_number": 3},
    }
    assert rows["data"]["bundle_bytes"] == 5 and "403" in rows["data"]["status_error"]
    assert rows["data"]["inventory"]["ref"] == bundle_ref


class Context:
    """A context manager returning value, as Kaggle's build_kaggle_client does."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_):
        return False


def kaggle_account_with(backend, deleted):
    """Two launch notebooks and datasets on the account, one of each made by this runner."""
    ran = datetime(2026, 9, 1)
    kernels = [
        Obj(ref="tester/kgr-train-0123456789ab-a1", last_run_time=ran),
        Obj(ref="tester/my-own-notebook", last_run_time=ran),
        None,
    ]
    datasets = [Obj(ref="tester/kgr-b-" + "d" * 40, total_bytes=10, last_updated=ran), Obj(ref="tester/mine")]

    def delete_dataset(request):
        deleted.append(f"{request.owner_slug}/{request.dataset_slug}")
        return Obj(error="")

    client = Obj(datasets=Obj(dataset_api_client=Obj(delete_dataset=delete_dataset)))
    backend._api = Obj(
        kernels_list_with_response=lambda **options: Obj(kernels=kernels, next_page_token=None),
        dataset_list=lambda mine, search, page: datasets if page == 1 else [],
        build_kaggle_client=lambda: Context(client),
    )
    backend._kernels = lambda method, request, ref=None: deleted.append(ref) or Obj(error_message="")


def test_kaggle_artifacts_are_only_this_runners_and_only_they_can_be_deleted(setup):
    client, backend, spec = setup
    deleted = []
    kaggle_account_with(backend, deleted)
    found = backend.artifacts()
    assert [(item.kind, item.name) for item in found] == [
        ("notebook", "tester/kgr-train-0123456789ab-a1"),
        ("dataset", "tester/kgr-b-" + "d" * 40),
    ]
    assert found[0].attempt == found[0].name and found[1].digest == "d" * 40 and found[1].bytes == 10
    for artifact in found:
        backend.delete_artifact(artifact)
    assert deleted == [found[0].name, found[1].name]
    for foreign in (Artifact("notebook", "tester/my-own-notebook"), Artifact("dataset", "tester/mine")):
        with pytest.raises(ValueError, match="Refusing to delete"):
            backend.delete_artifact(foreign)

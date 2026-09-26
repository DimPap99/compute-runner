import json
from pathlib import Path

import pytest

from kaggle_runner import Client, Config, JobSpec
from kaggle_runner.backend import RemoteError


class FakeBackend:
    def __init__(self):
        self.remote = {}
        self.pushes = []
        self.external = {}
        self.uploads = []
        self.upload_ready = True
        self.gpu_seconds = 100000
        self.push_error = None
        self.on_push = None
        self.download_error = None
        self.download_calls = 0
        self.discovery_error = None

    def active_runs(self):
        if self.discovery_error:
            raise self.discovery_error
        return dict(self.external)

    def quota(self):
        return {"gpu": {"available_seconds": self.gpu_seconds}, "refresh_at": None}

    def ensure_bundle(self, bundle):
        self.uploads.append(bundle["digest"])
        return f"tester/kgr-b-{bundle['digest'][:40]}/1" if self.upload_ready else None

    def resolve_dataset(self, ref):
        return ref if len(ref.split("/")) == 3 else ref + "/7"

    def push(self, folder, **options):
        metadata = json.loads((Path(folder) / "kernel-metadata.json").read_text())
        self.pushes.append(metadata | options)
        if self.push_error:
            raise self.push_error
        self.remote[metadata["id"]] = dict(state="QUEUED", error=None)
        if self.on_push:
            self.on_push()
        return dict(ref=metadata["id"], version=1)

    def status(self, ref):
        if ref not in self.remote:
            raise RemoteError("not found", "missing", definitive=True)
        return self.remote[ref]

    def download(self, ref, destination, patterns):
        self.download_calls += 1
        if self.download_error:
            raise self.download_error
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "run.log").write_text("remote completed\n")
        return {}

    def logs(self, ref, follow=False):
        yield "example log\n"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    backend = FakeBackend()
    client = Client(
        config=Config(
            owner="tester", state_dir=tmp_path / "state", poll_seconds=1, retry_seconds=1, reconcile_seconds=1
        ),
        backend=backend,
    )
    script = tmp_path / "hello.py"
    script.write_text("print('hello')\n")
    return client, backend, JobSpec(source=script)


def due(client, job_id):
    client.store.update(job_id, next_action_at=0, download_retry_at=0)

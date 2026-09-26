from types import SimpleNamespace as Obj

import pytest
import requests

from kaggle_runner.backend import KaggleBackend, RemoteError, download_outputs, remote_error


class Response:
    def __init__(self, data=b"hello", fail=False):
        self.data, self.fail = data, fail
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        yield self.data[:2]
        if self.fail:
            raise requests.ConnectionError("interrupted")
        yield self.data[2:]


def page(*names, log="log content"):
    return Obj(files=[Obj(file_name=n, url="https://example.test/" + n) for n in names], log=log)


def test_download_pages_filters_and_verified_reuse(tmp_path):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    pages = [page("one.txt", "skip.png"), page("nested/two.txt")]
    receipts = download_outputs(pages, tmp_path, ["*.txt"], get=get)
    assert set(receipts) == {"one.txt", "nested/two.txt"}
    assert (tmp_path / "run.log").read_text() == "log content"
    assert len(calls) == 2
    download_outputs(pages, tmp_path, ["*.txt"], get=get)
    assert len(calls) == 2


def test_interrupted_download_never_replaces_existing_output(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "one.txt").write_text("old version")
    with pytest.raises(requests.ConnectionError):
        download_outputs([page("one.txt")], tmp_path, get=lambda *a, **k: Response(fail=True))
    assert (root / "one.txt").read_text() == "old version"
    assert not list(root.glob(".kgr-*"))


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "path/../../escape", "bad\\escape"])
def test_download_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError):
        download_outputs([page(name)], tmp_path, get=lambda *a, **k: Response())


def test_download_rejects_symlink_escape(tmp_path):
    root = tmp_path / "result/outputs"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        download_outputs([page("link/file")], tmp_path / "result", get=lambda *a, **k: Response())


def test_push_error_body_is_not_success(tmp_path):
    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        kernels_push=lambda *a, **k: Obj(error="Maximum batch CPU session count of 5 reached", kernel_id=0)
    )
    with pytest.raises(RemoteError) as error:
        backend.push(tmp_path, timeout_seconds=60)
    assert error.value.definitive and error.value.kind == "capacity"


def test_server_failure_is_uncertain_for_mutations():
    response = requests.Response()
    response.status_code = 503
    response._content = b"{}"
    error = remote_error(requests.HTTPError("unavailable", response=response), mutation=True)
    assert error.kind == "uncertain" and not error.definitive


def test_quota_accounts_for_reservations(tmp_path):
    from datetime import timedelta

    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        quota_view=lambda: Obj(
            quota_refresh_time=None,
            tpu_quota=None,
            gpu_quota=Obj(
                time_used=timedelta(hours=2),
                time_reserved=timedelta(hours=3),
                total_time_allowed=timedelta(hours=10),
            ),
        )
    )
    assert backend.quota()["gpu"]["available_seconds"] == 5 * 3600


def test_push_normalizes_versioned_reference(tmp_path):
    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        kernels_push=lambda *a, **k: Obj(
            error=None,
            kernel_id=1,
            ref="tester/job/3",
            version_number=3,
            url="https://www.kaggle.com/code/tester/job",
        ),
        parse_kernel_string=lambda ref: ("tester", "job", "3"),
    )
    assert backend.push(tmp_path, timeout_seconds=60) == {"ref": "tester/job", "version": 3}


def test_dataset_missing_403_reconciles_owned_inventory(tmp_path):
    digest = "a" * 64
    archive = tmp_path / "bundles" / digest / "payload.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"example")
    response = requests.Response()
    response.status_code = 403
    response._content = b"{}"

    def missing(*args, **kwargs):
        raise requests.HTTPError("Permission denied", response=response)

    creates = []
    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        dataset_status=missing,
        dataset_list=lambda **kwargs: [],
        dataset_create_new=lambda *args, **kwargs: creates.append(kwargs) or Obj(error=None),
    )
    assert backend.ensure_bundle({"digest": digest}) is None
    assert len(creates) == 1 and creates[0]["public"] is False


def test_existing_dataset_403_does_not_trigger_creation(tmp_path):
    digest = "b" * 64
    response = requests.Response()
    response.status_code = 403
    response._content = b"{}"

    def forbidden(*args, **kwargs):
        raise requests.HTTPError("Permission denied", response=response)

    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        dataset_status=forbidden, dataset_list=lambda **k: [Obj(ref=f"tester/kgr-b-{digest[:40]}")]
    )
    with pytest.raises(RemoteError) as error:
        backend.ensure_bundle({"digest": digest})
    assert error.value.kind == "auth"


def test_push_accepts_bare_slug_and_saves_receipt(tmp_path):
    import json

    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        kernels_push=lambda *a, **k: Obj(
            error=None, kernel_id=9, ref="job", version_number=1, url="https://www.kaggle.com/code/tester/job"
        ),
        parse_kernel_string=lambda ref: (*ref.split("/"), None),
    )
    assert backend.push(tmp_path, timeout_seconds=60)["ref"] == "tester/job"
    assert json.loads((tmp_path / "push-receipt.json").read_text())["kernel_id"] == 9


def test_persisted_log_events_are_readable():
    from kaggle_runner.backend import render_log

    assert render_log('[{"stream_name":"stdout","data":"hello\\n"}]') == "hello\n"
    assert render_log("plain text") == "plain text"


@pytest.mark.parametrize(
    "ref", ["/code/tester/job", "https://www.kaggle.com/code/tester/job", "tester/job", "job"]
)
def test_live_save_response_reference_forms(tmp_path, ref):
    backend = KaggleBackend("tester", tmp_path)
    backend._api = Obj(
        kernels_push=lambda *a, **k: Obj(
            error=None, kernel_id=135, ref=ref, version_number=1, url="https://www.kaggle.com/code/tester/job"
        ),
        parse_kernel_string=lambda ref: (*ref.split("/"), None),
    )
    assert backend.push(tmp_path, timeout_seconds=180) == {"ref": "tester/job", "version": 1}


def test_http_quota_error_remains_retryable_not_auth_failure():
    response = requests.Response()
    response.status_code = 403
    response._content = b'{"message":"GPU quota exhausted"}'
    error = remote_error(requests.HTTPError("Forbidden", response=response), mutation=True)
    assert error.kind == "quota" and error.definitive
    assert "quota exhausted" in str(error)


@pytest.mark.parametrize(
    "code,message,kind,definitive",
    [
        (401, "Unauthenticated", "auth", True),
        (403, "Permission denied", "auth", True),
        (404, "Not found", "missing", True),
        (429, "Too many requests", "rate_limit", True),
        (429, "Maximum session limit reached", "capacity", True),
        (400, "Invalid slug", "invalid", True),
        (400, "Dataset storage limit exceeded", "storage", True),
        (408, "Timeout", "transient", False),
        (500, "Server error", "transient", False),
    ],
)
def test_remote_error_mapping(code, message, kind, definitive):
    response = requests.Response()
    response.status_code = code
    response._content = ('{"message": "%s"}' % message).encode()
    error = remote_error(requests.HTTPError(message, response=response))
    assert (error.kind, error.definitive) == (kind, definitive)

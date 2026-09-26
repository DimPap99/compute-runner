"""Version-isolated Kaggle adapter with explicit, operation-aware failures."""

from __future__ import annotations

import contextvars
import fnmatch
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import urllib3
from requests.adapters import HTTPAdapter

from .store import atomic_json


class RemoteError(RuntimeError):
    def __init__(self, message, kind="transient", *, definitive=False):
        super().__init__(message)
        self.kind = kind
        self.definitive = definitive


def classify(message):
    lower = message.lower()
    if "session" in lower and any(word in lower for word in ("maximum", "limit", "cap reached")):
        return "capacity"
    if any(word in lower for word in ("quota", "accelerator time", "gpu hours")):
        return "quota"
    if any(word in lower for word in ("storage", "disk space", "dataset limit")):
        return "storage"
    return "invalid"


def http_code(error):
    response = getattr(error, "response", None)
    if response is None:
        return None
    code = response.status_code
    try:
        body_code = response.json().get("code", 0)
        if isinstance(body_code, int) and body_code >= 400:
            code = body_code
    except (ValueError, AttributeError):
        pass
    return code


def safe_message(error):
    # Signed download URLs and bearer credentials must not enter state or logs.
    text = str(error)
    response = getattr(error, "response", None)
    if response is not None:
        try:
            body = response.json()
            detail = body.get("message") or body.get("error")
            if isinstance(detail, str):
                text = f"HTTP {http_code(error)}: {detail}"
        except (ValueError, AttributeError):
            pass
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", text)
    text = re.sub(r"(?i)(bearer\s+|KGAT_)[A-Za-z0-9._~+/=-]+", "[redacted]", text)
    return text[:2000]


def remote_error(error, *, mutation=False):
    if isinstance(error, RemoteError):
        return error
    message = safe_message(error)
    code = http_code(error)
    kind = classify(message)
    if code and 400 <= code < 500 and code != 408 and kind in {"capacity", "quota", "storage"}:
        return RemoteError(message, kind, definitive=True)
    if code in {401, 403}:
        return RemoteError(message, "auth", definitive=True)
    if code == 404:
        return RemoteError(message, "missing", definitive=True)
    if code == 429:
        return RemoteError(
            message, "capacity" if classify(message) == "capacity" else "rate_limit", definitive=True
        )
    if code and 400 <= code < 500 and code != 408:
        return RemoteError(message, classify(message), definitive=True)
    return RemoteError(message, "uncertain" if mutation else "transient")


# Longer than Kaggle's 12-hour session limit plus queueing; older runs cannot still be active.
ACTIVE_HORIZON = timedelta(hours=24)

# Read timeout for calls without an explicit timeout; lowered while snapshotting a live log stream.
READ_TIMEOUT = contextvars.ContextVar("kgr_read_timeout", default=90)


class TimeoutAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (15, READ_TIMEOUT.get())
        return super().send(request, **kwargs)


class KaggleBackend:
    def __init__(self, owner: str, state_dir: Path):
        self.owner = owner
        self.state_dir = state_dir
        self._api = None

    @property
    def api(self):
        if self._api is None:
            # Kaggle's package has eager authentication. Keep the import out of public model/API imports.
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi

                class BoundedApi(KaggleApi):
                    def build_kaggle_client(self):
                        client = super().build_kaggle_client()
                        session = client._http_client
                        session._init_session()
                        session._session.mount("https://", TimeoutAdapter(max_retries=0))
                        return client

                self._api = BoundedApi()
                self._api.authenticate()
            except (SystemExit, Exception) as error:
                self._api = None
                raise RemoteError(
                    "Kaggle authentication unavailable; configure standard Kaggle credentials", "auth"
                ) from error
        return self._api

    def resolve_dataset(self, ref):
        if len(ref.split("/")) == 3:
            return ref
        try:
            result = json.loads(self.api.dataset_status(ref, format="json(current_version_number)"))
            return ref + "/" + str(result["current_version_number"])
        except Exception as error:
            raise remote_error(error) from error

    def _owned_dataset_exists(self, ref):
        # A 403 on GetDatasetStatus can conceal nonexistence. Only a successful
        # authenticated inventory of our own datasets allows creation in that case.
        page = 1
        while True:
            rows = self.api.dataset_list(mine=True, page=page)
            if not rows:
                return False
            if any(row is not None and (row.ref or "").lower() == ref.lower() for row in rows):
                return True
            page += 1

    def ensure_bundle(self, bundle) -> str | None:
        digest = bundle["digest"]
        ref = f"{self.owner}/kgr-b-{digest[:40]}"
        try:
            status = self.api.dataset_status(ref).lower()
        except Exception as error:
            converted = remote_error(error)
            if converted.kind == "auth":
                if http_code(error) != 403 or self._owned_dataset_exists(ref):
                    raise converted from error
            elif converted.kind != "missing":
                raise converted from error
            folder = self.state_dir / "uploads" / digest
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            archive = folder / "payload.zip"
            if not archive.exists():
                os.link(self.state_dir / "bundles" / digest / "payload.zip", archive)
            atomic_json(
                folder / "dataset-metadata.json",
                dict(
                    id=ref,
                    title="kgr b " + digest[:40],
                    licenses=[{"name": "other"}],
                    description="Private workload snapshot. Original copyright and license terms in the included "
                    "files apply; no additional permissions are granted. SHA256: " + digest,
                ),
            )
            try:
                response = self.api.dataset_create_new(
                    str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip"
                )
                if response.error:
                    raise RemoteError(response.error, classify(response.error), definitive=True)
            except Exception as create_error:
                # Deterministic dataset identity is reconciled on the next tick, never versioned here.
                raise remote_error(create_error) from create_error
            return None
        if status == "ready":
            return ref + "/1"
        if any(word in status for word in ("error", "fail", "deleted")):
            raise RemoteError(f"Dataset {ref}: {status}", "invalid", definitive=True)
        return None

    def push(self, folder, *, timeout_seconds, accelerator=None):
        try:
            response = self.api.kernels_push(str(folder), timeout=str(timeout_seconds), acc=accelerator)
        except Exception as error:
            raise remote_error(error, mutation=True) from error
        if response.error:
            raise RemoteError(
                response.error, classify(response.error), definitive=not bool(response.kernel_id)
            )
        if not response.ref or not response.kernel_id:
            raise RemoteError("Kaggle returned no accepted kernel identity", "uncertain")
        atomic_json(
            Path(folder) / "push-receipt.json",
            {
                "ref": response.ref,
                "url": response.url.split("?")[0] if response.url else None,
                "kernel_id": response.kernel_id,
                "version": response.version_number,
            },
        )
        # The live service returns /code/owner/slug; also accept documented
        # owner/slug/version, bare slug and absolute Kaggle URL forms.
        from urllib.parse import urlsplit

        parsed = urlsplit(response.ref)
        if parsed.netloc and parsed.hostname not in {"kaggle.com", "www.kaggle.com"}:
            raise RemoteError("Unexpected host in returned notebook reference", "uncertain")
        ref = parsed.path.removeprefix("/code/").strip("/")
        if "/" not in ref:
            ref = f"{self.owner}/{ref}"
        owner, slug, version = self.api.parse_kernel_string(ref)
        return dict(
            ref=f"{owner}/{slug}", version=response.version_number or (int(version) if version else None)
        )

    def status(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

        request = ApiGetKernelSessionStatusRequest()
        request.user_name, request.kernel_slug = ref.split("/")[:2]
        try:
            with self.api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.get_kernel_session_status(request)
            return dict(state=response.status.name, error=response.failure_message)
        except Exception as error:
            raise remote_error(error) from error

    def active_runs(self):
        """Best-effort account inventory; no metadata or source is written locally.

        One status call per kernel quickly hits Kaggle's rate limit on accounts with many
        notebooks. The listing is newest run first and a session lasts at most 12 hours,
        so scanning stops at the first kernel whose last run is older than ACTIVE_HORIZON.
        """
        result = {}
        token = None
        seen = set()
        cutoff = datetime.now(timezone.utc) - ACTIVE_HORIZON
        try:
            while True:
                response = self.api.kernels_list_with_response(
                    mine=True, page_size=100, page_token=token, sort_by="dateRun"
                )
                for kernel in response.kernels or []:
                    if kernel is None:
                        continue
                    last_run = kernel.last_run_time
                    if last_run is not None:
                        # The service returns naive UTC timestamps.
                        if last_run.tzinfo is None:
                            last_run = last_run.replace(tzinfo=timezone.utc)
                        if last_run < cutoff:
                            return result
                    ref = kernel.ref
                    if not ref or ref.split("/")[0].lower() != self.owner.lower():
                        continue
                    try:
                        status = self.status(ref)
                    except RemoteError as error:
                        if error.kind == "missing":
                            continue
                        raise
                    if status["state"] in {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}:
                        result[ref] = (
                            "unknown"
                            if kernel.enable_gpu is None
                            else ("gpu" if kernel.enable_gpu else "cpu")
                        )
                token = response.next_page_token
                if not token:
                    return result
                if token in seen:
                    raise RemoteError("Repeated account pagination token")
                seen.add(token)
        except Exception as error:
            raise remote_error(error) from error

    def quota(self):
        try:
            response = self.api.quota_view()
            result = {
                "refresh_at": response.quota_refresh_time.timestamp() if response.quota_refresh_time else None
            }
            for name in ("gpu", "tpu"):
                quota = getattr(response, name + "_quota")
                if quota is None:
                    result[name] = None
                    continue

                def seconds(value):
                    return value.total_seconds() if value is not None else 0

                used = seconds(quota.time_used)
                reserved = seconds(quota.time_reserved)
                total = seconds(quota.total_time_allowed)
                result[name] = dict(
                    used_seconds=used,
                    reserved_seconds=reserved,
                    total_seconds=total,
                    available_seconds=max(0, total - used - reserved),
                )
            return result
        except Exception as error:
            raise remote_error(error) from error

    def logs(self, ref, *, follow=False):
        try:
            if not follow:
                yield render_log(self.api.kernels_logs(ref))
                return
            seen = 0
            failures = 0
            while failures < 5:
                before = seen
                try:
                    for index, event in enumerate(self.api.kernels_logs_stream(ref)):
                        if index < seen:
                            continue
                        seen = index + 1
                        if event.get("data") is not None:
                            yield event["data"]
                    return
                except requests.RequestException:
                    failures = failures + 1 if seen == before else 0
                    if failures < 5:
                        time.sleep(min(2**failures, 15))
            raise RemoteError(
                "Log stream disconnected repeatedly; retry logs --follow or open the Kaggle URL"
            )
        except Exception as error:
            raise remote_error(error) from error

    def live_log(self, ref, *, idle_seconds=5, max_seconds=20):
        """Bounded snapshot of a session's log, including one that is still running.

        Persisted logs appear only after a session ends. The stream endpoint replays the
        log from the start, so read until it goes idle, ends, or max_seconds elapse.
        """
        chunks = []
        deadline = time.monotonic() + max_seconds
        token = READ_TIMEOUT.set(idle_seconds)
        try:
            stream = self.api.kernels_logs_stream(ref)
            try:
                for event in stream:
                    if event.get("data") is not None:
                        chunks.append(str(event["data"]))
                    if time.monotonic() >= deadline:
                        break
            finally:
                stream.close()
        except requests.RequestException as error:
            # An idle stream ends the snapshot; other failures before any output are errors.
            if not chunks and not _read_timeout(error):
                raise remote_error(error) from error
        except Exception as error:
            raise remote_error(error) from error
        finally:
            READ_TIMEOUT.reset(token)
        return "".join(chunks)

    def output_pages(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        token = None
        seen = set()
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = ref.split("/")[:2]
            request.page_size = 100
            if token:
                request.page_token = token
            try:
                with self.api.build_kaggle_client() as client:
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            except Exception as error:
                raise remote_error(error) from error
            yield response
            token = response.next_page_token
            if not token:
                return
            if token in seen:
                raise RemoteError("Repeated output pagination token")
            seen.add(token)

    def download(self, ref, destination: Path, patterns=None, *, skip=None):
        return download_outputs(self.output_pages(ref), destination, patterns, skip=skip)


def _read_timeout(error):
    return isinstance(error, requests.ReadTimeout) or any(
        isinstance(arg, urllib3.exceptions.ReadTimeoutError) for arg in error.args
    )


def render_log(raw):
    """Kaggle persisted logs may be a JSON array of stream events."""
    try:
        events = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if isinstance(events, list) and all(isinstance(event, dict) and "data" in event for event in events):
        return "".join(str(event["data"]) for event in events)
    return raw


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_outputs(pages, destination, patterns=None, *, skip=None, get=requests.get):
    """Download session outputs; names matching the skip predicate are not fetched."""
    from .bundle import safe_relative

    destination = Path(destination)
    root = destination / "outputs"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path = destination / "downloads.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for page in pages:
        if page.log:
            # Always retrieve logs even when output filtering selects no files.
            log = destination / "run.log"
            temporary = log.with_suffix(".tmp")
            temporary.write_text(render_log(page.log), encoding="utf-8")
            os.replace(temporary, log)
        for item in page.files or []:
            name = safe_relative(item.file_name)
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Output resolves outside the destination")
            if skip is not None and skip(name):
                continue
            if patterns is not None and not any(fnmatch.fnmatchcase(name, p) for p in patterns):
                continue
            if name in receipts and target.is_file() and sha256(target) == receipts[name]["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with get(item.url, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    size = 0
                    digest = hashlib.sha256()
                    with tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix=".kgr-", delete=False
                    ) as output:
                        temporary = Path(output.name)
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    expected = response.headers.get("Content-Length")
                    if expected and not response.headers.get("Content-Encoding") and size != int(expected):
                        raise IOError("Incomplete output download")
                os.replace(temporary, target)
                receipts[name] = dict(bytes=size, sha256=digest.hexdigest())
                atomic_json(receipt_path, receipts)
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
    return receipts

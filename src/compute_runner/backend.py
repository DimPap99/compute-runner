"""Version-isolated Kaggle adapter with explicit, operation-aware failures."""

from __future__ import annotations

import contextvars
import fnmatch
import hashlib
import ipaddress
import json
import os
import re
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
import urllib3
from requests.adapters import HTTPAdapter

from .runtime import file_digest, safe_relative
from .security import redact_secrets
from .store import atomic_json, atomic_write


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
    try:
        detail = error.errors(include_input=False, include_url=False)[0]
        text = f"Invalid {'.'.join(map(str, detail['loc']))}: {detail['msg']}"
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    response = getattr(error, "response", None)
    if response is not None:
        try:
            body = response.json()
            detail = body.get("message") or body.get("error")
            if isinstance(detail, str):
                text = f"HTTP {http_code(error)}: {detail}"
        except (ValueError, AttributeError):
            pass
    return redact_secrets(text, strict=True)[:2000]


def remote_error(error, *, mutation=False):
    if isinstance(error, RemoteError):
        return error
    message, code = safe_message(error), http_code(error)
    if not code or not 400 <= code < 500 or code == 408:
        return RemoteError(message, "uncertain" if mutation else "transient")
    # Resource limits win over the status code: a 403 quota error is retryable, not an auth failure.
    kind = classify(message)
    if kind not in {"capacity", "quota", "storage"}:
        kind = {401: "auth", 403: "auth", 404: "missing", 429: "rate_limit"}.get(code, kind)
    return RemoteError(message, kind, definitive=True)


def paginate(fetch):
    """Yield fetch(token) pages until the service returns no next token."""
    token, seen = None, set()
    while True:
        page = fetch(token)
        yield page
        token = page.next_page_token
        if not token:
            return
        if token in seen:
            raise RemoteError("Repeated pagination token")
        seen.add(token)


# Longer than Kaggle's 12-hour session limit plus queueing; older runs cannot still be active.
ACTIVE_HORIZON = timedelta(hours=24)
# Remote session states that still occupy an execution slot.
RUNNING_STATES = {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}


def _utc(value):
    """The service returns naive UTC timestamps."""
    return value.replace(tzinfo=value.tzinfo or timezone.utc)

# Read timeout for calls without an explicit timeout; lowered while snapshotting a live log stream.
READ_TIMEOUT = contextvars.ContextVar("kgr_read_timeout", default=90)


class TimeoutAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (15, READ_TIMEOUT.get())
        return super().send(request, **kwargs)


class KaggleBackend:
    def __init__(self, owner: str, state_dir: Path, *, strict=False):
        self.owner = owner
        self.state_dir = state_dir
        self.strict = strict
        self._api = None
        self._resources = {}

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
        try:
            while rows := self.api.dataset_list(mine=True, page=page):
                if any(row is not None and (row.ref or "").lower() == ref.lower() for row in rows):
                    return True
                page += 1
        except Exception as error:
            raise remote_error(error) from error
        return False

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

    def _kernels(self, method, request, ref=None):
        if ref is not None:
            request.user_name, request.kernel_slug = ref.split("/")[:2]
        try:
            with self.api.build_kaggle_client() as client:
                return getattr(client.kernels.kernels_api_client, method)(request)
        except Exception as error:
            raise remote_error(error) from error

    def status(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

        response = self._kernels("get_kernel_session_status", ApiGetKernelSessionStatusRequest(), ref)
        return dict(state=response.status.name, error=response.failure_message)

    def _resource(self, ref):
        """The account listing reports every notebook as CPU; the notebook itself is accurate."""
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        metadata = self._kernels("get_kernel", ApiGetKernelRequest(), ref).metadata
        if metadata is None or metadata.enable_gpu is None:
            return "unknown"
        return "gpu" if metadata.enable_gpu else "cpu"

    def cancel(self, ref, job_id):
        """Stop a running session. The runtime logs its session ID, which the public API never returns."""
        from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest

        found = re.search(rf"^KGR workload {job_id} session (\d+)$", self.live_log(ref), re.M)
        if not found:
            raise ValueError(
                "No session ID in this run's log yet (it has not started, or predates remote cancellation)"
            )
        request = ApiCancelKernelSessionRequest()
        request.kernel_session_id = int(found.group(1))
        response = self._kernels("cancel_kernel_session", request)
        if response.error_message:
            raise RemoteError(response.error_message, "invalid", definitive=True)

    def active_runs(self):
        """Best-effort account inventory; no metadata or source is written locally.

        One status call per kernel quickly hits Kaggle's rate limit on accounts with many
        notebooks. The listing is newest run first and a session lasts at most 12 hours,
        so scanning stops at the first kernel whose last run is older than ACTIVE_HORIZON.
        """
        result = {}
        # Metadata is read once per run; entries for finished runs are dropped.
        resources, self._resources = self._resources, {}
        cutoff = datetime.now(timezone.utc) - ACTIVE_HORIZON
        pages = paginate(
            lambda token: self.api.kernels_list_with_response(
                mine=True, page_size=100, page_token=token, sort_by="dateRun"
            )
        )
        try:
            for response in pages:
                for kernel in response.kernels or []:
                    if kernel is None:
                        continue
                    last_run = kernel.last_run_time
                    if last_run is not None and _utc(last_run) < cutoff:
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
                    if status["state"] in RUNNING_STATES:
                        key = (ref, last_run)
                        self._resources[key] = resources.get(key) or self._resource(ref)
                        result[ref] = self._resources[key]
            return result
        except Exception as error:
            raise remote_error(error) from error

    def quota(self):
        try:
            response = self.api.quota_view()
            refresh = response.quota_refresh_time
            result = {"refresh_at": _utc(refresh).timestamp() if refresh else None}
            for name in ("gpu", "tpu"):
                quota = getattr(response, name + "_quota")
                if quota is None:
                    result[name] = None
                    continue
                used, reserved, total = (
                    value.total_seconds() if value is not None else 0
                    for value in (quota.time_used, quota.time_reserved, quota.total_time_allowed)
                )
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
                yield redact_secrets(render_log(self.api.kernels_logs(ref)), strict=self.strict)
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
                            yield redact_secrets(event["data"], strict=self.strict)
                    return
                except requests.RequestException as error:
                    # A session that prints nothing for a while is quiet, not disconnected.
                    if _read_timeout(error) and self.status(ref)["state"] in RUNNING_STATES:
                        continue
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
        return redact_secrets("".join(chunks), strict=self.strict)

    def output_pages(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        def fetch(token):
            request = ApiListKernelSessionOutputRequest()
            request.page_size = 100
            if token:
                request.page_token = token
            return self._kernels("list_kernel_session_output", request, ref)

        return paginate(fetch)

    def download(self, ref, destination: Path, patterns=None, *, skip=None):
        return download_outputs(self.output_pages(ref), destination, patterns, skip=skip, strict=self.strict)


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


def _content_length(response):
    raw = response.headers.get("Content-Length")
    if raw is None or response.headers.get("Content-Encoding"):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise OSError("Invalid output download size") from error
    if value < 0:
        raise OSError("Invalid output download size")
    return value


def _checked_chunks(response, digest, expected):
    """Yield the body into digest; fail before the file is replaced if it was cut short."""
    size = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        yield chunk
    if expected is not None and size != expected:
        raise OSError("Incomplete output download")


_REDIRECTS = {301, 302, 303, 307, 308}


def _validated_download_url(url, *, resolve=False):
    """Accept public HTTPS URLs only; signed output URLs need no local credentials."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid output download URL") from error
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Output downloads require HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Output download URLs must not contain credentials")
    if port not in (None, 443):
        raise ValueError("Output downloads require the standard HTTPS port")
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal", ".home.arpa")):
        raise ValueError("Output download URL points to a local host")
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        addresses = []
    if resolve:
        try:
            addresses.extend(
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
            )
        except socket.gaierror as error:
            raise ValueError("Output download host could not be resolved") from error
    if any(not address.is_global for address in addresses):
        raise ValueError("Output download URL points to a non-public address")
    return url


def _open_download(url, get, *, strict=False, resolve=False, max_redirects=5):
    if not strict:
        return get(url, stream=True, timeout=(15, 90))
    for _ in range(max_redirects + 1):
        _validated_download_url(url, resolve=resolve)
        response = get(url, stream=True, timeout=(15, 90), allow_redirects=False)
        if getattr(response, "status_code", 200) not in _REDIRECTS:
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise ValueError("Output download redirect has no destination")
        url = urljoin(url, location)
    raise ValueError("Too many output download redirects")


def download_outputs(pages, destination, patterns=None, *, skip=None, get=None, strict=False):
    """Download session outputs; names matching the skip predicate are not fetched.

    strict ignores proxy, CA and .netrc settings from the environment, and fetches only
    public HTTPS addresses, checking every redirect.
    """
    session = None
    resolve = strict and get is None
    if get is None:
        session = requests.Session()
        session.trust_env = not strict
        get = session.get
    try:
        destination = Path(destination)
        root = destination / "outputs"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt_path = destination / "downloads.json"
        receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        for page in pages:
            if page.log:
                # Always retrieve logs even when output filtering selects no files.
                log_data = redact_secrets(render_log(page.log), strict=strict).encode()
                atomic_write(
                    destination / "run.log",
                    log_data,
                    check_space=True,
                    expected_bytes=len(log_data),
                )
            for item in page.files or []:
                name = safe_relative(item.file_name)
                target = root / name
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Output resolves outside the destination")
                if skip is not None and skip(name):
                    continue
                if patterns is not None and not any(fnmatch.fnmatchcase(name, p) for p in patterns):
                    continue
                if name in receipts and target.is_file() and file_digest(target) == receipts[name]["sha256"]:
                    continue
                digest = hashlib.sha256()
                with _open_download(item.url, get, strict=strict, resolve=resolve) as response:
                    response.raise_for_status()
                    expected = _content_length(response)
                    atomic_write(
                        target,
                        _checked_chunks(response, digest, expected),
                        check_space=True,
                        expected_bytes=expected,
                    )
                receipts[name] = dict(bytes=target.stat().st_size, sha256=digest.hexdigest())
                atomic_json(receipt_path, receipts)
        return receipts
    finally:
        if session is not None:
            session.close()

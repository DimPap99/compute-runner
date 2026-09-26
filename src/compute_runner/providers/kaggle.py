"""Kaggle adapter: private notebooks and datasets, one instance per account."""

from __future__ import annotations

import base64
import contextlib
import contextvars
import io
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import nbformat
import requests
import urllib3
from requests.adapters import HTTPAdapter

from .. import runtime
from ..models import Account, JobRecord, JobSpec
from ..security import redact_secrets, redacted_env_record
from ..store import atomic_json
from . import RemoteError, classify, http_code, paginate, remote_error
from .downloads import download_outputs

DATASET = re.compile(r"[\w-]+/[\w-]+(?:/[1-9]\d*)?")
# Longer than Kaggle's 12-hour session limit plus queueing; older runs cannot still be active.
ACTIVE_HORIZON = timedelta(hours=24)
MAX_SECONDS = 43200
STATES = {
    "QUEUED": "queued",
    "RUNNING": "running",
    "CANCEL_REQUESTED": "cancelling",
    "COMPLETE": "succeeded",
    "ERROR": "failed",
    "CANCEL_ACKNOWLEDGED": "cancelled",
}
# Remote states that still occupy an execution slot.
RUNNING = {"queued", "running", "cancelling"}


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


def _quiet():
    """Kaggle's client prints setup help and banners to stdout, which carries the CLI's JSON."""
    return contextlib.redirect_stdout(io.StringIO())


def _api_class(credentials: Path | None):
    # Kaggle's package authenticates eagerly on import. Keep it out of public model/API imports.
    with _quiet():
        from kaggle.api.kaggle_api_extended import AuthMethod, KaggleApi

    class BoundedApi(KaggleApi):
        def build_kaggle_client(self):
            client = super().build_kaggle_client()
            session = client._http_client
            session._init_session()
            session._session.mount("https://", TimeoutAdapter(max_retries=0))
            # The SDK transport prefers any ambient access token to the credentials it was given.
            # Pin it to those authenticate() resolved, which the provider checks against its account.
            values = self.config_values
            if values.get(self.CONFIG_NAME_TOKEN):
                session._session.auth = session.BearerAuth(values[self.CONFIG_NAME_TOKEN])
            elif values.get(self.CONFIG_NAME_USER) and values.get(self.CONFIG_NAME_KEY):
                session._session.auth = (values[self.CONFIG_NAME_USER], values[self.CONFIG_NAME_KEY])
            return client

        def _load_config(self):
            if credentials is None:
                return super()._load_config()
            # An explicit file is authoritative: KAGGLE_* variables and ~/.kaggle belong to another account.
            text = credentials.read_text().strip()
            try:
                values = json.loads(text)
            except ValueError:
                values = None
            if isinstance(values, dict):
                self.config_values = {key: str(values[key]) for key in ("username", "key") if key in values}
                self._file_token = None
            else:
                self.config_values, self._file_token = {}, text

        def _authenticate_with_access_token(self):
            if credentials is None:
                return super()._authenticate_with_access_token()
            username = self._file_token and self._introspect_token(self._file_token)
            if not username:
                return False
            self.config_values = self.config_values | {
                self.CONFIG_NAME_TOKEN: self._file_token,
                self.CONFIG_NAME_USER: username,
                self.CONFIG_NAME_AUTH_METHOD: str(AuthMethod.ACCESS_TOKEN),
            }
            return True

        def _authenticate_with_oauth_creds(self):
            return credentials is None and super()._authenticate_with_oauth_creds()

    return BoundedApi


class KaggleProvider:
    def __init__(self, account: Account, state_dir: Path, *, strict=False):
        self.account = account
        self.owner = account.user
        self.state_dir = state_dir
        self.strict = strict
        self._api = None
        self._resources = {}

    @property
    def api(self):
        if self._api is None:
            try:
                api = _api_class(self.account.credentials)()
                with _quiet():
                    api.authenticate()
            except (SystemExit, Exception) as error:
                raise RemoteError(
                    f"Kaggle authentication unavailable for {self.account.id}; configure its credentials",
                    "auth",
                ) from error
            user = api.config_values.get(api.CONFIG_NAME_USER) or ""
            if user.casefold() != self.owner.casefold():
                raise RemoteError(
                    f"Kaggle credentials for {self.account.id} authenticate as {user or 'nobody'}", "auth"
                )
            self._api = api
        return self._api

    def check(self, spec: JobSpec):
        if spec.accelerator and not spec.accelerator.startswith("Nvidia"):
            raise ValueError("Kaggle accepts NVIDIA GPU accelerator IDs only")
        if spec.timeout_seconds > MAX_SECONDS:
            raise ValueError(f"Kaggle runs are limited to {MAX_SECONDS} seconds")
        inputs = [ref for provider, ref in spec.dataset_inputs().values() if provider == "kaggle"]
        for ref in [*spec.datasets, *inputs]:
            if not DATASET.fullmatch(ref):
                raise ValueError(f"Invalid Kaggle dataset reference: {ref}")

    def url(self, ref):
        return f"https://www.kaggle.com/code/{ref}"

    def stage(self, job: JobRecord, number: int) -> str:
        name = re.sub(r"[^a-z0-9]+", "-", job.spec.name.lower())[:16].strip("-") or "workload"
        ref = f"{self.owner}/kgr-{name}-{job.id[:12]}-a{number}"
        prepare_kernel(job, ref, self._folder(job, number), self.state_dir)
        return ref

    def _folder(self, job, number):
        return self.state_dir / "jobs" / job.id / f"attempt-{number}"

    def submit(self, job: JobRecord) -> dict:
        attempt = job.attempts[-1]
        result = self.push(
            self._folder(job, attempt.number),
            timeout_seconds=job.spec.timeout_seconds,
            accelerator=job.spec.accelerator,
        )
        if result["ref"].lower() != attempt.ref.lower():
            raise RemoteError(
                "Kaggle returned an unexpected notebook identity; reconcile manually", "uncertain"
            )
        return result

    def resolve_dataset(self, ref):
        owner, slug, *version = ref.split("/")
        try:
            status = self.api.dataset_status(f"{owner}/{slug}", format="json(current_version_number)")
            result = json.loads(status)
        except Exception as error:
            # Kaggle answers 403 or 404 for a private dataset of another owner.
            if http_code(error) in {403, 404}:
                return None
            raise remote_error(error) from error
        current = result["current_version_number"]
        if version and int(version[0]) > current:
            return None  # That version does not exist.
        return ref if version else f"{ref}/{current}"

    def fetch_dataset(self, ref, destination: Path):
        try:
            # The client prints the dataset's URL even when quiet.
            with _quiet():
                self.api.dataset_download_files(ref, path=str(destination), quiet=True, unzip=True)
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
        raw = response.status.name
        return dict(state=STATES.get(raw), detail=raw, error=response.failure_message)

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
                    # The listing can lead with empty placeholder entries dated 2010; they are not runs.
                    if kernel is None or not kernel.ref:
                        continue
                    last_run = kernel.last_run_time
                    if last_run is not None and _utc(last_run) < cutoff:
                        return result
                    ref = kernel.ref
                    if ref.split("/")[0].lower() != self.owner.lower():
                        continue
                    try:
                        status = self.status(ref)
                    except RemoteError as error:
                        if error.kind == "missing":
                            continue
                        raise
                    if status["state"] in RUNNING:
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
                    if _read_timeout(error) and self.status(ref)["state"] in RUNNING:
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

    def download(self, ref, sink):
        download_outputs(self.output_pages(ref), sink, strict=self.strict, render=render_log)


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


def prepare_kernel(job: JobRecord, ref: str, folder: Path, state_dir: Path) -> Path:
    """Construct a private kernel from immutable local snapshots."""
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshot = job.snapshot
    source = snapshot["source"]
    payload = state_dir / "bundles" / source["digest"] / "files"
    inputs = {}
    for alias, bundle in [*snapshot["inputs"].items(), *job.transfers.items()]:
        inputs[alias] = dict(ref=job.upload_refs["input:" + alias], digest=bundle["digest"])
    for alias in job.spec.dataset_inputs():
        # Attached directly; the runtime finds where Kaggle mounted it.
        inputs.setdefault(alias, dict(dataset=job.upload_refs["input:" + alias]))
    config = dict(
        job_id=job.id,
        source_digest=source["digest"],
        source_ref=job.upload_refs.get("source"),
        inline=None,
        inputs=inputs,
        module=snapshot["module"],
        entrypoint=snapshot["entrypoint"],
        args=job.spec.command_args(),
        params=job.spec.params,
        env=job.spec.env,
        requirements=job.spec.requirements,
    )
    if snapshot["single_file"]:
        # Directory bundles carry their own manifest; only embedded files need their checksums here.
        config["source_files"] = source["files"]
        config["inline"] = {
            name: base64.b64encode((payload / name).read_bytes()).decode() for name in source["files"]
        }
    bootstrap = Path(runtime.__file__).read_text() + "\n_KGR_CONFIG = " + repr(config) + "\n"
    if snapshot["kind"] == "notebook":
        notebook = nbformat.read(payload / snapshot["entrypoint"], as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap + "bootstrap(_KGR_CONFIG)\n"))
        code_file = "workload.ipynb"
        nbformat.write(notebook, folder / code_file)
    else:
        code_file = "workload.py"
        (folder / code_file).write_text(bootstrap + "run_script(_KGR_CONFIG)\n")
    metadata = dict(
        id=ref,
        title=ref.split("/")[1].replace("-", " "),
        code_file=code_file,
        language="python",
        kernel_type=snapshot["kind"],
        is_private=True,
        enable_gpu=job.spec.gpu,
        enable_tpu=False,
        enable_internet=job.spec.internet,
        dataset_sources=list(
            dict.fromkeys(
                [
                    *[job.upload_refs.get("dataset:" + ref, ref) for ref in job.spec.datasets],
                    *[value for key, value in job.upload_refs.items() if not key.startswith("dataset:")],
                ]
            )
        ),
        competition_sources=[],
        kernel_sources=[],
        model_sources=[],
    )
    if job.spec.accelerator:
        metadata["machine_shape"] = job.spec.accelerator
    atomic_json(folder / "kernel-metadata.json", metadata)
    atomic_json(folder / "provenance.json", redacted_env_record(job.model_dump(mode="json")))
    return folder

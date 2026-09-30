"""Kaggle adapter: private notebooks run workloads; private datasets carry their bundles."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ...credentials import account_secrets, credentials_path, kaggle_secrets
from ...models import Account, JobRecord, JobSpec
from ...security import redact_secrets
from ...store import atomic_json
from .. import RemoteError, classify, http_code, paginate, remote_error
from ..base import Artifact, Provider
from ..downloads import download_outputs
from ..launch import inline_project
from .bundles import BundleDatasets
from .client import READ_TIMEOUT, api_class, quiet, read_timeout, utc
from .notebooks import prepare_kernel, render_log

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


class KaggleProvider(Provider):
    def __init__(self, account: Account, state_dir: Path, *, strict=False):
        super().__init__(account, state_dir, strict=strict)
        self.owner = account.user
        self.bundles = BundleDatasets(self)
        self._api = None
        # Each active run's resource by (ref, last run), read once per run.
        self._resources = {}

    # Authentication ---------------------------------------------------------------------------

    def _secrets(self):
        """The credentials file's entry, else an older per-account file, else Kaggle's discovery."""
        if secrets := account_secrets(self.account.id):
            return secrets
        if self.account.credentials is not None:
            return kaggle_secrets(self.account.credentials.expanduser().read_text())
        return None

    @property
    def api(self):
        if self._api is None:
            self._api = self._authenticate()
        return self._api

    def _authenticate(self):
        try:
            secrets = self._secrets()
        except (OSError, ValueError) as error:  # Messages name the file, never its contents.
            raise RemoteError(f"Cannot read the credentials of {self.account.id}: {error}", "auth") from error
        try:
            api = api_class(secrets)()
            with quiet():
                api.authenticate()
        except (SystemExit, Exception) as error:
            raise RemoteError(
                f"Kaggle authentication unavailable for {self.account.id}. Add its credentials "
                f"with compute-runner account add, or in {credentials_path()}",
                "auth",
            ) from error
        user = api.config_values.get(api.CONFIG_NAME_USER) or ""
        if user.casefold() != self.owner.casefold():
            raise RemoteError(
                f"Kaggle credentials for {self.account.id} authenticate as {user or 'nobody'}", "auth"
            )
        return api

    def _kernels(self, method, request, ref=None):
        if ref is not None:
            request.user_name, request.kernel_slug = ref.split("/")[:2]
        try:
            with self.api.build_kaggle_client() as client:
                return getattr(client.kernels.kernels_api_client, method)(request)
        except Exception as error:
            raise remote_error(error) from error

    # Workloads and data -----------------------------------------------------------------------

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

    def bundles_for(self, job: JobRecord) -> dict[str, dict]:
        """Small script projects travel inside the notebook (see launch.inline_project), not as a dataset."""
        bundles = super().bundles_for(job)
        if "source" in bundles and inline_project(job.snapshot, self.state_dir) is not None:
            del bundles["source"]
        return bundles

    def ensure_bundle(self, bundle) -> str | None:
        return self.bundles.ensure(bundle["digest"])

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
            with quiet():
                self.api.dataset_download_files(ref, path=str(destination), quiet=True, unzip=True)
        except Exception as error:
            raise remote_error(error) from error

    def stage(self, job: JobRecord, number: int) -> str:
        ref = f"{self.owner}/{self.launch_name(job, number)}"
        prepare_kernel(job, ref, self.attempt_folder(job, number), self.state_dir)
        return ref

    def submit(self, job: JobRecord) -> dict:
        attempt = job.attempts[-1]
        result = self.push(
            self.attempt_folder(job, attempt.number),
            timeout_seconds=job.spec.timeout_seconds,
            accelerator=job.spec.accelerator,
        )
        if result["ref"].lower() != attempt.ref.lower():
            raise RemoteError(
                "Kaggle returned an unexpected notebook identity; reconcile manually", "uncertain"
            )
        return result

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
        receipt = {
            "ref": response.ref,
            "url": response.url.split("?")[0] if response.url else None,
            "kernel_id": response.kernel_id,
            "version": response.version_number,
        }
        atomic_json(Path(folder) / "push-receipt.json", receipt)
        owner, slug, version = self.api.parse_kernel_string(self._pushed_ref(response.ref))
        return dict(
            ref=f"{owner}/{slug}", version=response.version_number or (int(version) if version else None)
        )

    def _pushed_ref(self, returned: str) -> str:
        """OWNER/SLUG[/VERSION] from the returned reference.

        The live service returns /code/owner/slug; also accept documented owner/slug/version,
        bare slug and absolute Kaggle URL forms.
        """
        parsed = urlsplit(returned)
        if parsed.netloc and parsed.hostname not in {"kaggle.com", "www.kaggle.com"}:
            raise RemoteError("Unexpected host in returned notebook reference", "uncertain")
        ref = parsed.path.removeprefix("/code/").strip("/")
        return ref if "/" in ref else f"{self.owner}/{ref}"

    # Runs -------------------------------------------------------------------------------------

    def status(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

        response = self._kernels("get_kernel_session_status", ApiGetKernelSessionStatusRequest(), ref)
        raw = response.status.name
        return dict(state=STATES.get(raw), detail=raw, error=response.failure_message)

    def _metadata(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        return self._kernels("get_kernel", ApiGetKernelRequest(), ref).metadata

    def _resource(self, ref):
        """The account listing reports every notebook as CPU; the notebook itself is accurate."""
        metadata = self._metadata(ref)
        if metadata is None or metadata.enable_gpu is None:
            return "unknown"
        return "gpu" if metadata.enable_gpu else "cpu"

    def runtime(self, ref):
        """The accelerator Kaggle saved for the notebook, separately from what the job asked for."""
        metadata = self._metadata(ref)
        saved = dict(
            enable_gpu=getattr(metadata, "enable_gpu", None),
            machine_shape=getattr(metadata, "machine_shape", None),
        )
        return dict(url=self.url(ref), provider=saved, session=self.status(ref))

    def cancel(self, ref, job_id):
        """Stop a run; True when it was removed before it started.

        A running session is cancelled by the ID its runtime logged, which the public API never
        returns. A run still queued has no session yet, so the attempt's own launch notebook is
        deleted instead, which drops it from Kaggle's queue.
        """
        from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest

        self._require_launch_notebook(ref, "cancel")
        found = re.search(rf"^KGR workload {job_id} session (\d+)$", self.live_log(ref), re.M)
        if found:
            request = ApiCancelKernelSessionRequest()
            request.kernel_session_id = int(found.group(1))
            response = self._kernels("cancel_kernel_session", request)
            if response.error_message:
                raise RemoteError(response.error_message, "invalid", definitive=True)
            return False
        if self.status(ref)["state"] != "queued":
            raise ValueError(
                "No session ID in this run's log yet (it is starting, or predates remote cancellation)"
            )
        self._delete_notebook(ref)
        return True

    def _require_launch_notebook(self, ref, action):
        if not ref.casefold().startswith(f"{self.owner}/kgr-".casefold()):
            raise ValueError(f"Refusing to {action} {ref}: not a launch notebook of this runner")

    def _delete_notebook(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiDeleteKernelRequest

        response = self._kernels("delete_kernel", ApiDeleteKernelRequest(), ref)
        if response.error_message:
            raise RemoteError(response.error_message, "invalid", definitive=True)

    # Capacity ---------------------------------------------------------------------------------

    def active_runs(self):
        """Best-effort account inventory; no metadata or source is written locally.

        One status call per kernel quickly hits Kaggle's rate limit on accounts with many
        notebooks, so only notebooks run within ACTIVE_HORIZON are checked.
        """
        result = {}
        # Metadata is read once per run; entries for finished runs are dropped.
        resources, self._resources = self._resources, {}
        try:
            for kernel in self._recent_kernels():
                ref = kernel.ref
                if ref.split("/")[0].lower() != self.owner.lower() or not self._holds_slot(ref):
                    continue
                key = (ref, kernel.last_run_time)
                self._resources[key] = resources.get(key) or self._resource(ref)
                result[ref] = self._resources[key]
            return result
        except Exception as error:
            raise remote_error(error) from error

    def _recent_kernels(self):
        """The account's notebooks run within ACTIVE_HORIZON, newest run first."""
        cutoff = datetime.now(timezone.utc) - ACTIVE_HORIZON
        pages = paginate(
            lambda token: self.api.kernels_list_with_response(
                mine=True, page_size=100, page_token=token, sort_by="dateRun"
            )
        )
        for response in pages:
            for kernel in response.kernels or []:
                # The listing can lead with empty placeholder entries dated 2010; they are not runs.
                if kernel is None or not kernel.ref:
                    continue
                if kernel.last_run_time is not None and utc(kernel.last_run_time) < cutoff:
                    return
                yield kernel

    def _holds_slot(self, ref):
        try:
            return self.status(ref)["state"] in RUNNING
        except RemoteError as error:
            if error.kind == "missing":
                return False
            raise

    def quota(self):
        try:
            response = self.api.quota_view()
            refresh = response.quota_refresh_time
            result = {"refresh_at": utc(refresh).timestamp() if refresh else None}
            for name in ("gpu", "tpu"):
                result[name] = _time_quota(getattr(response, name + "_quota"))
            return result
        except Exception as error:
            raise remote_error(error) from error

    # Logs and outputs -------------------------------------------------------------------------

    def logs(self, ref, *, follow=False):
        try:
            if follow:
                yield from self._follow(ref)
            else:
                yield redact_secrets(render_log(self.api.kernels_logs(ref)), strict=self.strict)
        except Exception as error:
            raise remote_error(error) from error

    def _follow(self, ref, *, attempts=5):
        """Stream a run's log, reconnecting after drops; the stream replays from the start."""
        seen = failures = 0
        while failures < attempts:
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
                if read_timeout(error) and self.status(ref)["state"] in RUNNING:
                    continue
                failures = failures + 1 if seen == before else 0
                if failures < attempts:
                    time.sleep(min(2**failures, 15))
        raise RemoteError("Log stream disconnected repeatedly; retry logs --follow or open the Kaggle URL")

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
            if not chunks and not read_timeout(error):
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

    # Diagnostics and cleanup ------------------------------------------------------------------

    def input_status(self, job: JobRecord) -> list[dict]:
        """What Kaggle reports for up to ten datasets a job attaches, beside the local receipts."""
        bundles = {"source": job.snapshot["source"], **job.snapshot["inputs"], **job.transfers}
        refs = list(self._input_refs(job).items())[:10]
        return [self.bundles.describe(alias, ref, bundles.get(alias)) for alias, ref in refs]

    def _input_refs(self, job: JobRecord) -> dict[str, str]:
        """Each input's dataset by alias: its bundle's, the dataset it names, its copy's, or what it got."""
        uploads = self.bundles_for(job).items()
        refs = {key.removeprefix("input:"): self.bundles.ref(bundle["digest"]) for key, bundle in uploads}
        refs |= {alias: ref for alias, (kind, ref) in job.spec.dataset_inputs().items() if kind == "kaggle"}
        refs |= {alias: self.bundles.ref(bundle["digest"]) for alias, bundle in job.transfers.items()}
        refs |= {key.removeprefix("input:"): ref for key, ref in job.upload_refs.items()}
        return refs

    def artifacts(self) -> list[Artifact]:
        """This runner's launch notebooks (kgr-*) and bundle datasets (kgr-b-*) on the account."""
        try:
            return [*self._launch_notebooks(), *self.bundles.artifacts()]
        except Exception as error:
            raise remote_error(error) from error

    def _launch_notebooks(self):
        pages = paginate(
            lambda token: self.api.kernels_list_with_response(
                mine=True, page_size=100, page_token=token, search="kgr-"
            )
        )
        for response in pages:
            for kernel in response.kernels or []:
                if kernel is None or not (kernel.ref or "").casefold().startswith(
                    f"{self.owner}/kgr-".casefold()
                ):
                    continue
                last_run = kernel.last_run_time
                modified = utc(last_run).timestamp() if last_run else None
                yield Artifact("notebook", kernel.ref, modified_at=modified, attempt=kernel.ref)

    def delete_artifact(self, artifact: Artifact) -> None:
        if artifact.kind == "notebook":
            self._require_launch_notebook(artifact.name, "delete")
            self._delete_notebook(artifact.name)
        elif artifact.kind == "dataset":
            self.bundles.delete(artifact.name)
        else:
            super().delete_artifact(artifact)


def _time_quota(quota) -> dict | None:
    """Used, reserved, total and available seconds of one accelerator's weekly time; None without any."""
    if quota is None:
        return None
    used, reserved, total = (
        value.total_seconds() if value is not None else 0
        for value in (quota.time_used, quota.time_reserved, quota.total_time_allowed)
    )
    return dict(
        used_seconds=used,
        reserved_seconds=reserved,
        total_seconds=total,
        available_seconds=max(0, total - used - reserved),
    )

"""The contract between the queue and a compute provider, plus shared remote-error handling.

One provider instance serves one account. Adapters translate their service's identities,
states and errors into these shapes; the worker never sees provider-specific values.

Data follows one contract on every provider:

- Local inputs and copied datasets are content-addressed bundles. ensure_bundle() reuses one
  the account already holds and uploads it otherwise.
- A provider dataset is attached directly when resolve_dataset() says the account can read it.
  If it cannot, the worker copies it from an account that can (fetch_dataset, then
  ensure_bundle) when the user allowed copying.
- The workload finds every input the same way: KGR_INPUT_<ALIAS> and KGR_INPUTS_JSON, set by
  the launch package that stage() builds. Workloads never use provider paths.
- download() hands each output to an OutputSink, which decides what to fetch and where it
  goes, so every provider fills the same run folder.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..security import redact_secrets

if TYPE_CHECKING:
    from ..models import Account, Config, JobRecord, JobSpec
    from ..results import OutputSink


class Provider(Protocol):
    def check(self, spec: JobSpec) -> None:
        """Raise ValueError if this provider cannot run the specification."""

    def ensure_bundle(self, bundle: dict) -> str | None:
        """Make a content-addressed local bundle available to runs; None while it is still processing."""

    def resolve_dataset(self, ref: str) -> str | None:
        """Pin a dataset this account can read to an immutable version; None if it cannot read it.

        Every call checks access, even for a pinned reference, so the worker can ask any account.
        Raise RemoteError for failures that say nothing about access.
        """

    def fetch_dataset(self, ref: str, destination: Path) -> None:
        """Download a dataset this account can read, as plain files, into an empty folder."""

    def stage(self, job: JobRecord, number: int) -> str:
        """Build attempt number's launch package locally and return its remote reference.

        No remote calls. The reference is deterministic, so status() can find the run
        even when submit() is interrupted. The package exposes each input alias as
        KGR_INPUT_<ALIAS>: bundles by job.upload_refs["input:ALIAS"], verified against their
        digest (from job.snapshot["inputs"] or job.transfers), and datasets attached directly.
        """

    def url(self, ref: str) -> str | None: ...

    def submit(self, job: JobRecord) -> dict:
        """Launch the staged job.attempts[-1]; may return {"version": n}.

        Raise RemoteError. definitive=True asserts that nothing was launched.
        """

    def status(self, ref: str) -> dict:
        """{"state": ..., "detail": raw provider state, "error": str | None}.

        state is queued, running, cancelling, succeeded, failed or cancelled; None if unrecognized.
        """

    def cancel(self, ref: str, job_id: str) -> None: ...

    def active_runs(self) -> dict[str, str]:
        """Runs holding this account's capacity, including ones started elsewhere: {ref: cpu|gpu|unknown}."""

    def quota(self) -> dict:
        """{"gpu": {"available_seconds": ...} or None, ...}"""

    def logs(self, ref: str, *, follow: bool = False) -> Iterator[str]:
        """The stored log of a finished run, or a stream with follow=True."""

    def live_log(self, ref: str) -> str:
        """A bounded snapshot of an unfinished run's log."""

    def download(self, ref: str, sink: OutputSink) -> None:
        """Give sink the run's log (sink.log) and its output files.

        For each file, ask sink.target(name) for a path (None: skip it), write the bytes there
        atomically, then call sink.saved(name, path, sha256).
        """


def connect(account: Account, config: Config) -> Provider:
    # Imported here: adapters load their SDKs lazily and must not burden model imports.
    from .kaggle import KaggleProvider

    adapters = {"kaggle": KaggleProvider}
    return adapters[account.provider](account, config.state_dir, strict=config.strict)


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

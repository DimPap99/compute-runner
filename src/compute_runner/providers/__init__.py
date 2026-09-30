"""Compute providers: the adapter for each kind of account, and shared remote-error handling.

See base.Provider for the contract every adapter fulfils.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ..security import redact_secrets
from .base import Artifact, Inventory, Provider

if TYPE_CHECKING:
    from ..models import Account, Config

__all__ = [
    "Artifact",
    "Inventory",
    "Provider",
    "RemoteError",
    "connect",
    "remote_error",
    "safe_message",
    "short",
]

# Adapters by account provider, imported on first use: they load their SDKs lazily and must not
# burden model imports. A new provider is one entry here and one Provider subclass.
ADAPTERS = {
    "kaggle": (".kaggle", "KaggleProvider"),
    "ssh": (".ssh", "SshProvider"),
}


def connect(account: Account, config: Config) -> Provider:
    module, name = ADAPTERS[account.provider]
    adapter = getattr(importlib.import_module(module, __name__), name)
    return adapter(account, config.state_dir, strict=config.strict)


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
        # Validation errors: every problem up to three, so an unknown field is not hidden behind
        # the missing one it was meant to be.
        problems = []
        for detail in error.errors(include_input=False, include_url=False)[:3]:
            message = detail["msg"].removeprefix("Value error, ")
            field = ".".join(map(str, detail["loc"]))
            problems.append(f"Invalid {field}: {message}" if field else message)
        text = "; ".join(problems) or text
    except (AttributeError, KeyError, TypeError):
        pass
    response = getattr(error, "response", None)
    if response is not None:
        try:
            body = response.json()
            detail = body.get("message") or body.get("error") or body.get("detail") or body.get("title")
            if isinstance(detail, str):
                text = f"HTTP {http_code(error)}: {detail}"
            elif body:
                # Validation endpoints also return field-keyed errors. Keep their
                # diagnostic detail, subject to the redaction and bound below.
                text = f"HTTP {http_code(error)}: {body}"
        except (ValueError, AttributeError):
            pass
    return redact_secrets(text, strict=True)[:2000]


def short(value, limit=400):
    """A safe message on one line, at most limit characters; None stays None."""
    if value is None:
        return None
    text = " ".join(safe_message(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


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

"""Kaggle's API client, bound to one account's credentials, with timeouts and a quiet stdout."""

from __future__ import annotations

import contextlib
import contextvars
import io
import threading
from datetime import timezone

import requests
import urllib3
from requests.adapters import HTTPAdapter

# Read timeout for calls without an explicit timeout; lowered while snapshotting a live log stream.
READ_TIMEOUT = contextvars.ContextVar("kgr_read_timeout", default=90)
# Redirecting stdout swaps it for the whole process, so one redirect at a time.
_STDOUT = threading.RLock()


def utc(value):
    """The service returns naive UTC timestamps."""
    return value.replace(tzinfo=value.tzinfo or timezone.utc)


def read_timeout(error) -> bool:
    return isinstance(error, requests.ReadTimeout) or any(
        isinstance(arg, urllib3.exceptions.ReadTimeoutError) for arg in error.args
    )


@contextlib.contextmanager
def quiet():
    """Kaggle's client prints setup help and banners to stdout, which carries the CLI's JSON."""
    with _STDOUT, contextlib.redirect_stdout(io.StringIO()):
        yield


class TimeoutAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (15, READ_TIMEOUT.get())
        return super().send(request, **kwargs)


def api_class(secrets: dict | None):
    """Kaggle's client bound to secrets (see credentials.kaggle_secrets), or its usual discovery for None."""
    # Kaggle's package authenticates eagerly on import. Keep it out of public model/API imports.
    with quiet():
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
            if secrets is None:
                return super()._load_config()
            # Saved secrets are authoritative: KAGGLE_* variables and ~/.kaggle belong to another account.
            self.config_values = {key: secrets[key] for key in ("username", "key") if key in secrets}
            self._file_token = secrets.get("token")

        def _authenticate_with_access_token(self):
            if secrets is None:
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
            return secrets is None and super()._authenticate_with_oauth_creds()

    return BoundedApi

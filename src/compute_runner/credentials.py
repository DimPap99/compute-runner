"""The credentials file: every account's secrets in one place, beside the configuration.

    {
      "kaggle:alice": {"username": "alice", "key": "..."},
      "kaggle:bob": {"token": "KGAT_..."},
      "ssh:lab": {"key": "~/.ssh/id_ed25519", "passphrase": "..."},
      "ssh:lab-cpu": {"password": "..."}
    }

account add writes it, and the user may edit it. It is read only on this machine, to log in:
nothing in it is uploaded, stored with jobs, logged, or shown in responses.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Account
from .store import atomic_json, config_path

FIELDS = {"kaggle": {"username", "key", "token"}, "ssh": {"key", "password", "passphrase"}}


def credentials_path() -> Path:
    return config_path().parent / "credentials.json"


def _load() -> dict:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as error:  # The message gives a position, never the file's contents.
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must map account IDs to their credentials")
    return data


def account_secrets(account_id: str) -> dict:
    """The saved secrets of one account, or {} when it has none (the provider's defaults apply)."""
    found = {key.casefold(): value for key, value in _load().items()}.get(account_id.casefold(), {})
    provider = account_id.split(":", 1)[0].casefold()
    unknown = set(found) - FIELDS.get(provider, set()) if isinstance(found, dict) else None
    if unknown is None or unknown or not all(isinstance(value, str) for value in found.values()):
        allowed = ", ".join(sorted(FIELDS.get(provider, ())))
        raise ValueError(f"Credentials of {account_id} in {credentials_path()} take text fields: {allowed}")
    return found


def imported_key_path(account_id: str) -> Path:
    """Where account import keeps an SSH account's private key: beside the credentials file."""
    return credentials_path().parent / "keys" / account_id.casefold().replace(":", "-")


def save_secrets(account_id: str, secrets: dict | None) -> Path:
    """Replace one account's secrets; None forgets them. The file is readable by this user only.

    A key that account import saved for the account is deleted once its secrets no longer name it.
    """
    saved = _load()
    data = {key: value for key, value in saved.items() if key.casefold() != account_id.casefold()}
    if secrets:
        data[account_id] = secrets
    path = credentials_path()
    if data != saved:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json(path, data)
    imported = imported_key_path(account_id)
    if (secrets or {}).get("key") != str(imported):
        imported.unlink(missing_ok=True)
    return path


def login_secrets(account: Account) -> dict:
    """The secrets an account logs in with: its credentials file entry, else the key, password or
    credentials file an older configuration names; {} when the provider's defaults apply."""
    if secrets := account_secrets(account.id):
        return secrets
    if account.credentials is not None:
        return kaggle_secrets(account.credentials.expanduser().read_text())
    if account.ssh and account.ssh.password_file is not None:
        return {"password": account.ssh.password_file.expanduser().read_text().rstrip("\n")}
    return {"key": str(account.ssh.key)} if account.ssh and account.ssh.key else {}


def kaggle_secrets(text: str) -> dict:
    """Secrets from a kaggle.json (username and key) or an access token."""
    text = text.strip()
    try:
        values = json.loads(text)
    except ValueError:
        values = None
    if isinstance(values, dict):
        return {name: str(values[name]) for name in ("username", "key") if name in values}
    return {"token": text}

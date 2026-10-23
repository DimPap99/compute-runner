"""Account exports: accounts, their secrets and SSH host keys in one passphrase-encrypted file.

    {"format": "compute-runner-accounts", "version": 1, "salt": "...", "nonce": "...", "data": "..."}

data is AES-256-GCM over the JSON of an AccountExport, with a key derived from the passphrase
by scrypt, so a wrong passphrase or a changed byte fails to decrypt. account export writes it on
one computer, and account import adds its accounts on another.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import unicodedata
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import Field, model_validator

from .credentials import FIELDS, imported_key_path, login_secrets, save_secrets
from .models import Account, Model
from .providers.ssh import exported_host_keys
from .security import ACCOUNT_EXPORT_FORMAT as FORMAT
from .store import atomic_write, json_bytes

VERSION = 1
# 128 * n * r bytes of memory (128 MiB) and a fraction of a second for each passphrase tried.
SCRYPT = dict(n=2**17, r=8, p=1, maxmem=2**28)
MIN_PASSPHRASE = 8


class ExportedAccount(Model):
    account: Account
    # As in the credentials file, except that an SSH key file travels as its base64 bytes, private_key.
    secrets: dict[str, str] = Field(default_factory=dict)
    # An SSH machine's known host keys: {type: base64}, as known_hosts lists them.
    host_keys: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def provider_fields(self):
        provider = self.account.provider
        allowed = FIELDS[provider] - {"key"} | {"private_key"} if provider == "ssh" else FIELDS[provider]
        if unknown := set(self.secrets) - allowed:
            raise ValueError(f"{self.account.id} has unknown secrets: {', '.join(sorted(unknown))}")
        if self.host_keys and provider != "ssh":
            raise ValueError(f"{self.account.id} is not an SSH machine and has no host keys")
        return self


class AccountExport(Model):
    # In order of preference.
    accounts: list[ExportedAccount]

    @model_validator(mode="after")
    def unique(self):
        ids = [entry.account.id.casefold() for entry in self.accounts]
        if len(set(ids)) != len(ids):
            raise ValueError("Accounts must be unique")
        return self


def exported(account: Account) -> ExportedAccount:
    """An account as an export holds it: settings without this computer's paths, and the secrets."""
    try:
        secrets = dict(login_secrets(account))
        if account.provider == "ssh" and "key" in secrets:
            key = Path(secrets.pop("key")).expanduser().read_bytes()
            secrets["private_key"] = base64.b64encode(key).decode()
    except (OSError, ValueError) as error:  # Messages name the file, never its contents.
        raise ValueError(f"Cannot read the credentials of {account.id}: {error}") from None
    # The files an older configuration names are read into secrets above.
    settings = account.model_dump(exclude={"credentials": True, "ssh": {"key", "password_file"}})
    host_keys = exported_host_keys(account) if account.provider == "ssh" else {}
    return ExportedAccount(account=settings, secrets=secrets, host_keys=host_keys)


def install_secrets(account_id: str, secrets: dict) -> None:
    """Save an exported account's secrets; its SSH key goes to a private file of its own first."""
    secrets = dict(secrets)
    if "private_key" in secrets:
        path = imported_key_path(account_id)
        atomic_write(path, base64.b64decode(secrets.pop("private_key")))
        secrets["key"] = str(path)
    save_secrets(account_id, secrets)


def _cipher(passphrase: str, salt: bytes) -> AESGCM:
    # Composed the same way everywhere, so accents typed on another system give the same key.
    secret = unicodedata.normalize("NFC", passphrase).encode()
    return AESGCM(hashlib.scrypt(secret, salt=salt, dklen=32, **SCRYPT))


def seal(export: AccountExport, passphrase: str) -> bytes:
    """The export encrypted with a passphrase, as its file holds it."""
    if len(passphrase) < MIN_PASSPHRASE:
        raise ValueError(f"The passphrase must have at least {MIN_PASSPHRASE} characters")
    salt, nonce = os.urandom(16), os.urandom(12)
    data = _cipher(passphrase, salt).encrypt(nonce, export.model_dump_json().encode(), None)
    fields = dict(salt=salt, nonce=nonce, data=data)
    encoded = {name: base64.b64encode(value).decode() for name, value in fields.items()}
    return json_bytes(dict(format=FORMAT, version=VERSION) | encoded)


def envelope(path: Path) -> dict:
    """An export file's unencrypted fields, once they show this version can read it."""
    try:
        value = json.loads(path.expanduser().read_bytes())
    except ValueError:
        value = None
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError(f"{path} is not an account export")
    if value.get("version") != VERSION:
        raise ValueError(
            f"{path} was exported by another version of compute-runner (format {value.get('version')}); "
            f"this one reads format {VERSION}"
        )
    return value


def unseal(sealed: dict, passphrase: str) -> AccountExport:
    """The export inside an envelope, decrypted with its passphrase."""
    try:
        salt, nonce, data = (
            base64.b64decode(sealed[name], validate=True) for name in ("salt", "nonce", "data")
        )
        plain = _cipher(passphrase, salt).decrypt(nonce, data, None)
    except (KeyError, TypeError, ValueError, InvalidTag):
        raise ValueError("Wrong passphrase, or the export is damaged") from None
    return AccountExport.model_validate_json(plain)

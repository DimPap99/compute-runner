"""Verified, resumable downloads of run outputs exposed as signed HTTPS URLs."""

from __future__ import annotations

import fnmatch
import hashlib
import ipaddress
import json
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from ..runtime import file_digest, safe_relative
from ..security import redact_secrets
from ..store import atomic_json, atomic_write


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


def download_outputs(pages, destination, patterns=None, *, skip=None, get=None, strict=False, render=str):
    """Download run outputs; names matching the skip predicate are not fetched.

    Each page has a log (converted to text by render) and files with file_name and url.

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
                log_data = redact_secrets(render(page.log), strict=strict).encode()
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

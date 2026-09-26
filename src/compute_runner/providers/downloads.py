"""Verified, resumable downloads of run outputs exposed as signed HTTPS URLs."""

from __future__ import annotations

import hashlib
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import requests

from ..runtime import safe_relative
from ..store import atomic_write


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


def download_outputs(pages, sink, *, get=None, strict=False, render=str):
    """Fetch run outputs into sink (see results.OutputSink), which chooses and places each file.

    Each page has a log (converted to text by render) and files with file_name and url.
    Returns the sink's receipts.

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
        for page in pages:
            if page.log:
                # Always retrieve logs even when output filtering selects no files.
                sink.log(render(page.log))
            for item in page.files or []:
                name = safe_relative(item.file_name)
                target = sink.target(name)
                if target is None:
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
                sink.saved(name, target, digest.hexdigest())
        return sink.receipts
    finally:
        if session is not None:
            session.close()

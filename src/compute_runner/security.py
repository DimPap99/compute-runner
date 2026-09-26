"""Credential screening and redaction shared by local and remote artifacts."""

from __future__ import annotations

import re


SECRET_FILE_NAMES = {
    ".env",
    ".envrc",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "access_token",
    "application_default_credentials.json",
    "auth.json",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "kaggle.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "service-account.json",
}
SECRET_SUFFIXES = {".jks", ".key", ".keystore", ".p12", ".pem", ".pfx", ".tfstate"}

_SECRET_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_KEY|AUTH|CREDENTIALS?|KEY|PASSWORD|PASSWD|PRIVATE_KEY|SECRET|TOKEN)(?:$|_)"
)
_REDACT_FIELD = re.compile(
    r"(['\"]?[A-Z0-9_-]{0,64}(?:KAGGLE[_-]?KEY|API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|"
    r"PRIVATE[_-]?KEY|CREDENTIALS?)[A-Z0-9_-]{0,64}['\"]?\s*[:=]\s*['\"]?)[^'\"\s,;}\]]+",
    re.I,
)
_SECRET_PATTERNS = (
    (
        "private key",
        re.compile(
            rb"-----BEGIN (?:ENCRYPTED |RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
            rb"(?:[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)?"
        ),
    ),
    ("Kaggle token", re.compile(rb"(?i)(?<![A-Za-z0-9])KGAT_[A-Za-z0-9._~+/=-]{12,}")),
    ("Kaggle API key", re.compile(rb"(?i)KAGGLE_KEY\s*[:=]\s*['\"]?[0-9a-f]{32}")),
    ("AWS access key", re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}")),
    ("GitLab token", re.compile(rb"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{20,}")),
    ("OpenAI key", re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("Google API key", re.compile(rb"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{30,}")),
    ("Slack token", re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("Stripe live key", re.compile(rb"(?<![A-Za-z0-9])(?:sk|rk)_live_[A-Za-z0-9]{16,}")),
    (
        "JSON web token",
        re.compile(
            rb"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            rb"\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "assigned credential",
        re.compile(
            rb"(?ix)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
            rb"private[_-]?key|password|passwd|secret[_-]?key)\s*[:=]\s*['\"]?"
            rb"[A-Za-z0-9_./+~=-]{16,}"
        ),
    ),
)


def secret_filename(name: str) -> bool:
    """Return whether a basename should never enter a source or input bundle."""
    folded = name.casefold()
    return (
        folded in SECRET_FILE_NAMES
        or folded.startswith(".env.")
        or folded.startswith(("id_dsa_", "id_ecdsa_", "id_ed25519_", "id_rsa_"))
        or any(folded.endswith(suffix) for suffix in SECRET_SUFFIXES)
        or folded.endswith(".tfstate.backup")
    )


def detected_secret(data: bytes | str) -> str | None:
    """Identify high-confidence credential material without returning the value."""
    raw = data.encode() if isinstance(data, str) else data
    for kind, pattern in _SECRET_PATTERNS:
        if pattern.search(raw):
            return kind
    return None


def validate_nonsecret_env(env: dict[str, str]) -> None:
    """The env feature is persisted and uploaded, so reject values that look secret."""
    for name, value in env.items():
        if _SECRET_ENV_NAME.search(name.upper()):
            raise ValueError(f"env variable {name} looks secret; env accepts nonsecret values only")
        if kind := detected_secret(value):
            raise ValueError(f"env variable {name} contains a detected {kind}; do not submit credentials")


def redacted_env_record(value: dict) -> dict:
    """Copy a dumped job record while hiding environment values in user-facing artifacts."""
    value = dict(value)
    spec = dict(value.get("spec", {}))
    if "env" in spec:
        spec["env"] = {name: "[redacted]" for name in spec["env"]}
    value["spec"] = spec
    return value


def redact_secrets(value, *, strict=False) -> str:
    """Remove credentials from text before persistence or display.

    URL passwords and known credential formats are always removed. strict also removes
    anything credential-shaped (URL query strings, words after Bearer/Basic, values of
    token/key/secret/password fields), which can hide ordinary text such as num_tokens=512.
    """
    text = re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[redacted]@", str(value), flags=re.I)
    raw = text.encode("utf-8", "surrogatepass")
    for _, pattern in _SECRET_PATTERNS:
        raw = pattern.sub(b"[redacted]", raw)
    text = raw.decode("utf-8", "surrogatepass")
    if strict:
        text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", text, flags=re.I)
        text = re.sub(r"(?i)(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+", "[redacted]", text)
        if any(
            marker in text.casefold()
            for marker in ("key", "token", "secret", "password", "passwd", "credential")
        ):
            text = _REDACT_FIELD.sub(r"\1[redacted]", text)
    return text

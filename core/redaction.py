"""Small, shared credential-redaction helpers for logs and persistent diagnostics."""

from __future__ import annotations

import re


SECRET_PATTERNS = (
    re.compile(r"(?i)(?:password|passwd|api[_ -]?key|secret|token|private[_ -]?key)\s*[:=]"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])(AIza[0-9A-Za-z_-]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16})(?![A-Za-z0-9])"),
)
PATH_EXFIL_PATTERNS = (
    re.compile(r"(?i)(?:[A-Za-z]:[\\/]|\\\\|/home/|/users/|/workspace/|file://)"),
    re.compile(r"(?<!\w)(?:\.{1,2}[\\/])"),
)

_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(password|passwd|api[_ -]?key|secret|token|private[_ -]?key|authorization|cookie|set-cookie)
    \b\s*[:=]\s*
    (?:"[^"]*"|'[^']*'|Bearer\\s+[A-Za-z0-9._~+/=-]{8,}|[^\\s,;]+)
    """
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
_PATH_VALUE = re.compile(r"(?i)(?:[A-Z]:[\\/]|\\\\|/home/|/users/|/workspace/|file://)[^\s\"'<>]+")
_TOKEN_PATTERNS = tuple(pattern for pattern in SECRET_PATTERNS[2:])


def contains_suspected_secret(value: str) -> bool:
    """Return True when text matches a common credential or private-key marker."""
    text = str(value or "")
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def redact_sensitive_text(value: object, *, max_chars: int | None = None) -> str:
    """Redact common credential forms and host paths before diagnostics are stored."""
    text = str(value or "")
    text = _PRIVATE_KEY_BLOCK.sub("[REDACTED PRIVATE KEY]", text)
    text = _CREDENTIAL_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _BEARER_CREDENTIAL.sub("Bearer [REDACTED]", text)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub("[REDACTED CREDENTIAL]", text)
    text = _PATH_VALUE.sub("[REDACTED PATH]", text)

    # Catch long high-entropy tokens that do not match a known provider prefix.
    def redact_high_entropy(match: re.Match[str]) -> str:
        token = match.group(0)
        categories = sum((
            bool(re.search(r"[a-z]", token)),
            bool(re.search(r"[A-Z]", token)),
            bool(re.search(r"[0-9]", token)),
            bool(re.search(r"[^A-Za-z0-9]", token)),
        ))
        return "[REDACTED TOKEN]" if len(token) >= 40 and categories >= 3 else token

    text = re.sub(r"[A-Za-z0-9_+/=-]{40,}", redact_high_entropy, text)
    if max_chars is not None:
        text = text[:max(0, int(max_chars))]
    return text

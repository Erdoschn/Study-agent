"""Backward-compatible facade for storage security guards.

The implementation now lives in study_agent.infrastructure. Existing core.*
imports remain supported while callers migrate.
"""
from study_agent.infrastructure.storage_security import (
    MAX_SQLITE_FILE_BYTES,
    _SQLITE_SIDECARS,
    _ensure_directory_chain,
    _validate_regular_file,
    ensure_private_directory,
    ensure_safe_sqlite_path,
)

__all__ = [
    "MAX_SQLITE_FILE_BYTES",
    "_SQLITE_SIDECARS",
    "_ensure_directory_chain",
    "_validate_regular_file",
    "ensure_private_directory",
    "ensure_safe_sqlite_path",
]

"""Discover and cache reasoning-effort metadata for explicitly configured API models.

Only small capability metadata is persisted. Credentials and provider responses are never
written to disk. Provider catalogs are advisory; configured models are never auto-added.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


ALLOWED_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
DEFAULT_TTL_SECONDS = 24 * 60 * 60


class ModelCapabilityCache:
    def __init__(self, root: Path, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.path = Path(root) / "data" / "model_capabilities.json"
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._generation = 0
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _key(provider_name: str, provider: dict[str, Any], model_id: str) -> str:
        # Hash credentials into the identity so accounts with different permissions
        # cannot accidentally share capabilities. The secret itself is never persisted.
        headers = provider.get("headers", {})
        auth_identity = provider.get("api_key", "")
        if not auth_identity and isinstance(headers, dict):
            auth_identity = headers.get("Authorization", "")
        identity = "\0".join((
            provider_name,
            str(provider.get("base_url", "")).rstrip("/"),
            model_id,
            hashlib.sha256(str(auth_identity).encode()).hexdigest(),
        ))
        return hashlib.sha256(identity.encode()).hexdigest()

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if number == number and abs(number) != float("inf") else default
        except (TypeError, ValueError, OverflowError):
            return default

    def _load(self) -> None:
        try:
            if self.path.stat().st_size > 1024 * 1024:
                self._entries = {}
                return
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
            if isinstance(entries, dict):
                self._entries = {
                    str(key): value for key, value in entries.items()
                    if isinstance(value, dict)
                    and isinstance(value.get("capabilities"), dict)
                }
        except (OSError, UnicodeError, json.JSONDecodeError):
            self._entries = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"schema_version": 1, "entries": self._entries},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fd, temp_path = tempfile.mkstemp(prefix=".model-capabilities-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def _levels(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if item is None:
                continue
            level = str(item).strip().lower()
            if level in ALLOWED_EFFORTS and level not in result:
                result.append(level)
        return result

    @classmethod
    def _extract_metadata(cls, item: dict[str, Any]) -> tuple[list[str], str | None]:
        effort = item.get("effort")
        if not isinstance(effort, dict):
            effort = item.get("reasoning_effort")
        if not isinstance(effort, dict):
            effort = {}
        levels = cls._levels(effort.get("supported_levels", effort.get("levels")))
        default = str(effort.get("default_level", "") or "").strip().lower()
        if default not in ALLOWED_EFFORTS:
            default = None
        return levels, default

    def refresh(self, config: dict[str, Any], force: bool = False) -> dict[str, Any]:
        # Single-flight refresh; do not hold the data lock during network I/O.
        if not self._refresh_lock.acquire(blocking=False):
            return self.snapshot()
        try:
            providers = config.get("providers", {})
            models = config.get("models", {})
            if not isinstance(providers, dict) or not isinstance(models, dict):
                return self.snapshot()
            with self._lock:
                generation = self._generation

            changed = False
            provider_catalogs: dict[str, tuple[list[dict[str, Any]] | None, str | None]] = {}
            for model_name, model in models.items():
                if not isinstance(model, dict):
                    continue
                provider_name = str(model.get("provider", "")).strip()
                provider = providers.get(provider_name)
                if not isinstance(provider, dict) or not provider.get("enabled", False):
                    continue
                if str(provider.get("type", "openai_compatible")).lower() != "openai_compatible":
                    continue
                base_url = str(provider.get("base_url", "")).rstrip("/")
                model_id = str(model.get("model", model_name)).strip()
                if not base_url or not model_id:
                    continue
                key = self._key(provider_name, provider, model_id)
                with self._lock:
                    if generation != self._generation:
                        return self.snapshot()
                    cached = dict(self._entries.get(key, {}))
                now = time.time()
                configured_profile_levels = self._levels(model.get("reasoning_efforts"))
                if not configured_profile_levels and isinstance(model.get("extra"), dict):
                    configured_profile_levels = self._levels(model["extra"].get("reasoning_efforts"))
                configured_default = str(model.get("reasoning_effort_default", "")).strip().lower()
                if configured_default not in ALLOWED_EFFORTS:
                    configured_default = None
                profile_changed = (
                    cached.get("source") == "configured_profile"
                    and (
                        cached.get("capabilities", {}).get("supported_levels", []) != configured_profile_levels
                        or cached.get("capabilities", {}).get("default_level") != configured_default
                    )
                )
                if not force and cached and not profile_changed and now - self._number(cached.get("fetched_at")) < self.ttl_seconds:
                    continue

                levels: list[str] = []
                default: str | None = None
                source = "unknown"
                raw_headers = provider.get("headers", {})
                headers = {"Accept": "application/json"}
                if isinstance(raw_headers, dict):
                    headers.update({str(k): str(v) for k, v in raw_headers.items()})
                auth_identity = provider.get("api_key", "") or headers.get("Authorization", "")
                provider_cache_key = hashlib.sha256("\0".join((
                    provider_name,
                    base_url,
                    hashlib.sha256(str(auth_identity).encode()).hexdigest(),
                )).encode()).hexdigest()
                if provider_cache_key not in provider_catalogs:
                    error_type = None
                    try:
                        if "Authorization" not in headers and provider.get("api_key"):
                            headers["Authorization"] = "Bearer " + str(provider["api_key"])
                        request = urllib.request.Request(base_url + "/models", headers=headers, method="GET")
                        timeout = min(max(int(provider.get("timeout", 30)), 3), 15)
                        with urllib.request.urlopen(request, timeout=timeout) as response:
                            raw = response.read(2 * 1024 * 1024 + 1)
                        if len(raw) > 2 * 1024 * 1024:
                            raise ValueError("models catalog exceeds 2 MiB")
                        payload = json.loads(raw.decode("utf-8"))
                        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                            raise ValueError("invalid models catalog")
                        catalog = [item for item in payload["data"] if isinstance(item, dict)]
                    except Exception as exc:
                        catalog = None
                        error_type = type(exc).__name__
                    provider_catalogs[provider_cache_key] = (catalog, error_type)
                catalog, error_type = provider_catalogs[provider_cache_key]
                if catalog is not None:
                    match = next((item for item in catalog if str(item.get("id", "")).strip() == model_id), None)
                    if match:
                        levels, default = self._extract_metadata(match)
                        source = "provider_metadata" if levels or default else "provider_catalog_no_effort_metadata"
                    else:
                        source = "model_not_in_provider_catalog"

                # Provider metadata wins. Existing trusted config/profile is the
                # fallback for providers whose models catalog omits effort data.
                if not levels:
                    configured_levels = self._levels(model.get("reasoning_efforts"))
                    if not configured_levels and isinstance(model.get("extra"), dict):
                        configured_levels = self._levels(model["extra"].get("reasoning_efforts"))
                    if configured_levels:
                        levels = configured_levels
                        default_value = str(model.get("reasoning_effort_default", "")).strip().lower()
                        default = default_value if default_value in ALLOWED_EFFORTS else None
                        source = "configured_profile"
                if not levels and not default and cached.get("capabilities", {}).get("supported_levels") and error_type:
                    cached["stale"] = True
                    cached["last_refresh_error"] = error_type
                    with self._lock:
                        if generation != self._generation:
                            return self.snapshot()
                        self._entries[key] = cached
                        changed = True
                    continue

                entry = {
                    "provider": provider_name,
                    "model": model_id,
                    "capabilities": {
                        "supported_levels": levels,
                        "default_level": default,
                        "supported": bool(levels),
                    },
                    "source": source,
                    "fetched_at": now,
                    "expires_at": now + self.ttl_seconds,
                    "stale": bool(error_type),
                    **({"last_refresh_error": error_type} if error_type else {}),
                }
                with self._lock:
                    if generation != self._generation:
                        return self.snapshot()
                    self._entries[key] = entry
                    changed = True

            if changed:
                with self._lock:
                    if generation == self._generation:
                        self._save()
            return self.snapshot()
        finally:
            self._refresh_lock.release()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            entries = []
            for item in self._entries.values():
                record = dict(item)
                record["stale"] = bool(record.get("stale")) or now >= self._number(record.get("expires_at"))
                entries.append(record)
            return {
                "items": entries,
                "count": len(entries),
                "ttl_seconds": self.ttl_seconds,
            }

    def storage_info(self) -> dict[str, Any]:
        with self._lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                size = 0
            return {"bytes": size, "items": len(self._entries), "path": self.path.name}

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._generation += 1
            self._entries.clear()
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                return {"cleared": False, "error": type(exc).__name__, **self.storage_info()}
            return {"cleared": True, "bytes": 0, "items": 0}

"""Privacy-conscious token-usage telemetry for OpenAI-compatible model APIs.

Only provider/model identifiers, timing, token counters, and configured cost
estimates are stored. Prompts, completions, API keys, and response bodies are
never written to the usage log.
"""
from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _non_negative_int(value: Any) -> int | None:
    """Accept integer counters (or digit-only JSON/config strings), never truncate floats."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isascii() and normalized.isdecimal():
            try:
                return int(normalized)
            except (ValueError, OverflowError):
                return None
    return None


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _first_present(source: Mapping[str, Any], names: tuple[str, ...]) -> tuple[bool, Any]:
    for name in names:
        if name in source:
            return True, source.get(name)
    return False, None


def _token_count(usage: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    present, value = _first_present(usage, names)
    return _non_negative_int(value) if present else None


class TokenUsageRecorder:
    """Thread-safe JSONL token usage recorder with optional size-based rotation."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        settings = dict(config) if isinstance(config, Mapping) else {}
        enabled = settings.get("enabled", False)
        console_log = settings.get("console_log", True)
        if not isinstance(enabled, bool):
            raise ValueError("observability.token_usage.enabled must be a boolean")
        if not isinstance(console_log, bool):
            raise ValueError("observability.token_usage.console_log must be a boolean")

        raw_path = settings.get("path", "data/token_usage.jsonl")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("observability.token_usage.path must be a non-empty string")
        root = Path(project_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
        candidate = Path(raw_path).expanduser()
        self.path = (candidate if candidate.is_absolute() else root / candidate).resolve()
        self.enabled = enabled
        self.console_log = console_log

        self.max_file_bytes = _non_negative_int(settings.get("max_file_bytes", 10 * 1024 * 1024))
        self.backup_count = _non_negative_int(settings.get("backup_count", 3))
        if self.max_file_bytes is None or self.backup_count is None:
            raise ValueError("token usage log rotation settings must be non-negative integers")
        self._lock = threading.Lock()

    def record(
        self,
        *,
        provider: str,
        model: str,
        status: str,
        usage: Mapping[str, Any] | None,
        elapsed_seconds: float,
        error_type: str | None = None,
        pricing_usd_per_million: Mapping[str, Any] | None = None,
        role_id: str | None = None,
        user_id: str | None = None,
        task_id: str | None = None,
        prefix_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        """Append one metadata-only event. Returns the event, or None when disabled."""
        if not self.enabled:
            return None

        raw_usage = usage if isinstance(usage, Mapping) else {}
        normalized_status = str(status or "unknown").strip().lower() or "unknown"
        input_tokens = (
            _token_count(raw_usage, ("prompt_tokens", "input_tokens"))
            if normalized_status == "success"
            else None
        )
        output_tokens = (
            _token_count(raw_usage, ("completion_tokens", "output_tokens"))
            if normalized_status == "success"
            else None
        )
        usage_available = input_tokens is not None or output_tokens is not None

        details = raw_usage.get("prompt_tokens_details", {})
        details = details if isinstance(details, Mapping) else {}
        hit_present, hit_raw = _first_present(
            raw_usage,
            ("prompt_cache_hit_tokens", "cache_read_input_tokens"),
        )
        if not hit_present and "cached_tokens" in details:
            hit_present, hit_raw = True, details.get("cached_tokens")
        miss_present, miss_raw = _first_present(
            raw_usage,
            ("prompt_cache_miss_tokens",),
        )
        cache_hit_tokens = _non_negative_int(hit_raw) if hit_present else None
        cache_miss_tokens = _non_negative_int(miss_raw) if miss_present else None
        cache_stats_available = (
            cache_hit_tokens is not None or cache_miss_tokens is not None
        )

        if cache_stats_available and input_tokens is not None:
            # Providers may report either hits or misses alone; infer the other
            # only from a valid subset of the reported prompt token count.
            if cache_hit_tokens is None and cache_miss_tokens is not None:
                if cache_miss_tokens <= input_tokens:
                    cache_hit_tokens = input_tokens - cache_miss_tokens
            elif cache_miss_tokens is None and cache_hit_tokens is not None:
                if cache_hit_tokens <= input_tokens:
                    cache_miss_tokens = input_tokens - cache_hit_tokens

            counters_exceed_input = (
                (cache_hit_tokens is not None and cache_hit_tokens > input_tokens)
                or (cache_miss_tokens is not None and cache_miss_tokens > input_tokens)
                or (
                    cache_hit_tokens is not None
                    and cache_miss_tokens is not None
                    and cache_hit_tokens + cache_miss_tokens > input_tokens
                )
            )
            if counters_exceed_input:
                # Bad metrics must not create hit rates above 100% or fake cost savings.
                cache_stats_available = False
                cache_hit_tokens = None
                cache_miss_tokens = None

        total_tokens = (
            _token_count(raw_usage, ("total_tokens",))
            if normalized_status == "success"
            else None
        )
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        cache_hit_rate = None
        if cache_stats_available and input_tokens is not None and input_tokens > 0:
            cache_hit_rate = round((cache_hit_tokens or 0) / input_tokens, 6)

        elapsed = _non_negative_number(elapsed_seconds)
        pricing = pricing_usd_per_million if isinstance(pricing_usd_per_million, Mapping) else {}
        estimated_cost = self._estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit_tokens or 0,
            cache_stats_available=cache_stats_available,
            pricing=pricing,
        )

        event: dict[str, Any] = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": str(provider or "unknown"),
            "model": str(model or "unknown"),
            "role_id": str(role_id)[:80] if role_id else None,
            "user_id": str(user_id)[:128] if user_id else None,
            "task_id": str(task_id)[:128] if task_id else None,
            "prefix_fingerprint": str(prefix_fingerprint)[:64] if prefix_fingerprint else None,
            "status": normalized_status,
            "duration_ms": round(elapsed * 1000, 2) if elapsed is not None else None,
            "error_type": str(error_type)[:120] if error_type else None,
            "usage_available": usage_available,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_stats_available": cache_stats_available,
            "cache_hit_tokens": cache_hit_tokens if cache_stats_available else None,
            "cache_miss_tokens": cache_miss_tokens if cache_stats_available else None,
            "cache_hit_rate": cache_hit_rate,
            "estimated_cost_usd": estimated_cost,
        }
        encoded = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(len(encoded))
            # One append write avoids interleaving partial JSON lines between threads.
            descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)

        if self.console_log:
            self._log_event(event)
        return event

    @staticmethod
    def _estimate_cost(
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        cache_hit_tokens: int,
        cache_stats_available: bool,
        pricing: Mapping[str, Any],
    ) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        input_price = _non_negative_number(pricing.get("input"))
        output_price = _non_negative_number(pricing.get("output"))
        cached_input_price = _non_negative_number(pricing.get("cache_hit_input"))
        if input_price is None or output_price is None:
            return None
        if cache_stats_available and cache_hit_tokens > 0:
            if cached_input_price is None:
                return None
            uncached_tokens = max(0, input_tokens - cache_hit_tokens)
            amount = (
                uncached_tokens * input_price
                + cache_hit_tokens * cached_input_price
                + output_tokens * output_price
            ) / 1_000_000
        else:
            amount = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        return round(amount, 10)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self.max_file_bytes <= 0 or not self.path.exists():
            return
        try:
            if self.path.stat().st_size + incoming_bytes <= self.max_file_bytes:
                return
            if self.backup_count == 0:
                self.path.unlink(missing_ok=True)
                return
            oldest = self.path.with_name(self.path.name + f".{self.backup_count}")
            oldest.unlink(missing_ok=True)
            for index in range(self.backup_count - 1, 0, -1):
                source = self.path.with_name(self.path.name + f".{index}")
                target = self.path.with_name(self.path.name + f".{index + 1}")
                if source.exists():
                    source.replace(target)
            if self.path.exists():
                self.path.replace(self.path.with_name(self.path.name + ".1"))
        except OSError:
            # Rotation is best-effort; a telemetry failure must not affect model work.
            return

    def _log_event(self, event: Mapping[str, Any]) -> None:
        if event.get("usage_available"):
            cache_detail = "cache=unreported"
            if event.get("cache_stats_available"):
                rate = event.get("cache_hit_rate")
                percent = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else "n/a"
                cache_detail = (
                    f"cache_hit={event.get('cache_hit_tokens', 0)} "
                    f"cache_miss={event.get('cache_miss_tokens', 0)} hit_rate={percent}"
                )
            message = (
                f"[TokenUsage] provider={event['provider']} model={event['model']} "
                f"input={event['input_tokens']} output={event['output_tokens']} "
                f"total={event['total_tokens']} {cache_detail} "
                f"duration_ms={event['duration_ms']}"
            )
            if event.get("estimated_cost_usd") is not None:
                message += f" estimated_cost_usd={event['estimated_cost_usd']:.8f}"
        else:
            message = (
                f"[TokenUsage] provider={event['provider']} model={event['model']} "
                f"status={event['status']} usage=unavailable "
                f"duration_ms={event['duration_ms']}"
            )
        print(message, flush=True)


def _empty_summary() -> dict[str, Any]:
    return {
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "usage_available_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_stats_requests": 0,
        "cache_input_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_estimate_requests": 0,
    }


def _add_to_summary(summary: dict[str, Any], event: Mapping[str, Any]) -> None:
    summary["requests"] += 1
    if event.get("status") == "success":
        summary["successful_requests"] += 1
    else:
        summary["failed_requests"] += 1

    if event.get("usage_available"):
        summary["usage_available_requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = _non_negative_int(event.get(key))
            if value is not None:
                summary[key] += value

    if event.get("cache_stats_available"):
        summary["cache_stats_requests"] += 1
        input_tokens = _non_negative_int(event.get("input_tokens"))
        hit_tokens = _non_negative_int(event.get("cache_hit_tokens"))
        miss_tokens = _non_negative_int(event.get("cache_miss_tokens"))
        if input_tokens is not None:
            summary["cache_input_tokens"] += input_tokens
        if hit_tokens is not None:
            summary["cache_hit_tokens"] += hit_tokens
        if miss_tokens is not None:
            summary["cache_miss_tokens"] += miss_tokens

    cost = _non_negative_number(event.get("estimated_cost_usd"))
    if cost is not None:
        summary["estimated_cost_usd"] += cost
        summary["cost_estimate_requests"] += 1


def summarize_usage_log(path: str | Path, *, backup_count: int = 3) -> dict[str, Any]:
    """Summarize current and rotated JSONL logs using token-weighted cache rates."""
    if isinstance(backup_count, bool) or not isinstance(backup_count, int) or backup_count < 0:
        raise ValueError("backup_count must be a non-negative integer")
    source = Path(path).expanduser().resolve()
    summary: dict[str, Any] = {
        "path": str(source),
        "records": 0,
        "malformed_lines": 0,
        "totals": _empty_summary(),
        "by_provider_model": {},
    }
    candidates = [
        source.with_name(source.name + f".{index}")
        for index in range(backup_count, 0, -1)
    ]
    candidates.append(source)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        summary["malformed_lines"] += 1
                        continue
                    if not isinstance(event, dict):
                        summary["malformed_lines"] += 1
                        continue
                    summary["records"] += 1
                    _add_to_summary(summary["totals"], event)
                    group_key = (
                        str(event.get("provider") or "unknown"),
                        str(event.get("model") or "unknown"),
                    )
                    groups = summary["by_provider_model"]
                    display_key = f"{group_key[0]} / {group_key[1]}"
                    group = groups.setdefault(display_key, _empty_summary())
                    _add_to_summary(group, event)
        except OSError:
            summary["malformed_lines"] += 1

    for item in [summary["totals"], *summary["by_provider_model"].values()]:
        cache_input = item["cache_input_tokens"]
        item["cache_hit_rate"] = (
            round(item["cache_hit_tokens"] / cache_input, 6)
            if cache_input > 0
            else None
        )
        item["estimated_cost_usd"] = round(item["estimated_cost_usd"], 10)
    return summary

"""Cancellable, testable timing policy for browser-driven Agent operations."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from typing import Any

from .browser_automation_config import DEFAULT_BROWSER_AUTOMATION_CONFIG


RandomUniform = Callable[[float, float], float]
SleepFunction = Callable[[float], None]
CheckFunction = Callable[[], None]


class OperationTiming:
    """Resolve configurable random delays without coupling them to Playwright.

    Timing values are seconds. A range may be written as {"min": x, "max": y}
    or [x, y]. Injecting random_uniform makes timing tests deterministic.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        fixed_send_interval_seconds: float | None = None,
        random_uniform: RandomUniform | None = None,
    ) -> None:
        default_timing = DEFAULT_BROWSER_AUTOMATION_CONFIG.get("timing", {})
        supplied = dict(config or {})
        self._config = dict(default_timing) if isinstance(default_timing, Mapping) else {}
        self._config.update(supplied)
        self._random_uniform = random_uniform or random.uniform

        self.send_interval_min_seconds, self.send_interval_max_seconds = (
            self._parse_range(
                self._config.get("send_interval_seconds", {"min": 6, "max": 15}),
                "send_interval_seconds",
                default=(6.0, 15.0),
            )
        )
        if fixed_send_interval_seconds is not None:
            fixed = self._parse_number(
                fixed_send_interval_seconds, "min_send_interval_seconds"
            )
            self.send_interval_min_seconds = fixed
            self.send_interval_max_seconds = fixed

        self.retry_wait_seconds = self._parse_number(
            self._config.get("retry_wait_seconds", 20.0),
            "retry_wait_seconds",
        )
        raw_delays = self._config.get("ui_delay_seconds", {})
        if not isinstance(raw_delays, Mapping):
            raise ValueError("ui_delay_seconds 必须是 JSON 对象。")
        self._delay_ranges: dict[str, tuple[float, float]] = {}
        defaults = (
            DEFAULT_BROWSER_AUTOMATION_CONFIG.get("timing", {})
            if isinstance(DEFAULT_BROWSER_AUTOMATION_CONFIG, Mapping)
            else {}
        )
        default_delays = defaults.get("ui_delay_seconds", {}) if isinstance(defaults, Mapping) else {}
        keys = set(default_delays) | set(raw_delays)
        for key in keys:
            value = raw_delays.get(key, default_delays.get(key))
            self._delay_ranges[str(key)] = self._parse_range(
                value, f"ui_delay_seconds.{key}", default=(0.0, 0.0)
            )

    @staticmethod
    def _parse_number(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} 必须是非负数。")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是非负数。") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} 必须是有限的非负数。")
        return result

    @classmethod
    def _parse_range(
        cls,
        value: Any,
        name: str,
        *,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        if value is None:
            low, high = default
        elif isinstance(value, Mapping):
            low = value.get("min", default[0])
            high = value.get("max", default[1])
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
        else:
            raise ValueError(f"{name} 必须使用 {{'min': ..., 'max': ...}} 或二元数组。")
        minimum = cls._parse_number(low, f"{name}.min")
        maximum = cls._parse_number(high, f"{name}.max")
        if maximum < minimum:
            raise ValueError(f"{name}.max 不能小于 {name}.min。")
        return minimum, maximum

    def sample(self, operation: str) -> float:
        """Pick a fresh delay for the requested operation."""
        if operation == "send_interval_seconds":
            low, high = self.send_interval_min_seconds, self.send_interval_max_seconds
        elif operation == "retry_wait_seconds":
            return self.retry_wait_seconds
        else:
            try:
                low, high = self._delay_ranges[operation]
            except KeyError as exc:
                raise KeyError(f"没有配置操作延时：{operation}") from exc
        if high <= low:
            return low
        sampled = self._parse_number(
            self._random_uniform(low, high), f"sampled_delay.{operation}"
        )
        # Defensive clamp for custom random sources and test doubles.
        return min(high, max(low, sampled))

    def wait(
        self,
        operation: str,
        *,
        sleep: SleepFunction,
        check_cancelled: CheckFunction | None = None,
    ) -> float:
        """Sample and perform a cancellable wait via the caller's sleep hook."""
        delay = self.sample(operation)
        if check_cancelled is not None:
            check_cancelled()
        if delay > 0:
            sleep(delay)
        if check_cancelled is not None:
            check_cancelled()
        return delay

    def wait_before_retry(
        self,
        *,
        sleep: SleepFunction,
        check_cancelled: CheckFunction | None = None,
    ) -> float:
        """Wait the fixed retry interval; do not add the normal send interval."""
        return self.wait(
            "retry_wait_seconds",
            sleep=sleep,
            check_cancelled=check_cancelled,
        )

"""Load the checked-in browser automation settings and merge user overrides.

Selectors, XPath fallbacks, user-facing button labels, timeouts, and timing
defaults live in config/browser_automation.json so UI changes do not require
editing browser automation code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


BROWSER_AUTOMATION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "browser_automation.json"
)


def load_browser_automation_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Read browser automation JSON, validating its top-level sections."""
    source = Path(path) if path is not None else BROWSER_AUTOMATION_CONFIG_PATH
    if not source.exists():
        # Keep lightweight library/test usage functional when packaged without
        # optional config files; BrowserModel's final fallbacks remain safe.
        return {"schema_version": 1, "timing": {}, "ui": {}}
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"浏览器自动化配置 JSON 格式错误：{source}") from exc
    except OSError as exc:
        raise ValueError(f"无法读取浏览器自动化配置：{source}") from exc

    if not isinstance(value, dict):
        raise ValueError("browser_automation.json 顶层必须是 JSON 对象。")
    for section in ("timing", "ui"):
        if section in value and not isinstance(value[section], dict):
            raise ValueError(f"browser_automation.json 的 {section} 必须是对象。")
    value.setdefault("schema_version", 1)
    value.setdefault("timing", {})
    value.setdefault("ui", {})
    return value


def deep_merge_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a nested copy of base with user overrides applied."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge_config(previous, value)
        else:
            merged[key] = value
    return merged


DEFAULT_BROWSER_AUTOMATION_CONFIG = load_browser_automation_config()

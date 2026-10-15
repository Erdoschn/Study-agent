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
    """Read browser automation JSON, validating its required config sections."""
    source = Path(path) if path is not None else BROWSER_AUTOMATION_CONFIG_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"缺少浏览器自动化配置文件：{source}。"
            "请从 config/browser_automation.json 恢复此文件。"
        )
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

    timing = value["timing"]
    ui = value["ui"]
    selectors = ui.get("selectors", {})
    labels = ui.get("labels", {})
    timeouts_ms = ui.get("timeouts_ms", {})
    timeouts_seconds = ui.get("timeouts_seconds", {})
    if not isinstance(timing, dict):
        raise ValueError("browser_automation.json 的 timing 必须是对象。")
    for section_name, section in (
        ("ui.selectors", selectors),
        ("ui.labels", labels),
        ("ui.timeouts_ms", timeouts_ms),
        ("ui.timeouts_seconds", timeouts_seconds),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"browser_automation.json 的 {section_name} 必须是对象。")

    required = {
        "ui.selectors": (
            "response", "loading", "input", "chat_message", "body",
            "file_input", "upload_busy", "send_button",
            "rate_limit_retry_xpath", "history_more_icon_path_prefix",
            "history_more_xpath", "copy_button_path_prefix", "copy_item_xpath",
            "history_session_link_template", "history_row_parent_xpath",
            "svg_path_button_template", "cookie_banner",
            "disabled_button_class", "button_container", "deep_thinking",
        ),
        "ui.labels": (
            "new_chat", "more", "delete_chat", "stop_generating",
            "continue_generating", "rate_limit", "resend", "send",
            "send_exclude", "cookie_accept", "attachment", "deep_thinking",
        ),
        "ui.timeouts_ms": (
            "attachment_select", "send_action", "click_action", "copy_click",
        ),
        "ui.timeouts_seconds": (
            "attachment_upload", "send_confirmation",
            "send_button_ready", "rate_limit_retry_button_wait",
        ),
    }
    sections = {
        "ui.selectors": selectors,
        "ui.labels": labels,
        "ui.timeouts_ms": timeouts_ms,
        "ui.timeouts_seconds": timeouts_seconds,
    }
    for section_name, keys in required.items():
        section = sections[section_name]
        missing = [key for key in keys if key not in section]
        if missing:
            raise ValueError(
                f"browser_automation.json 缺少配置项 {section_name}: "
                + ", ".join(missing)
            )

    if "send_interval_seconds" not in timing or "retry_wait_seconds" not in timing:
        raise ValueError(
            "browser_automation.json 的 timing 必须配置 "
            "send_interval_seconds 和 retry_wait_seconds。"
        )
    if not isinstance(timing.get("ui_delay_seconds"), dict):
        raise ValueError("browser_automation.json 的 timing.ui_delay_seconds 必须是对象。")

    value["ui"] = {
        "selectors": selectors,
        "labels": labels,
        "timeouts_ms": timeouts_ms,
        "timeouts_seconds": timeouts_seconds,
    }
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

import os
import re
import sys
import time
from io import UnsupportedOperation
from pathlib import Path
from threading import Event
from typing import Any

from .__debug__ import debug
from .reasoner import ModelClient
from .prompt_config import get_prompt
from .storage_security import ensure_private_directory
from .cancellation import RunCancelled, raise_if_cancelled
from .errors import AttachmentUploadError
from .browser_automation_config import (
    DEFAULT_BROWSER_AUTOMATION_CONFIG,
    deep_merge_config,
)
from .operation_timing import OperationTiming


_DEFAULT_BROWSER_UI = DEFAULT_BROWSER_AUTOMATION_CONFIG.get("ui", {})
if not isinstance(_DEFAULT_BROWSER_UI, dict):
    _DEFAULT_BROWSER_UI = {}
_DEFAULT_UI_SELECTORS = _DEFAULT_BROWSER_UI.get("selectors", {})
if not isinstance(_DEFAULT_UI_SELECTORS, dict):
    _DEFAULT_UI_SELECTORS = {}
_DEFAULT_UI_LABELS = _DEFAULT_BROWSER_UI.get("labels", {})
if not isinstance(_DEFAULT_UI_LABELS, dict):
    _DEFAULT_UI_LABELS = {}


def _config_sequence(config: dict[str, Any], key: str, fallback=()) -> tuple[str, ...]:
    value = config.get(key) if isinstance(config, dict) else None
    if isinstance(value, (list, tuple)):
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if normalized:
            return normalized
    return tuple(fallback)


def _config_string(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key) if isinstance(config, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _config_number(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key) if isinstance(config, dict) else None
    if isinstance(value, bool):
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number >= 0 and number < float("inf") else fallback


class BrowserModel(ModelClient):
    """LLM client backed by a normal local browser session.

    This adapter only drives the visible DeepSeek web UI. It does not call
    DeepSeek's private HTTP endpoints or manufacture browser headers/cookies.
    The first run opens a persistent local profile so the user can log in
    manually; later calls reuse that browser session.
    """

    DEFAULT_URL = "https://chat.deepseek.com/"
    # UI locators and button labels are sourced from config/browser_automation.json.
    DEFAULT_RESPONSE_SELECTORS = _config_sequence(_DEFAULT_UI_SELECTORS, "response")
    DEFAULT_LOADING_SELECTORS = _config_sequence(_DEFAULT_UI_SELECTORS, "loading")
    INPUT_SELECTORS = _config_sequence(_DEFAULT_UI_SELECTORS, "input")
    CHAT_MESSAGE_SELECTOR = _config_string(_DEFAULT_UI_SELECTORS, "chat_message", "")
    BODY_SELECTOR = _config_string(_DEFAULT_UI_SELECTORS, "body", "")
    FILE_INPUT_SELECTOR = _config_string(_DEFAULT_UI_SELECTORS, "file_input", "")
    UPLOAD_BUSY_SELECTORS = _config_sequence(_DEFAULT_UI_SELECTORS, "upload_busy")
    BUTTON_CONTAINER_SELECTOR = _config_string(
        _DEFAULT_UI_SELECTORS, "button_container", ""
    )
    HISTORY_SESSION_LINK_TEMPLATE = _config_string(
        _DEFAULT_UI_SELECTORS, "history_session_link_template", ""
    )
    HISTORY_ROW_PARENT_XPATH = _config_string(
        _DEFAULT_UI_SELECTORS, "history_row_parent_xpath", ""
    )
    COPY_ITEM_XPATH = _config_string(
        _DEFAULT_UI_SELECTORS, "copy_item_xpath", ""
    )
    SVG_PATH_BUTTON_TEMPLATE = _config_string(
        _DEFAULT_UI_SELECTORS, "svg_path_button_template", ""
    )
    COOKIE_BANNER_SELECTOR = _config_string(
        _DEFAULT_UI_SELECTORS, "cookie_banner", ""
    )
    DISABLED_BUTTON_CLASS = _config_string(
        _DEFAULT_UI_SELECTORS, "disabled_button_class", ""
    )
    DEFAULT_NEW_CHAT_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "new_chat")
    DEFAULT_MORE_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "more")
    DEFAULT_MORE_ICON_PATH_PREFIX = _config_string(
        _DEFAULT_UI_SELECTORS, "history_more_icon_path_prefix", ""
    )
    DEFAULT_MORE_XPATHS = _config_sequence(_DEFAULT_UI_SELECTORS, "history_more_xpath")
    DEFAULT_DELETE_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "delete_chat")
    DEFAULT_STOP_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "stop_generating")
    DEFAULT_CONTINUE_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "continue_generating")
    DEFAULT_RATE_LIMIT_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "rate_limit")
    DEFAULT_RESEND_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "resend")
    DEFAULT_SEND_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "send")
    DEFAULT_SEND_EXCLUDE_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "send_exclude")
    DEFAULT_SEND_BUTTON_SELECTORS = _config_sequence(_DEFAULT_UI_SELECTORS, "send_button")
    DEFAULT_RATE_LIMIT_RETRY_XPATHS = _config_sequence(
        _DEFAULT_UI_SELECTORS, "rate_limit_retry_xpath"
    )
    DEFAULT_COOKIE_ACCEPT_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "cookie_accept")
    DEFAULT_ATTACHMENT_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "attachment")
    DEFAULT_DEEP_THINKING_LABELS = _config_sequence(_DEFAULT_UI_LABELS, "deep_thinking")
    DEFAULT_DEEP_THINKING_SELECTORS = _config_sequence(_DEFAULT_UI_SELECTORS, "deep_thinking")
    DEFAULT_COPY_PATH_PREFIX = _config_string(
        _DEFAULT_UI_SELECTORS, "copy_button_path_prefix", ""
    )

    MAX_CONTINUE_GENERATIONS = 3
    MAX_RATE_LIMIT_RETRIES = 3

    _DEFAULT_TIMEOUTS_MS = _DEFAULT_BROWSER_UI.get("timeouts_ms", {})
    if not isinstance(_DEFAULT_TIMEOUTS_MS, dict):
        _DEFAULT_TIMEOUTS_MS = {}
    _DEFAULT_TIMEOUTS_SECONDS = _DEFAULT_BROWSER_UI.get("timeouts_seconds", {})
    if not isinstance(_DEFAULT_TIMEOUTS_SECONDS, dict):
        _DEFAULT_TIMEOUTS_SECONDS = {}
    ATTACHMENT_SELECT_TIMEOUT_MS = int(_config_number(_DEFAULT_TIMEOUTS_MS, "attachment_select", 120000))
    ATTACHMENT_UPLOAD_TIMEOUT_SECONDS = _config_number(_DEFAULT_TIMEOUTS_SECONDS, "attachment_upload", 120.0)
    SEND_CONFIRMATION_TIMEOUT_SECONDS = _config_number(_DEFAULT_TIMEOUTS_SECONDS, "send_confirmation", 2.0)
    SEND_BUTTON_READY_TIMEOUT_SECONDS = _config_number(_DEFAULT_TIMEOUTS_SECONDS, "send_button_ready", 120.0)
    RATE_LIMIT_RETRY_BUTTON_WAIT_SECONDS = _config_number(_DEFAULT_TIMEOUTS_SECONDS, "rate_limit_retry_button_wait", 5.0)
    COPY_CLICK_TIMEOUT_MS = int(_config_number(_DEFAULT_TIMEOUTS_MS, "copy_click", 1200))
    SEND_ACTION_TIMEOUT_MS = int(_config_number(_DEFAULT_TIMEOUTS_MS, "send_action", 1500))
    CLICK_ACTION_TIMEOUT_MS = int(_config_number(_DEFAULT_TIMEOUTS_MS, "click_action", 1500))

    def __init__(
        self,
        model: str = "deepseek-web",
        *,
        url: str | None = None,
        user_data_dir: str | None = None,
        browser_channel: str | None = None,
        timeout: int = 180,
        response_selectors: tuple[str, ...] | None = None,
        loading_selectors: tuple[str, ...] | None = None,
        session_pause_seconds: float = 1.5,
        cleanup_pause_seconds: float = 3.0,
        post_cleanup_pause_seconds: float = 1.5,
        cleanup_after_generate: bool = True,
        reuse_chat: bool = False,
        min_send_interval_seconds: float | None = None,
        timing_config: dict[str, Any] | None = None,
        ui_config: dict[str, Any] | None = None,
        poll_interval: float = 0.5,
        stable_seconds: float = 1.2,
        invalid_json_grace_seconds: float = 3.0,
        debug_mode: bool = False,
        cancellation_event: Event | None = None,
    ):
        self.model = model
        self.url = url or os.getenv("STUDY_AGENT_WEB_URL", self.DEFAULT_URL)
        profile = user_data_dir or os.getenv(
            "STUDY_AGENT_WEB_PROFILE",
            str(Path(".study-agent-browser").resolve()),
        )
        self.user_data_dir = str(Path(profile).expanduser())
        self.browser_channel = browser_channel or os.getenv(
            "STUDY_AGENT_WEB_BROWSER",
            "msedge",
        )
        self.timeout = max(1, int(timeout))
        if ui_config is not None and not isinstance(ui_config, dict):
            raise ValueError("ui_config 必须是对象。")
        self.ui_config = deep_merge_config(_DEFAULT_BROWSER_UI, ui_config or {})
        selectors = self.ui_config.get("selectors", {})
        labels = self.ui_config.get("labels", {})
        timeouts_ms = self.ui_config.get("timeouts_ms", {})
        timeouts_seconds = self.ui_config.get("timeouts_seconds", {})
        if not isinstance(selectors, dict):
            selectors = {}
        if not isinstance(labels, dict):
            labels = {}
        if not isinstance(timeouts_ms, dict):
            timeouts_ms = {}
        if not isinstance(timeouts_seconds, dict):
            timeouts_seconds = {}

        # A provider may override its own UI locators and labels without code changes.
        for attribute, key in (
            ("DEFAULT_RESPONSE_SELECTORS", "response"),
            ("DEFAULT_LOADING_SELECTORS", "loading"),
            ("INPUT_SELECTORS", "input"),
            ("UPLOAD_BUSY_SELECTORS", "upload_busy"),
            ("DEFAULT_NEW_CHAT_LABELS", "new_chat"),
            ("DEFAULT_MORE_LABELS", "more"),
            ("DEFAULT_MORE_XPATHS", "history_more_xpath"),
            ("DEFAULT_DELETE_LABELS", "delete_chat"),
            ("DEFAULT_STOP_LABELS", "stop_generating"),
            ("DEFAULT_CONTINUE_LABELS", "continue_generating"),
            ("DEFAULT_RATE_LIMIT_LABELS", "rate_limit"),
            ("DEFAULT_RESEND_LABELS", "resend"),
            ("DEFAULT_SEND_LABELS", "send"),
            ("DEFAULT_SEND_EXCLUDE_LABELS", "send_exclude"),
            ("DEFAULT_SEND_BUTTON_SELECTORS", "send_button"),
            ("DEFAULT_RATE_LIMIT_RETRY_XPATHS", "rate_limit_retry_xpath"),
            ("DEFAULT_COOKIE_ACCEPT_LABELS", "cookie_accept"),
            ("DEFAULT_ATTACHMENT_LABELS", "attachment"),
            ("DEFAULT_DEEP_THINKING_LABELS", "deep_thinking"),
            ("DEFAULT_DEEP_THINKING_SELECTORS", "deep_thinking"),
        ):
            fallback = getattr(type(self), attribute)
            setting_container = labels if attribute in {
                "DEFAULT_NEW_CHAT_LABELS", "DEFAULT_MORE_LABELS",
                "DEFAULT_DELETE_LABELS", "DEFAULT_STOP_LABELS",
                "DEFAULT_CONTINUE_LABELS", "DEFAULT_RATE_LIMIT_LABELS",
                "DEFAULT_RESEND_LABELS", "DEFAULT_SEND_LABELS",
                "DEFAULT_SEND_EXCLUDE_LABELS", "DEFAULT_COOKIE_ACCEPT_LABELS",
                "DEFAULT_ATTACHMENT_LABELS", "DEFAULT_DEEP_THINKING_LABELS",
            } else selectors
            setattr(self, attribute, _config_sequence(setting_container, key, fallback))

        for attribute, key in (
            ("CHAT_MESSAGE_SELECTOR", "chat_message"),
            ("BODY_SELECTOR", "body"),
            ("FILE_INPUT_SELECTOR", "file_input"),
            ("BUTTON_CONTAINER_SELECTOR", "button_container"),
            ("HISTORY_SESSION_LINK_TEMPLATE", "history_session_link_template"),
            ("HISTORY_ROW_PARENT_XPATH", "history_row_parent_xpath"),
            ("COPY_ITEM_XPATH", "copy_item_xpath"),
            ("SVG_PATH_BUTTON_TEMPLATE", "svg_path_button_template"),
            ("COOKIE_BANNER_SELECTOR", "cookie_banner"),
            ("DISABLED_BUTTON_CLASS", "disabled_button_class"),
            ("DEFAULT_MORE_ICON_PATH_PREFIX", "history_more_icon_path_prefix"),
            ("DEFAULT_COPY_PATH_PREFIX", "copy_button_path_prefix"),
        ):
            setattr(
                self,
                attribute,
                _config_string(selectors, key, getattr(type(self), attribute)),
            )

        self.ATTACHMENT_SELECT_TIMEOUT_MS = max(
            1, int(_config_number(timeouts_ms, "attachment_select", type(self).ATTACHMENT_SELECT_TIMEOUT_MS))
        )
        self.COPY_CLICK_TIMEOUT_MS = max(
            1, int(_config_number(timeouts_ms, "copy_click", type(self).COPY_CLICK_TIMEOUT_MS))
        )
        self.SEND_ACTION_TIMEOUT_MS = max(
            1, int(_config_number(timeouts_ms, "send_action", type(self).SEND_ACTION_TIMEOUT_MS))
        )
        self.CLICK_ACTION_TIMEOUT_MS = max(
            1, int(_config_number(timeouts_ms, "click_action", type(self).CLICK_ACTION_TIMEOUT_MS))
        )
        self.ATTACHMENT_UPLOAD_TIMEOUT_SECONDS = _config_number(
            timeouts_seconds, "attachment_upload", type(self).ATTACHMENT_UPLOAD_TIMEOUT_SECONDS
        )
        self.SEND_CONFIRMATION_TIMEOUT_SECONDS = _config_number(
            timeouts_seconds, "send_confirmation", type(self).SEND_CONFIRMATION_TIMEOUT_SECONDS
        )
        self.SEND_BUTTON_READY_TIMEOUT_SECONDS = _config_number(
            timeouts_seconds, "send_button_ready", type(self).SEND_BUTTON_READY_TIMEOUT_SECONDS
        )
        self.RATE_LIMIT_RETRY_BUTTON_WAIT_SECONDS = _config_number(
            timeouts_seconds, "rate_limit_retry_button_wait", type(self).RATE_LIMIT_RETRY_BUTTON_WAIT_SECONDS
        )
        self.response_selectors = (
            tuple(str(value) for value in response_selectors if str(value).strip())
            if response_selectors is not None
            else self.DEFAULT_RESPONSE_SELECTORS
        )
        self.loading_selectors = (
            tuple(str(value) for value in loading_selectors if str(value).strip())
            if loading_selectors is not None
            else self.DEFAULT_LOADING_SELECTORS
        )
        self.session_pause_seconds = max(0.0, float(session_pause_seconds))
        self.cleanup_pause_seconds = max(0.0, float(cleanup_pause_seconds))
        self.post_cleanup_pause_seconds = max(0.0, float(post_cleanup_pause_seconds))
        self.reuse_chat = bool(reuse_chat)
        self.operation_timing = OperationTiming(
            timing_config,
            fixed_send_interval_seconds=min_send_interval_seconds,
        )
        self.min_send_interval_seconds = self.operation_timing.send_interval_min_seconds
        self.max_send_interval_seconds = self.operation_timing.send_interval_max_seconds
        self.cleanup_after_generate = bool(cleanup_after_generate)
        self.poll_interval = max(0.1, float(poll_interval))
        self.stable_seconds = max(0.3, float(stable_seconds))
        self.invalid_json_grace_seconds = max(
            self.stable_seconds, float(invalid_json_grace_seconds)
        )
        self.debug_mode = bool(debug_mode)
        self.cancellation_event = cancellation_event
        self._created_edge_window_handles: set[int] = set()
        self._json_mode_active = False
        self._call_role = "general"

        self._playwright = None
        self._context = None
        self._page = None
        self._pending_prompt: str | None = None
        self._submission_baseline_message_count: int | None = None
        self._chat_initialized = False
        self._chat_reset_count = 0
        self._last_send_monotonic: float | None = None
        self._attachment_session_active = False
        self._attachment_chat_started = False
        self._attachment_uploaded = False
        self._attachment_current_role: str | None = None
        self._attachment_files: tuple[dict[str, Any], ...] = ()


    def begin_attachment_session(self, files) -> None:
        """Attach original file bytes in the visible DeepSeek chat for one Agent run.

        File payloads use Playwright's {name, mimeType, buffer} form. No document
        parsing or text extraction occurs in Study Agent.
        """
        normalized = []
        for item in files or ():
            if not isinstance(item, dict):
                raise TypeError("附件必须是包含 name/mimeType/buffer 的对象。")
            name = str(item.get("name") or "").strip()
            mime_type = str(item.get("mimeType") or "application/octet-stream")
            content = item.get("buffer")
            if not name or not isinstance(content, (bytes, bytearray, memoryview)):
                raise ValueError("附件缺少文件名或原始二进制内容。")
            if (
                "/" in name
                or chr(92) in name
                or any(ord(char) < 32 or ord(char) == 127 for char in name)
            ):
                raise ValueError("附件文件名包含路径或不允许的控制字符。")
            if len(name) > 240:
                raise ValueError("附件文件名过长（上限 240 个字符）。")
            normalized.append({
                "name": name,
                "mimeType": mime_type,
                "buffer": bytes(content),
            })
        if not normalized:
            raise ValueError("附件会话至少需要一个文件。")
        if self._attachment_session_active:
            raise RuntimeError("当前 DeepSeek Web 会话已有未结束的附件任务。")
        self._attachment_files = tuple(normalized)
        self._attachment_session_active = True
        self._attachment_chat_started = False
        self._attachment_uploaded = False
        self._attachment_current_role = None
        debug.log(
            "BrowserModel",
            "ATTACHMENT SESSION BEGIN → "
            f"count={len(normalized)}, total_bytes={sum(len(item['buffer']) for item in normalized)}",
        )

    def end_attachment_session(self) -> None:
        """Best-effort delete the temporary DeepSeek chat that contains uploaded files."""
        if not self._attachment_session_active:
            return
        original_event = self.cancellation_event
        try:
            # Cleanup must still run if the user cancelled the model request.
            self.cancellation_event = None
            page = self._page
            if (
                page is not None
                and not page.is_closed()
                and self._attachment_chat_started
            ):
                self._cleanup_current_chat(page)
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"ATTACHMENT CLEANUP WARNING → {type(exc).__name__}: {exc}",
            )
        finally:
            self.cancellation_event = original_event
            self._attachment_session_active = False
            self._attachment_chat_started = False
            self._attachment_uploaded = False
            self._attachment_current_role = None
            self._attachment_files = ()
            self._chat_initialized = False
            debug.log("BrowserModel", "ATTACHMENT SESSION END")

    def cancel(self) -> None:
        """Signal the current browser operation to stop as soon as possible."""
        if self.cancellation_event is not None:
            self.cancellation_event.set()
        self._chat_initialized = False

    def reset_cancellation(self) -> None:
        if self.cancellation_event is not None:
            self.cancellation_event.clear()

    def _check_cancelled(self) -> None:
        try:
            raise_if_cancelled(self.cancellation_event)
        except RunCancelled:
            self._chat_initialized = False
            raise

    def _wait_random(self, operation: str) -> float:
        """Apply a configured, cancellable random delay at an operation boundary."""
        delay = self.operation_timing.sample(operation)
        if delay > 0:
            self._check_cancelled()
            debug.log(
                "BrowserModel",
                f"OPERATION WAIT → operation={operation}, delay={delay:.3f}s",
            )
            self._sleep(delay)
        self._check_cancelled()
        return delay

    def _sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        self._check_cancelled()
        if seconds <= 0:
            return
        event = self.cancellation_event
        if event is not None:
            if event.wait(seconds):
                raise RunCancelled("Coder 任务已被用户中止。")
        else:
            time.sleep(seconds)


    def prepare_browser(self) -> None:
        """Open this model's browser profile without sending a model prompt.

        Authentication remains manual. This is used by API shells so the user
        can log into the correct isolated browser profile before an Agent run.
        """
        self._check_cancelled()
        page = self._ensure_page()
        self._dismiss_cookie_banner(page)
        # DeepSeek renders its chat input asynchronously after DOMContentLoaded.
        # Poll briefly instead of misclassifying a logged-in session as logged out.
        input_ready = self._wait_for_chat_input(page, timeout=8.0)
        if not input_ready:
            debug.log(
                "BrowserModel",
                "PREWARM LOGIN → chat input not detected within 8s",
            )
        # The user explicitly wants the browser kept out of the foreground even
        # while login/input detection is incomplete. Do not confuse "not logged
        # in" with "do not attempt window placement".
        self._send_browser_window_to_back(page)
        debug.log(
            "BrowserModel",
            f"BROWSER READY → profile={self.user_data_dir}, url={self.url}, "
            f"chat_input_ready={input_ready}",
        )

    @staticmethod
    def _infer_session_role(system_prompt: str) -> str:
        """Classify only known Study Agent roles for strict context isolation."""
        text = str(system_prompt or "").casefold()
        if "teaching validator" in text or "校验器" in text:
            return "answer_audit"
        if "teacher" in text or "教学引擎" in text:
            return "answer_generation"
        if "task analyzer" in text or "任务分析器" in text:
            return "task_analysis"
        if "assessment generator" in text or "测评题生成器" in text or "assessmentgenerator" in text:
            return "assessment_generation"
        if "reasoner" in text or "决策器" in text:
            return "reasoning"
        return "general"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        deep_thinking: bool | None = None,
    ) -> str:
        self._check_cancelled()
        # Direct Coder/legacy callers keep the previous default. Study Agent
        # passes the TaskAnalyzer's boolean explicitly on every role call.
        requested_deep_thinking = True if deep_thinking is None else bool(deep_thinking)
        debug.log(
            "BrowserModel",
            f"THINKING MODE REQUEST → {'deep' if requested_deep_thinking else 'standard'}",
        )
        if reasoning_effort is not None:
            debug.log(
                "BrowserModel",
                f"REASONING EFFORT HINT → {reasoning_effort}; visible DeepThink state is controlled separately",
            )
        prompt = self._build_prompt(system_prompt, user_prompt, json_mode=json_mode)
        call_role = self._infer_session_role(system_prompt)
        role_requires_isolation = call_role in {"answer_generation", "answer_audit"}
        isolate_this_call = self._attachment_session_active or role_requires_isolation
        debug.log(
            "BrowserModel",
            f"SESSION POLICY → role={call_role}, isolated={isolate_this_call}, "
            f"attachments={len(self._attachment_files) if self._attachment_session_active else 0}",
        )
        page = self._ensure_page()
        self._check_cancelled()

        debug.log(
            "BrowserModel",
            f"REQUEST → model={self.model}, chars={len(prompt)}, url={self.url}",
        )

        self._ensure_logged_in(page)
        self._check_cancelled()
        self._dismiss_cookie_banner(page)
        self._json_mode_active = bool(json_mode)
        self._call_role = call_role
        self._response_binding_fallback_logged = False
        # Restore a previously minimized browser if necessary, then keep it
        # behind other windows. Playwright can interact with a normal window
        # in the background, but not reliably with a minimized one.
        self._send_browser_window_to_back(page)
        if self._attachment_session_active:
            # Keep a single attachment-backed chat while the Agent is making
            # consecutive calls as the same role (e.g. Reasoner's observe/act
            # loop). On role changes, delete the prior temporary chat and start
            # a role-specific one. Teacher and Validator are isolated per call
            # so the audit never shares the answer-generation conversation.
            must_start_new_chat = (
                not self._attachment_chat_started
                or self._attachment_current_role != call_role
                or role_requires_isolation
            )
            if self._attachment_chat_started and must_start_new_chat:
                try:
                    self._cleanup_current_chat(page)
                finally:
                    self._attachment_chat_started = False
                    self._attachment_uploaded = False
                    self._attachment_current_role = None
                    self._chat_initialized = False
            if must_start_new_chat:
                self._start_fresh_chat(page, force=True)
                self._chat_initialized = True
                self._attachment_current_role = call_role
                self._attachment_chat_started = True
                self._attachment_uploaded = False
            if not self._attachment_uploaded:
                self._upload_attachments(page, self._attachment_files)
                self._attachment_uploaded = True
            session_id = self._session_id_from_url(getattr(page, "url", ""))
            debug.log(
                "BrowserModel",
                f"ATTACHMENT CHAT → role={call_role}, "
                f"reused={not must_start_new_chat}, session={session_id or 'new/undisclosed'}",
            )
        elif role_requires_isolation:
            # Answer generation and answer auditing must never share history,
            # even when no uploaded file is present.
            self._start_fresh_chat(page, force=True)
            self._chat_initialized = True
            session_id = self._session_id_from_url(getattr(page, "url", ""))
            debug.log(
                "BrowserModel",
                f"ISOLATED CHAT → role={call_role}, session={session_id or 'new/undisclosed'}",
            )
        elif not self.reuse_chat or not self._chat_initialized:
            self._start_fresh_chat(page)
            self._chat_initialized = True
        completed = False
        try:
            before_snapshot = self._response_snapshot(page)
            self._pending_prompt = prompt
            self._submission_baseline_message_count = self._visible_chat_message_count(page)
            self._send_prompt_for_mode(page, prompt, requested_deep_thinking)
            answer = self._wait_for_response(page, before_snapshot)
            self._check_cancelled()

            # Continue when DeepSeek explicitly stops a long answer.
            for continuation in range(self.MAX_CONTINUE_GENERATIONS):
                if not self._continue_generation_if_available(page):
                    break
                debug.log(
                    "BrowserModel",
                    f"CONTINUE GENERATION → round={continuation + 1}/"
                    f"{self.MAX_CONTINUE_GENERATIONS}",
                )
                before_snapshot = self._response_snapshot(page)
                answer = self._wait_for_response(page, before_snapshot)
                self._check_cancelled()

            self._dismiss_cookie_banner(page)
            markdown = self._copy_latest_response_markdown(page)
            self._check_cancelled()
            if markdown:
                if json_mode:
                    extracted = (
                        self._extract_json_object(markdown)
                        if call_role == "reasoning"
                        else self._extract_any_json_object(markdown)
                    )
                    if extracted:
                        answer = extracted
                    else:
                        debug.log(
                            "BrowserModel",
                            f"COPY SKIP → response did not contain a valid JSON object for role={call_role}",
                        )
                else:
                    answer = markdown
            if not answer:
                raise RuntimeError("DeepSeek Web 返回为空。")

            if json_mode and self._needs_json_recovery(answer, role=call_role):
                completed = True
                answer = self._recover_json_response(page, answer, deep_thinking=requested_deep_thinking)

            completed = True
            debug.log(
                "BrowserModel",
                f"SUCCESS → model={self.model}, chars={len(answer)}",
            )
            return answer
        finally:
            self._pending_prompt = None
            self._submission_baseline_message_count = None
            self._json_mode_active = False
            # In attachment mode, ordinary same-role calls keep their
            # chat so Reasoner's multi-step loop can retain context without
            # re-uploading the original file. Isolated answer/audit calls are
            # deleted after each invocation; role switches clean up the prior
            # role chat before opening another one.
            if self._attachment_session_active and role_requires_isolation:
                try:
                    self._cleanup_current_chat(page)
                finally:
                    self._chat_initialized = False
                    self._attachment_chat_started = False
                    self._attachment_uploaded = False
                    self._attachment_current_role = None
            elif (
                not self._attachment_session_active
                and (role_requires_isolation or (completed and self.cleanup_after_generate))
            ):
                try:
                    self._cleanup_current_chat(page)
                finally:
                    self._chat_initialized = False
            # Leave the browser out of the user's way after UI work, including
            # failed requests, without minimizing it and breaking the next run.
            self._send_browser_window_to_back(page)

    @classmethod
    def _extract_json_object(cls, answer: str) -> str:
        """Extract the last complete Agent JSON object outside quoted text."""
        import json

        text = str(answer or "").strip()
        if not text:
            return ""

        allowed_actions = {
            "SEARCH", "CALCULATE", "VERIFY", "ASSESS", "ANSWER", "STOP",
            "PLAN", "LIST_FILES", "READ_FILE", "WRITE_FILE", "WRITE_NOTEBOOK",
            "PATCH_FILE", "PATCH_NOTEBOOK", "CREATE_TEST", "RUN_PYTHON", "RUN_PYTEST", "RUN_PROJECT", "ASK_STUDY_AGENT", "ASK_USER",
            "READ_DIFF", "VERIFY_GOAL", "FINISH", "NEW_CHAT",
        }

        decoder = json.JSONDecoder()
        last = ""
        in_quote = False
        escaped = False

        for index, char in enumerate(text):
            if in_quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_quote = False
                continue

            if char == '"':
                in_quote = True
                continue

            if char != "{":
                continue

            try:
                data, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue

            if (
                isinstance(data, dict)
                and str(data.get("action", "")).upper() in allowed_actions
            ):
                last = text[index:index + end].strip()

        return last

    @classmethod
    def _extract_any_json_object(cls, answer: str) -> str:
        """Extract a JSON object for role-specific schemas, not just Agent actions."""
        import json

        text = str(answer or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return text
        except (TypeError, ValueError):
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                data, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return text[index:index + end].strip()
        return ""

    @classmethod
    def _is_json_object(cls, answer: str) -> bool:
        return bool(cls._extract_any_json_object(answer))

    @classmethod
    def _needs_json_recovery(cls, answer: str, *, role: str = "reasoning") -> bool:
        """Validate structured output against the contract for its actual Agent role."""
        import json

        text = str(answer or "").strip()
        candidate = (
            cls._extract_json_object(text) or text
            if role == "reasoning"
            else cls._extract_any_json_object(text)
        )
        try:
            data = json.loads(candidate)
        except (TypeError, ValueError):
            return True
        if not isinstance(data, dict):
            return True
        if role == "answer_audit":
            return (
                str(data.get("status", "")).upper() not in {"PASS", "REVISE", "UNCERTAIN"}
                or not isinstance(data.get("claims"), list)
                or not isinstance(data.get("summary"), str)
            )
        if role == "task_analysis":
            return not any(key in data for key in ("task_type", "domain", "goal"))
        if role not in {"reasoning", "general"}:
            return False
        return str(data.get("action", "")).upper() not in {
            "SEARCH", "CALCULATE", "VERIFY", "ASSESS", "ANSWER", "STOP",
            "PLAN", "LIST_FILES", "READ_FILE", "WRITE_FILE", "WRITE_NOTEBOOK",
            "PATCH_FILE", "PATCH_NOTEBOOK", "CREATE_TEST", "RUN_PYTHON", "RUN_PYTEST", "RUN_PROJECT",
            "ASK_STUDY_AGENT", "ASK_USER", "READ_DIFF", "VERIFY_GOAL", "FINISH", "NEW_CHAT",
        }

    @staticmethod
    def _build_json_recovery_prompt(
        previous_answer: str,
        *,
        role: str = "reasoning",
    ) -> str:
        """Repair malformed JSON without changing non-Reasoner roles to the action schema."""
        previous = str(previous_answer or "")[:16_000]
        if role == "reasoning":
            contract = get_prompt("web_model.json_recovery")
            repair_rule = get_prompt("web_model.templates.agent_json_recovery_rule")
        else:
            contract = get_prompt("web_model.templates.recovery_role_json_contract")
            repair_rule = get_prompt("web_model.templates.recovery_role_json_rule")
        wrapper = get_prompt(
            "web_model.templates.recovery_untrusted_wrapper"
        ).replace("{{previous_output}}", previous)
        return "\n\n".join((contract, repair_rule, wrapper))

    def _recover_json_response(
        self,
        page,
        previous_answer: str,
        *,
        deep_thinking: bool | None = None,
    ) -> str:
        """Repair JSON according to the current Agent role's expected schema."""
        role = getattr(self, "_call_role", "reasoning")
        if role == "general":
            # Preserve the historical default: unknown JSON-mode callers use
            # the Agent action schema unless they identify a role-specific schema.
            role = "reasoning"
        recovery_prompt = self._build_json_recovery_prompt(previous_answer, role=role)
        debug.log(
            "BrowserModel",
            f"JSON RECOVERY → previous_chars={len(str(previous_answer or ''))}",
        )
        try:
            before_snapshot = self._response_snapshot(page)
            self._pending_prompt = recovery_prompt
            self._submission_baseline_message_count = self._visible_chat_message_count(page)
            self._send_prompt_for_mode(
                page,
                recovery_prompt,
                True if deep_thinking is None else bool(deep_thinking),
            )
            answer = self._wait_for_response(page, before_snapshot)
            markdown = self._copy_latest_response_markdown(page)
            if markdown:
                extracted = (
                    self._extract_json_object(markdown)
                    if role == "reasoning"
                    else self._extract_any_json_object(markdown)
                )
                if extracted:
                    answer = extracted
            if self._needs_json_recovery(answer, role=role):
                preview = " ".join(str(answer or "")[:240].split())
                debug.log(
                    "BrowserModel",
                    f"JSON RECOVERY → invalid response for role={role}",
                )
                if role == "reasoning":
                    return self._json_stop_response(
                        "DeepSeek Web 的原始响应和一次 JSON 修复响应都不完整或无效；"
                        "为避免执行猜测出来的文件操作，本轮已安全停止。"
                        + (f" 响应片段：{preview}" if preview else "")
                    )
                raise RuntimeError(
                    f"DeepSeek Web 为角色 {role} 返回的 JSON 在修复后仍不符合角色 schema；"
                    "请检查模型输出。"
                )
        except RunCancelled:
            raise
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"JSON RECOVERY → failed for role={role} ({type(exc).__name__})",
            )
            if role == "reasoning":
                return self._json_stop_response(
                    f"DeepSeek Web 的 JSON 修复失败（{type(exc).__name__}）；"
                    "为避免执行猜测出来的操作，本轮已安全停止。"
                )
            raise RuntimeError(
                f"DeepSeek Web 无法为角色 {role} 生成符合 schema 的 JSON："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        debug.log("BrowserModel", "JSON RECOVERY → success")
        return answer

    @staticmethod
    def _json_stop_response(reason: str) -> str:
        import json

        return json.dumps(
            {
                "action": "STOP",
                "arguments": {},
                "reasoning_summary": str(reason or "结构化输出无效，已安全停止。")[:500],
                "answer": "",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def close(self) -> None:
        """Close the browser context owned by this client, even after failures."""
        context, self._context = self._context, None
        playwright, self._playwright = self._playwright, None
        self._page = None
        self._chat_initialized = False
        self._last_send_monotonic = None
        try:
            if context is not None:
                context.close()
        except Exception as exc:
            debug.log("BrowserModel", f"CONTEXT CLOSE SKIP → {type(exc).__name__}: {exc}")
        finally:
            if playwright is not None:
                try:
                    playwright.stop()
                except Exception as exc:
                    debug.log("BrowserModel", f"PLAYWRIGHT STOP SKIP → {type(exc).__name__}: {exc}")

    @staticmethod
    def _build_prompt(
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
    ) -> str:
        system = str(system_prompt or "").strip()
        user = str(user_prompt or "").strip()

        templates = "web_model.templates"
        if json_mode:
            # The web UI accepts one text message; preserve the caller's role
            # contract as labelled inline instructions rather than fake API roles.
            role = BrowserModel._infer_session_role(system)
            if role in {"reasoning", "general"}:
                parts = [get_prompt("web_model.json_contract")]
                if system:
                    parts.append(f"{get_prompt(templates + '.agent_instructions_label')}\n{system}")
                if user:
                    parts.append(f"{get_prompt(templates + '.current_agent_state_label')}\n{user}")
                parts.append(get_prompt(templates + ".agent_json_reminder"))
            else:
                parts = [get_prompt(templates + ".role_json_contract")]
                if system:
                    parts.append(f"{get_prompt(templates + '.role_system_label')}\n{system}")
                if user:
                    parts.append(f"{get_prompt(templates + '.current_task_data_label')}\n{user}")
                parts.append(get_prompt(templates + ".role_json_reminder"))
            return "\n\n".join(parts).strip()

        # API clients receive system_prompt and user_prompt as separate roles.
        # The browser has only one composer, so render the same distinction in text.
        parts = []
        if system:
            parts.append(f"{get_prompt(templates + '.plain_system_label')}\n{system}")
        if user:
            parts.append(f"{get_prompt(templates + '.plain_user_label')}\n{user}")
        return "\n\n".join(parts).strip()

    def _ensure_page(self):
        if self._page is not None and not self._page.is_closed():
            return self._page

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "WebModel 需要 Playwright。请运行："
                "pip install playwright && playwright install"
            ) from exc

        # Browser profiles contain persistent login cookies; they must not
        # follow a pre-existing symlink or inherit world-readable permissions.
        self.user_data_dir = str(ensure_private_directory(self.user_data_dir))
        edge_windows_before = self._edge_window_handles()
        self._playwright = sync_playwright().start()

        launch_kwargs = self._browser_launch_kwargs()
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                **launch_kwargs
            )
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"无法启动本地浏览器（channel={self.browser_channel!r}）。"
                "可设置 STUDY_AGENT_WEB_BROWSER=chromium 并安装 Playwright Chromium。"
            ) from exc

        try:
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
            self._page.goto(
                self.url,
                wait_until="domcontentloaded",
                timeout=self.timeout * 1000,
            )
            # Edge may create its top-level HWND slightly after Playwright returns.
            # Poll for it so prewarming can reliably move the correct window behind
            # other applications rather than silently skipping the operation.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                self._created_edge_window_handles = self._find_new_edge_window_handles(
                    edge_windows_before
                )
                if self._created_edge_window_handles:
                    break
                self._sleep(0.1)
            if not self._created_edge_window_handles:
                debug.log(
                    "BrowserModel",
                    "WINDOW HANDLE → no new Edge HWND detected after 3s",
                )
            return self._page
        except Exception:
            # Navigation can fail after Playwright and the persistent context
            # have already started. Release both before propagating the error.
            self.close()
            raise

    def _browser_launch_kwargs(self) -> dict[str, Any]:
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": self.user_data_dir,
            "headless": False,
            "args": [],
        }
        if self.browser_channel:
            launch_kwargs["channel"] = self.browser_channel
        # Edge on Windows does not need Chromium's --no-sandbox default arg.
        # Filtering it avoids Edge's unsupported-command warning while keeping
        # the visible browser session fully sandboxed by the OS/browser.
        if os.name == "nt":
            launch_kwargs["ignore_default_args"] = ["--no-sandbox"]
        return launch_kwargs

    @staticmethod
    def _edge_window_handles() -> set[int]:
        """Return visible top-level windows owned by msedge.exe on Windows."""
        if os.name != "nt":
            return set()
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            handles: set[int] = set()
            process_query = 0x1000

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def enum_window(hwnd, _lparam):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                process = kernel32.OpenProcess(process_query, False, pid.value)
                if not process:
                    return True
                try:
                    size = wintypes.DWORD(32768)
                    buffer = ctypes.create_unicode_buffer(size.value)
                    ok = kernel32.QueryFullProcessImageNameW(
                        process, 0, buffer, ctypes.byref(size)
                    )
                    if ok and Path(buffer.value).name.casefold() == "msedge.exe":
                        handles.add(int(hwnd))
                finally:
                    kernel32.CloseHandle(process)
                return True

            user32.EnumWindows(enum_window, 0)
            return handles
        except Exception as exc:
            debug.log("BrowserModel", f"WINDOW ENUM SKIP → {type(exc).__name__}: {exc}")
            return set()

    @classmethod
    def _find_new_edge_window_handles(cls, before: set[int]) -> set[int]:
        return cls._edge_window_handles() - set(before)

    def _send_browser_window_to_back(self, page) -> None:
        """Restore this model's Edge window if minimized and put it behind others.

        Unlike SW_MINIMIZE, placing the window at HWND_BOTTOM keeps it visible
        to Playwright's actionability checks while avoiding a foreground window
        that blocks the user's other work.
        """
        debug.log(
            "BrowserModel",
            f"WINDOW BACK ATTEMPT → platform={os.name}, "
            f"tracked_handles={sorted(self._created_edge_window_handles)}",
        )
        if os.name != "nt":
            debug.log("BrowserModel", "WINDOW BACK SKIP → automatic placement is Windows-only")
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            handles = set(self._created_edge_window_handles)
            if not handles:
                debug.log(
                    "BrowserModel",
                    "WINDOW BACK FAILED → no newly-created Edge window handle was tracked",
                )
                return

            target = None
            try:
                page_title = str(page.title() or "").strip().casefold()
            except Exception as exc:
                page_title = ""
                debug.log(
                    "BrowserModel",
                    f"WINDOW BACK TITLE READ FAILED → {type(exc).__name__}: {exc}",
                )
            debug.log(
                "BrowserModel",
                f"WINDOW BACK MATCH → page_title={page_title!r}, candidates={len(handles)}",
            )
            if page_title:
                for hwnd in handles:
                    buffer = ctypes.create_unicode_buffer(1024)
                    user32.GetWindowTextW(hwnd, buffer, 1024)
                    title = str(buffer.value).strip().casefold()
                    debug.log(
                        "BrowserModel",
                        f"WINDOW BACK CANDIDATE → hwnd={hwnd}, title={title!r}",
                    )
                    if page_title in title or title in page_title:
                        target = hwnd
                        break
            if target is None:
                if len(handles) != 1:
                    debug.log(
                        "BrowserModel",
                        "WINDOW BACK FAILED → multiple new Edge windows and no title match",
                    )
                    return
                target = next(iter(handles))

            if not user32.IsWindow(target):
                debug.log(
                    "BrowserModel",
                    f"WINDOW BACK FAILED → hwnd={target} is not a valid window",
                )
                return

            # Restore a minimized window, then put it at HWND_BOTTOM without
            # activating it. Log each OS operation so a failed attempt is visible.
            restored = user32.ShowWindow(target, 4)
            flags = 0x0001 | 0x0002 | 0x0010 | 0x0040  # NOSIZE|NOMOVE|NOACTIVATE|SHOWWINDOW
            debug.log(
                "BrowserModel",
                f"WINDOW BACK SET POSITION → hwnd={target}, z_order=HWND_BOTTOM, "
                f"flags={flags:#x}, restore_previous_state={restored}",
            )
            # Win32 constants: HWND_BOTTOM = 1, HWND_NOTOPMOST = -2.
            # -2 only removes topmost status; it does NOT place the window at the bottom.
            positioned = user32.SetWindowPos(target, 1, 0, 0, 0, 0, flags)
            if positioned:
                debug.log("BrowserModel", f"WINDOW BACK SUCCESS → hwnd={target}, z_order=HWND_BOTTOM")
            else:
                error_code = getattr(ctypes, "get_last_error", lambda: 0)()
                debug.log(
                    "BrowserModel",
                    f"WINDOW BACK FAILED → SetWindowPos returned false, last_error={error_code}",
                )
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"WINDOW BACK FAILED → {type(exc).__name__}: {exc}",
            )

    def _minimize_browser_window(self, page) -> None:
        """Minimize only Edge windows created by this BrowserModel instance."""
        if self.debug_mode or os.name != "nt":
            return
        try:
            import ctypes
            handles = set(self._created_edge_window_handles)
            if not handles:
                debug.log("BrowserModel", "WINDOW MINIMIZE SKIP → new Edge window not identified")
                return

            target = None
            try:
                page_title = str(page.title() or "").strip().casefold()
            except Exception:
                page_title = ""
            if page_title:
                for hwnd in handles:
                    buffer = ctypes.create_unicode_buffer(1024)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buffer, 1024)
                    title = str(buffer.value).strip().casefold()
                    if page_title in title:
                        target = hwnd
                        break
            if target is None:
                if len(handles) != 1:
                    debug.log(
                        "BrowserModel",
                        "WINDOW MINIMIZE SKIP → multiple new Edge windows and no title match",
                    )
                    return
                target = next(iter(handles))
            if ctypes.windll.user32.IsWindow(target):
                ctypes.windll.user32.ShowWindow(target, 6)
                debug.log("BrowserModel", f"WINDOW → minimized hwnd={target}")
        except Exception as exc:
            debug.log("BrowserModel", f"WINDOW MINIMIZE SKIP → {type(exc).__name__}: {exc}")
    def new_chat(self) -> None:
        """Start a fresh browser conversation for the next generation."""
        self._check_cancelled()
        page = self._ensure_page()
        self._ensure_logged_in(page)
        self._check_cancelled()
        self._json_mode_active = False
        self._send_browser_window_to_back(page)
        self._start_fresh_chat(page)
        self._send_browser_window_to_back(page)
        self._chat_initialized = True
        self._chat_reset_count += 1
        debug.log("BrowserModel", f"NEW CHAT → reset_count={self._chat_reset_count}")

    def _wait_for_chat_input(self, page, *, timeout: float = 8.0) -> bool:
        """Wait for the web app to render a usable chat input during prewarm."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            self._check_cancelled()
            if self._find_visible(
                page, self.INPUT_SELECTORS
            ) is not None:
                debug.log("BrowserModel", "PREWARM LOGIN → chat input detected")
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._sleep(min(0.25, remaining))

    def _dismiss_cookie_banner(self, page) -> bool:
        """Dismiss a visible cookie-consent overlay without bypassing security checks."""
        try:
            banner = page.locator(self.COOKIE_BANNER_SELECTOR)
            if banner.count() == 0:
                return False
            for index in range(banner.count() - 1, -1, -1):
                candidate = banner.nth(index)
                if not candidate.is_visible():
                    continue
                for label in self.DEFAULT_COOKIE_ACCEPT_LABELS:
                    import re as _re
                    try:
                        button = candidate.get_by_role(
                            "button",
                            name=_re.compile(_re.escape(label), _re.IGNORECASE),
                        )
                        if button.count() <= 0:
                            continue
                        for button_index in range(button.count() - 1, -1, -1):
                            target = button.nth(button_index)
                            if target.is_visible():
                                target.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                                debug.log("BrowserModel", "COOKIE BANNER → consent accepted")
                                return True
                    except Exception:
                        continue
        except Exception as exc:
            debug.log("BrowserModel", f"COOKIE BANNER SKIP → {type(exc).__name__}: {exc}")
        return False

    def _ensure_logged_in(self, page) -> None:
        """Give the user time to complete the first manual web login.

        A browser-backed model cannot safely infer or automate authentication.
        If the chat input is not visible, keep the browser alive and wait for
        the user to finish logging in. Pressing Enter in the terminal resumes
        the request; the input is checked again before sending the prompt.
        """
        if self._find_visible(
            page,
            self.INPUT_SELECTORS,
        ) is not None:
            return

        if self.debug_mode:
            print(
                "\n[BrowserModel] 未检测到聊天输入框。"
                "\n[BrowserModel] 请在 Edge 中完成 DeepSeek 登录。"
                "\n[BrowserModel] 完成后回终端按 Enter 继续。"
            )
        else:
            print("\n[BrowserModel] 请在 Edge 中登录 DeepSeek，完成后回终端按 Enter 继续。")
        self._wait_for_terminal_enter()

        if self._find_visible(
            page,
            self.INPUT_SELECTORS,
        ) is None:
            raise RuntimeError(
                "登录后仍未找到 DeepSeek Web 输入框。"
                "请确认已经进入聊天页面，再重试。"
            )

    def _wait_for_terminal_enter(self) -> None:
        """Wait for Enter while still allowing an active run to be cancelled."""
        if os.name == "nt":
            import msvcrt
            while True:
                self._check_cancelled()
                if msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char == chr(3):
                        raise KeyboardInterrupt
                    if char in {"\r", "\n"}:
                        return
                self._sleep(0.1)

        import select
        try:
            sys.stdin.fileno()
        except (AttributeError, OSError, ValueError, UnsupportedOperation):
            input()
            self._check_cancelled()
            return

        while True:
            self._check_cancelled()
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.readline()
                return

    @staticmethod
    def _session_id_from_url(url: str) -> str | None:
        match = re.search(r"/a/chat/s/([^/?#]+)", str(url or ""))
        return match.group(1) if match else None

    @staticmethod
    def _click_first_visible(
        page,
        labels: tuple[str, ...],
        *,
        role: str | None = None,
        button_container_selector: str | None = None,
        click_timeout_ms: int | None = None,
    ) -> bool:
        """Resolve and click a matching visible control, tolerating DOM rerenders.

        DeepSeek can rerender the sidebar after a locator is highlighted. Keep
        each candidate failure local, re-query before retrying, and log the
        actual click exception instead of silently treating it as a missing
        button.
        """
        import re as _re
        patterns = [_re.compile(re.escape(label), _re.IGNORECASE) for label in labels]

        def try_click(locator, description: str) -> bool:
            for attempt in range(2):
                try:
                    count = locator.count()
                    if count <= 0:
                        return False
                    candidate = locator.nth(count - 1)
                    if not candidate.is_visible():
                        return False
                    try:
                        candidate.scroll_into_view_if_needed(timeout=click_timeout_ms or BrowserModel.CLICK_ACTION_TIMEOUT_MS)
                    except (AttributeError, TypeError):
                        # Lightweight test doubles and older locator wrappers
                        # may not expose scroll_into_view_if_needed.
                        pass
                    candidate.click(timeout=click_timeout_ms or BrowserModel.CLICK_ACTION_TIMEOUT_MS)
                    return True
                except Exception as exc:
                    debug.log(
                        "BrowserModel",
                        f"CLICK RETRY → {description}, attempt={attempt + 1}/2, "
                        f"error={type(exc).__name__}: {exc}",
                    )
                    if attempt == 0:
                        continue
            return False

        for pattern in patterns:
            try:
                locator = page.get_by_role(role, name=pattern) if role else page.get_by_text(pattern)
                if try_click(locator, f"label={pattern.pattern!r}"):
                    return True
            except Exception as exc:
                debug.log("BrowserModel", f"CLICK LOCATOR SKIP → {type(exc).__name__}: {exc}")

        try:
            buttons = page.locator(button_container_selector or BrowserModel.BUTTON_CONTAINER_SELECTOR)
            for index in range(buttons.count() - 1, -1, -1):
                button = buttons.nth(index)
                try:
                    if not button.is_visible():
                        continue
                    text = " ".join(str(value or "") for value in (
                        button.inner_text(),
                        button.get_attribute("aria-label"),
                        button.get_attribute("title"),
                    )).strip()
                    if any(pattern.search(text) for pattern in patterns):
                        if try_click(buttons.nth(index), f"button-text={text!r}"):
                            return True
                except Exception as exc:
                    debug.log(
                        "BrowserModel",
                        f"CLICK CANDIDATE SKIP → index={index}, "
                        f"error={type(exc).__name__}: {exc}",
                    )
        except Exception as exc:
            debug.log("BrowserModel", f"CLICK BUTTON SCAN SKIP → {type(exc).__name__}: {exc}")
        return False

    def _click_first_visible_configured(
        self, page, labels: tuple[str, ...], *, role: str | None = None
    ) -> bool:
        """Use the compatibility helper with this browser's configured locators."""
        return BrowserModel._click_first_visible(
            page,
            labels,
            role=role,
            button_container_selector=self.BUTTON_CONTAINER_SELECTOR,
            click_timeout_ms=self.CLICK_ACTION_TIMEOUT_MS,
        )

    def _start_fresh_chat(self, page, *, force: bool = False) -> None:
        if self.session_pause_seconds:
            self._sleep(self.session_pause_seconds)

        old_session = self._session_id_from_url(getattr(page, "url", ""))

        # For ordinary work, an empty root page can be accepted as a new chat.
        # For answer generation/auditing, force a new session whenever the page
        # still identifies the previous conversation, even if its messages were
        # cleared or hidden.
        if self._find_visible(page, self.INPUT_SELECTORS) is not None:
            visible_messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            has_visible_message = False
            for index in range(visible_messages.count() - 1, -1, -1):
                try:
                    if visible_messages.nth(index).is_visible():
                        has_visible_message = True
                        break
                except Exception:
                    continue
            if not has_visible_message and not (force and old_session):
                debug.log("BrowserModel", "FRESH CHAT → current blank chat accepted")
                return

        clicked_new_chat = self._click_first_visible_configured(
            page, self.DEFAULT_NEW_CHAT_LABELS, role="button"
        )
        if not clicked_new_chat:
            clicked_new_chat = self._click_first_visible_configured(
                page, self.DEFAULT_NEW_CHAT_LABELS
            )
        if not clicked_new_chat:
            raise RuntimeError(
                "DeepSeek Web 未找到“New chat/新对话”控件，且当前会话已有消息，"
                "无法保证 Coder 使用独立上下文。"
            )
        self._wait_random("page_transition")

        deadline = time.monotonic() + min(15.0, float(self.timeout))
        while time.monotonic() < deadline:
            current_session = self._session_id_from_url(getattr(page, "url", ""))

            # Prefer an explicit session transition when DeepSeek exposes it.
            if old_session and current_session and current_session != old_session:
                debug.log(
                    "BrowserModel",
                    f"FRESH CHAT → session {old_session} -> {current_session}",
                )
                if self._find_visible(
                    page, self.INPUT_SELECTORS
                ) is not None:
                    return

            # Also accept a cleared message list: some DeepSeek UI states
            # create the new session lazily and update the URL later.
            visible_messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            has_visible_message = False
            for index in range(visible_messages.count() - 1, -1, -1):
                try:
                    message = visible_messages.nth(index)
                    if message.is_visible():
                        has_visible_message = True
                        break
                except Exception:
                    continue

            if not has_visible_message and self._find_visible(
                page, self.INPUT_SELECTORS
            ) is not None:
                if force and old_session and current_session == old_session:
                    # The history pane disappearing is not proof of a new
                    # conversation when the URL still points to the old one.
                    self._sleep(self.poll_interval)
                    continue
                debug.log(
                    "BrowserModel",
                    f"FRESH CHAT CONFIRMED → previous={old_session or 'none'}, "
                    f"current={current_session or 'new/undisclosed'}",
                )
                return

            self._sleep(self.poll_interval)

        raise TimeoutError("创建新的 DeepSeek Web 对话后未确认旧消息已清除。")

    def _cleanup_current_chat(self, page) -> None:
        if self.cleanup_pause_seconds:
            self._sleep(self.cleanup_pause_seconds)
        session_id = self._session_id_from_url(getattr(page, "url", ""))
        if not session_id:
            debug.log("BrowserModel", "CLEANUP SKIP → current page has no session id")
            return

        row_link = None
        try:
            links = page.locator(self.HISTORY_SESSION_LINK_TEMPLATE.format(session_id=session_id))
            for index in range(links.count() - 1, -1, -1):
                candidate = links.nth(index)
                if candidate.is_visible():
                    row_link = candidate
                    break
        except Exception:
            pass
        if row_link is None:
            debug.log("BrowserModel", f"CLEANUP SKIP → session row not found: {session_id}")
            return

        try:
            row_link.hover()
        except Exception:
            pass
        if not self._click_session_more(page, row_link):
            debug.log("BrowserModel", f"CLEANUP SKIP → session menu not found: {session_id}")
            return
        self._wait_random("panel_transition")
        if not self._click_first_visible_configured(page, self.DEFAULT_DELETE_LABELS, role="menuitem"):
            if not self._click_first_visible_configured(page, self.DEFAULT_DELETE_LABELS):
                debug.log("BrowserModel", f"CLEANUP SKIP → delete action not found: {session_id}")
                return
        if not self._click_first_visible_configured(page, self.DEFAULT_DELETE_LABELS, role="button"):
            if not self._click_first_visible_configured(page, self.DEFAULT_DELETE_LABELS):
                debug.log("BrowserModel", f"CLEANUP SKIP → delete confirmation not found: {session_id}")
                return

        deadline = time.monotonic() + min(15.0, float(self.timeout))
        while time.monotonic() < deadline:
            try:
                if page.locator(self.HISTORY_SESSION_LINK_TEMPLATE.format(session_id=session_id)).count() == 0:
                    if self.post_cleanup_pause_seconds:
                        self._sleep(self.post_cleanup_pause_seconds)
                    debug.log("BrowserModel", f"CLEANUP SUCCESS → session={session_id}")
                    return
            except Exception:
                pass
            self._sleep(self.poll_interval)
        debug.log("BrowserModel", f"CLEANUP WARNING → session may still exist: {session_id}")

    def _click_session_more(self, page, row_link) -> bool:
        import re as _re

        # Prefer the observed three-dot SVG path scoped to this history row.
        # The path avoids generated CSS hashes and is less brittle than child indexes.
        icon_selector = self.SVG_PATH_BUTTON_TEMPLATE.format(
            path=self.DEFAULT_MORE_ICON_PATH_PREFIX
        )
        try:
            buttons = row_link.locator(icon_selector)
            for index in range(buttons.count() - 1, -1, -1):
                button = buttons.nth(index)
                if button.is_visible():
                    button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                    self._wait_random("click")
                    debug.log("BrowserModel", "CLEANUP MENU → three-dot SVG selector")
                    return True
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"CLEANUP MENU → SVG selector skipped ({type(exc).__name__})",
            )

        # Compatibility fallback for DOM versions where the menu is not inside
        # the row link or DeepSeek changes the SVG path.
        for xpath in self.DEFAULT_MORE_XPATHS:
            try:
                button = row_link.locator(f"xpath={xpath}")
                if button.count() > 0 and button.is_visible():
                    button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                    self._wait_random("click")
                    debug.log(
                        "BrowserModel",
                        f"CLEANUP MENU → structural selector {xpath}",
                    )
                    return True
            except Exception:
                continue

        # Keep semantic fallbacks for future DeepSeek UI variants.
        patterns = [_re.compile(re.escape(label), _re.IGNORECASE) for label in self.DEFAULT_MORE_LABELS]
        try:
            parent = row_link.locator(f"xpath={self.HISTORY_ROW_PARENT_XPATH}")
            for _ in range(6):
                buttons = parent.locator(self.BUTTON_CONTAINER_SELECTOR)
                for index in range(buttons.count() - 1, -1, -1):
                    button = buttons.nth(index)
                    if not button.is_visible():
                        continue
                    text = " ".join(str(value or "") for value in (
                        button.inner_text(),
                        button.get_attribute("aria-label"),
                        button.get_attribute("title"),
                    )).strip()
                    if any(pattern.search(text) for pattern in patterns):
                        button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                        self._wait_random("click")
                        return True
                parent = parent.locator(f"xpath={self.HISTORY_ROW_PARENT_XPATH}")
        except Exception:
            pass
        return False

    def _find_copy_button(self, page):
        """Find Copy in the item belonging to the current response."""
        user_index = self._current_user_message_index(page)
        message = self._latest_response_message(page)
        if message is None and user_index is not None:
            return None
        if message is None:
            messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            for index in range(messages.count() - 1, -1, -1):
                try:
                    candidate = messages.nth(index)
                    if not candidate.is_visible():
                        continue

                    has_response = False
                    for response_selector in self.response_selectors:
                        try:
                            if candidate.locator(response_selector).count() > 0:
                                has_response = True
                                break
                        except Exception:
                            continue

                    if has_response:
                        message = candidate
                        break
                except Exception:
                    continue

        if message is None:
            debug.log("BrowserModel", "COPY SKIP → latest visible ds-message not found")
            return None

        try:
            item = message.locator(
                f"xpath={self.COPY_ITEM_XPATH}"
            )
            if item.count() == 0:
                debug.log(
                    "BrowserModel",
                    "COPY SKIP → latest ds-message has no virtual-list item",
                )
                return None

            copy_selector = self.SVG_PATH_BUTTON_TEMPLATE.format(
                path=self.DEFAULT_COPY_PATH_PREFIX
            )
            buttons = item.locator(copy_selector)
            for index in range(buttons.count() - 1, -1, -1):
                button = buttons.nth(index)
                if button.is_visible():
                    debug.log(
                        "BrowserModel",
                        f"COPY TARGET → virtual_key={item.get_attribute('data-virtual-list-item-key')}",
                    )
                    return button
        except Exception as exc:
            debug.log("BrowserModel", f"COPY LOOKUP SKIP → {exc}")

        return None

    def _read_browser_clipboard(self, page) -> str:
        """Read the content copied by the web UI through a temporary textarea."""
        probe_id = "__study_agent_clipboard_probe"
        try:
            page.evaluate(
                """id => {
                    const old = document.getElementById(id);
                    if (old) old.remove();

                    const el = document.createElement("textarea");
                    el.id = id;
                    el.setAttribute("aria-hidden", "true");
                    el.style.position = "fixed";
                    el.style.left = "-10000px";
                    el.style.top = "0";
                    el.style.width = "1px";
                    el.style.height = "1px";
                    el.style.opacity = "0";
                    document.body.appendChild(el);
                }""",
                probe_id,
            )
            probe = page.locator(f"#{probe_id}")
            page.evaluate(
                "id => document.getElementById(id)?.focus()",
                probe_id,
            )
            page.keyboard.press("Control+V")
            value = probe.input_value()
            page.evaluate(
                "id => document.getElementById(id)?.remove()",
                probe_id,
            )
            return value if isinstance(value, str) else str(value or "")
        except Exception as exc:
            try:
                page.evaluate(
                    "id => document.getElementById(id)?.remove()",
                    probe_id,
                )
            except Exception:
                pass
            debug.log("BrowserModel", f"CLIPBOARD READ SKIP → {exc}")
            return ""

    def _clear_browser_clipboard(self, page) -> str:
        """Seed the OS/browser clipboard with a sentinel before clicking Copy."""
        probe_id = "__study_agent_clipboard_clear_probe"
        sentinel = "__study_agent_clipboard_pending__"
        try:
            page.evaluate(
                """([id, value]) => {
                    const old = document.getElementById(id);
                    if (old) old.remove();

                    const el = document.createElement("textarea");
                    el.id = id;
                    el.setAttribute("aria-hidden", "true");
                    el.value = value;
                    el.style.position = "fixed";
                    el.style.left = "-10000px";
                    el.style.top = "0";
                    el.style.width = "1px";
                    el.style.height = "1px";
                    el.style.opacity = "0";
                    document.body.appendChild(el);
                    el.focus();
                    el.select();
                }""",
                [probe_id, sentinel],
            )
            page.keyboard.press("Control+C")
        except Exception as exc:
            debug.log("BrowserModel", f"CLIPBOARD CLEAR SKIP → {exc}")
        finally:
            try:
                page.evaluate(
                    "id => document.getElementById(id)?.remove()",
                    probe_id,
                )
            except Exception:
                pass
        return sentinel

    def _copy_latest_response_markdown(self, page) -> str:
        """Click DeepSeek's native Copy button and capture its Markdown payload."""
        deadline = time.monotonic() + min(3.0, float(self.timeout))

        while time.monotonic() < deadline:
            button = self._find_copy_button(page)
            if button is not None:
                stale_marker = self._clear_browser_clipboard(page)
                try:
                    # Keep the pointer click bounded. Playwright's default
                    # actionability timeout can otherwise outlive this method's
                    # own copy deadline and make the Agent appear frozen.
                    button.click(timeout=self.COPY_CLICK_TIMEOUT_MS)
                except Exception as exc:
                    debug.log(
                        "BrowserModel",
                        f"COPY BUTTON CLICK SKIP → {type(exc).__name__}: {exc}",
                    )
                    try:
                        # A DOM click bypasses transient pointer hit-testing
                        # races while still invoking the page's real handler.
                        button.evaluate("(element) => element.click()")
                        debug.log("BrowserModel", "COPY BUTTON → DOM click fallback")
                    except Exception as fallback_exc:
                        debug.log(
                            "BrowserModel",
                            "COPY BUTTON FALLBACK SKIP → "
                            f"{type(fallback_exc).__name__}: {fallback_exc}",
                        )
                        # The message can be re-rendered between locating the
                        # button and invoking its handler. Re-resolve the target
                        # within the existing deadline instead of abandoning
                        # clipboard capture on this transient stale-element race.
                        self._sleep(min(0.1, self.poll_interval))
                        continue

                read_deadline = time.monotonic() + min(2.0, float(self.timeout))
                while time.monotonic() < read_deadline:
                    markdown = self._read_browser_clipboard(page)
                    if markdown.strip() and markdown.strip() != stale_marker:
                        # Never log clipboard content; model answers may quote private attachments.
                        debug.log(
                            "BrowserModel",
                            f"COPY SUCCESS → markdown_chars={len(markdown)}",
                        )
                        # Do not leave potentially sensitive answer text on the OS clipboard.
                        self._clear_browser_clipboard(page)
                        return markdown
                    self._sleep(min(0.1, self.poll_interval))

                debug.log("BrowserModel", "COPY WARNING → clipboard remained empty")
                self._clear_browser_clipboard(page)
                return ""

            self._sleep(self.poll_interval)

        debug.log("BrowserModel", "COPY SKIP → native Copy button not found")
        return ""

    def _wait_for_send_slot(self) -> None:
        """Wait a randomized interval between normal submissions."""
        if self._last_send_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - self._last_send_monotonic)
            target_interval = self.operation_timing.sample("send_interval_seconds")
            remaining = max(0.0, target_interval - elapsed)
            if remaining > 0:
                debug.log(
                    "BrowserModel",
                    f"SEND THROTTLE → sleeping {remaining:.2f}s "
                    f"(sampled_interval={target_interval:.2f}s, elapsed={elapsed:.2f}s)",
                )
                self._sleep(remaining)
        self._last_send_monotonic = time.monotonic()

    def _upload_attachments(self, page, files) -> None:
        """Use DeepSeek's visible file-input control to upload original file bytes."""
        payloads = [
            {
                "name": str(item["name"]),
                "mimeType": str(item.get("mimeType") or "application/octet-stream"),
                "buffer": bytes(item["buffer"]),
            }
            for item in files
        ]
        if not payloads:
            return

        file_inputs = page.locator(self.FILE_INPUT_SELECTOR)
        if file_inputs.count() <= 0:
            clicked = self._click_first_visible_configured(
                page, self.DEFAULT_ATTACHMENT_LABELS, role="button"
            ) or self._click_first_visible_configured(page, self.DEFAULT_ATTACHMENT_LABELS)
            if not clicked:
                raise AttachmentUploadError(
                    "DeepSeek Web 未找到文件上传控件或 input[type=file]；"
                    "无法将原始附件交给模型。"
                )
            deadline = time.monotonic() + min(
                5.0, float(self.ATTACHMENT_UPLOAD_TIMEOUT_SECONDS)
            )
            while time.monotonic() < deadline:
                self._check_cancelled()
                file_inputs = page.locator(self.FILE_INPUT_SELECTOR)
                if file_inputs.count() > 0:
                    break
                self._sleep(min(0.1, self.poll_interval))
        count = file_inputs.count()
        if count <= 0:
            raise AttachmentUploadError(
                "点击 DeepSeek Web 文件上传控件后，仍未找到 input[type=file]。"
            )

        total_bytes = sum(len(item["buffer"]) for item in payloads)
        debug.log(
            "BrowserModel",
            f"ATTACHMENT SELECT BEGIN → count={len(payloads)}, "
            f"total_bytes={total_bytes}, timeout_ms={self.ATTACHMENT_SELECT_TIMEOUT_MS}",
        )
        try:
            file_inputs.nth(count - 1).set_input_files(
                payloads,
                timeout=self.ATTACHMENT_SELECT_TIMEOUT_MS,
            )
        except Exception as exc:
            # Upload exceptions may echo local filenames; log only the exception type.
            debug.log(
                "BrowserModel",
                f"ATTACHMENT SELECT FAILED → {type(exc).__name__}",
            )
            raise AttachmentUploadError(
                "DeepSeek Web 无法在附件选择超时时间内接收这些文件；"
                "文件未发送给模型。"
            ) from exc

        expected_names = [item["name"].casefold() for item in payloads]
        deadline = time.monotonic() + float(self.ATTACHMENT_UPLOAD_TIMEOUT_SECONDS)
        stable_since = None
        while time.monotonic() < deadline:
            self._check_cancelled()
            try:
                body = self._compact_text(page.locator(self.BODY_SELECTOR).inner_text()).casefold()
            except Exception:
                body = ""
            names_visible = all(name in body for name in expected_names)
            upload_busy = False
            for selector in self.UPLOAD_BUSY_SELECTORS:
                try:
                    locator = page.locator(selector)
                    if any(
                        locator.nth(index).is_visible()
                        for index in range(locator.count())
                    ):
                        upload_busy = True
                        break
                except Exception:
                    continue
            if names_visible and not upload_busy:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 0.5:
                    debug.log(
                        "BrowserModel",
                        f"ATTACHMENT UPLOAD READY → count={len(expected_names)}",
                    )
                    return
            else:
                stable_since = None
            remaining = max(0.0, deadline - time.monotonic())
            self._sleep(min(self.operation_timing.sample("upload_poll"), remaining))

        raise AttachmentUploadError(
            "DeepSeek Web 附件上传未能在 120 秒内确认完成；"
            "为避免模型在未收到文件时回答，已中止本次请求。"
        )

    def _find_deep_thinking_toggle(self, page):
        """Find the visible DeepSeek Web DeepThink control using configured locators."""
        label_patterns = [
            re.compile(re.escape(label), re.IGNORECASE)
            for label in self.DEFAULT_DEEP_THINKING_LABELS
        ]
        selectors = self.DEFAULT_DEEP_THINKING_SELECTORS
        seen = set()
        for selector in selectors:
            try:
                locator = page.locator(selector)
                count = locator.count()
            except Exception:
                continue
            for index in range(count):
                try:
                    candidate = locator.nth(index)
                    if not candidate.is_visible():
                        continue
                    text_parts = []
                    try:
                        text_parts.append(str(candidate.inner_text() or ""))
                    except Exception:
                        pass
                    for attr in ("aria-label", "title"):
                        try:
                            text_parts.append(str(candidate.get_attribute(attr) or ""))
                        except Exception:
                            pass
                    text = " ".join(text_parts).strip()
                    if not any(pattern.search(text) for pattern in label_patterns):
                        continue
                    try:
                        node_key = candidate.evaluate("(el) => el")
                    except Exception:
                        node_key = (selector, index, text)
                    marker = repr(node_key)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    return candidate
                except Exception:
                    continue
        return None

    @staticmethod
    def _deep_thinking_enabled_state(control) -> bool | None:
        """Read explicit accessible toggle state; never guess based on button color."""
        true_values = {"true", "1", "yes", "on", "checked", "active", "selected"}
        false_values = {"false", "0", "no", "off", "unchecked", "inactive", "unselected"}
        for attr in ("aria-checked", "aria-pressed", "data-active", "data-state", "aria-selected"):
            try:
                raw = control.get_attribute(attr)
            except Exception:
                raw = None
            if raw is None:
                continue
            value = str(raw).strip().casefold()
            if value in true_values:
                return True
            if value in false_values:
                return False
        try:
            classes = str(control.get_attribute("class") or "").casefold()
        except Exception:
            classes = ""
        tokens = {token for token in re.split(r"[^a-z0-9]+", classes) if token}
        if tokens & {"active", "is-active", "checked", "is-checked", "selected", "is-selected", "on"}:
            return True
        if tokens & {"inactive", "is-inactive", "unchecked", "is-unchecked", "unselected", "is-unselected", "off"}:
            return False
        return None

    def _set_deep_thinking_state(self, page, enabled: bool) -> None:
        """Set DeepThink to the requested state and verify the visible control changed."""
        desired = bool(enabled)
        target = "开启" if desired else "关闭"
        toggle = self._find_deep_thinking_toggle(page)
        if toggle is None:
            raise RuntimeError(
                f"未找到 DeepSeek Web 的「深度思考」开关，无法确认本次请求应{target}该模式；"
                "为避免使用错误的推理模式，本次请求未发送。"
            )
        try:
            is_enabled = getattr(toggle, "is_enabled", None)
            if callable(is_enabled) and not is_enabled():
                raise RuntimeError(
                    "DeepSeek Web 的「深度思考」开关当前不可操作；"
                    "请检查网页所选模型、账号状态或页面提示后重试。"
                )
        except RuntimeError:
            raise
        except Exception:
            pass

        state = self._deep_thinking_enabled_state(toggle)
        if state is desired:
            debug.log("BrowserModel", f"THINKING MODE → DeepThink already {'enabled' if desired else 'disabled'}")
            return
        if state is None:
            raise RuntimeError(
                "找到了 DeepSeek Web 的「深度思考」控制，但无法可靠读取当前状态；"
                "为避免误设模式，本次请求未发送。"
            )

        try:
            toggle.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
            self._wait_random("click")
            debug.log("BrowserModel", f"THINKING MODE → toggled DeepThink {'on' if desired else 'off'}")
        except Exception as exc:
            raise RuntimeError(
                f"无法{target} DeepSeek Web 的「深度思考」模式（{type(exc).__name__}）；"
                "本次请求未发送。"
            ) from exc

        deadline = time.monotonic() + min(3.0, max(0.5, float(self.timeout)))
        while time.monotonic() < deadline:
            self._check_cancelled()
            current = self._find_deep_thinking_toggle(page)
            if current is not None and self._deep_thinking_enabled_state(current) is desired:
                debug.log("BrowserModel", f"THINKING MODE → DeepThink confirmed {'enabled' if desired else 'disabled'}")
                return
            self._sleep(min(0.1, self.poll_interval))
        raise RuntimeError(
            f"已尝试{target} DeepSeek Web 的「深度思考」模式，但无法确认状态已生效；"
            "为避免静默使用错误模式，本次请求未发送。"
        )

    def _ensure_deep_thinking_enabled(self, page) -> None:
        """Backward-compatible helper for callers that explicitly require DeepThink."""
        self._set_deep_thinking_state(page, True)

    def _send_prompt_for_mode(self, page, prompt: str, deep_thinking: bool) -> None:
        """Pass the mode to the sender while keeping older monkeypatched adapters compatible."""
        import inspect

        sender = self._send_prompt
        try:
            parameters = inspect.signature(sender).parameters.values()
            accepts_mode = any(
                parameter.name == "deep_thinking"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_mode = True

        if accepts_mode:
            sender(page, prompt, deep_thinking=bool(deep_thinking))
        else:
            sender(page, prompt)

    def _send_prompt(self, page, prompt: str, *, deep_thinking: bool = True) -> None:
        textbox = self._find_visible(
            page,
            self.INPUT_SELECTORS,
        )
        if textbox is None:
            raise RuntimeError(
                "DeepSeek Web 输入框未找到。页面结构可能已变化，"
                "可检查 BrowserModel 的输入选择器。"
            )

        self._send_button_disabled_logged = False
        if deep_thinking:
            # Keep the established hook for callers/tests that customize this precondition.
            self._ensure_deep_thinking_enabled(page)
        else:
            self._set_deep_thinking_state(page, False)
        self._wait_for_send_slot()
        self._check_cancelled()
        debug.log("BrowserModel", f"SEND FILL → chars={len(prompt)}")
        try:
            textbox.fill(prompt, timeout=self.SEND_ACTION_TIMEOUT_MS)
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"SEND FILL SKIP → {type(exc).__name__}: {exc}",
            )
            raise RuntimeError("DeepSeek Web 输入框填充超时或不可操作。") from exc

        self._wait_random("input_settle")
        self._check_cancelled()
        debug.log("BrowserModel", "SEND ENTER → keyboard submission requested")
        try:
            textbox.press("Enter", timeout=self.SEND_ACTION_TIMEOUT_MS)
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"SEND ENTER SKIP → {type(exc).__name__}: {exc}",
            )
            # Direct private-method callers without an active Agent turn have
            # no safe way to establish whether the draft was submitted.
            if not self._pending_prompt:
                raise RuntimeError("DeepSeek Web 输入框发送超时或不可操作。") from exc
            # During a real Agent turn, verify the page state before deciding
            # whether a pointer fallback is safe.
        self._check_cancelled()

        # In production generate() sets _pending_prompt before reaching here.
        # Keep direct unit/test callers backwards-compatible when no pending
        # turn has been registered, but never report success for an unconfirmed
        # real Agent submission.
        if not self._pending_prompt:
            debug.log("BrowserModel", "SEND ACTION ISSUED → no pending turn to verify")
            return

        if self._wait_for_submission_confirmation(page):
            debug.log("BrowserModel", "SEND CONFIRMED → user message appeared in DeepSeek chat")
            return

        # Only click Send if the full prompt is still in the composer. This is
        # the key duplicate-send guard: if the composer was cleared but the
        # message node is not yet discoverable, don't blindly submit again.
        if not self._composer_contains_prompt(textbox, prompt):
            raise self._submission_failure(
                "DeepSeek Web 的发送状态无法确认：输入框已清空，但当前用户消息尚未出现在页面。"
                "为避免重复发送，本次请求已中止；请检查浏览器页面和终端日志。"
            )

        debug.log(
            "BrowserModel",
            "SEND ENTER UNCONFIRMED → prompt still in composer; waiting for enabled Send button",
        )
        self._wait_for_send_button_and_confirmation(page, textbox, prompt)
        debug.log("BrowserModel", "SEND CONFIRMED → user message appeared after Send became ready")

    def _visible_chat_message_count(self, page) -> int | None:
        try:
            messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            return messages.count()
        except Exception:
            return None

    def _new_user_message_visible(self, page) -> bool:
        """Fallback confirmation using message-list growth after this send attempt."""
        baseline = self._submission_baseline_message_count
        if baseline is None:
            return False
        try:
            messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            count = messages.count()
            if count <= baseline:
                return False
            head = self._compact_text(self._pending_prompt or "")[:100]
            for index in range(count - 1, baseline - 1, -1):
                message = messages.nth(index)
                if not message.is_visible():
                    continue
                body = self._compact_text(message.inner_text())
                has_response = any(
                    message.locator(selector).count() > 0
                    for selector in self.response_selectors
                )
                if has_response:
                    continue
                if body:
                    if head and head not in body:
                        debug.log(
                            "BrowserModel",
                            "SEND CONFIRMATION → new user row appeared after submission; "
                            "rendered text differs from the draft prefix",
                        )
                    # A non-assistant chat row that appeared after the send
                    # baseline is stronger evidence than composer events alone.
                    return True
        except Exception:
            return False
        return False

    def _submission_is_confirmed(self, page) -> bool:
        """Return whether the pending prompt is visibly submitted in the chat."""
        if not self._pending_prompt:
            return False
        return (
            self._current_user_message_index(page) is not None
            or self._new_user_message_visible(page)
        )

    def _wait_for_submission_confirmation(self, page) -> bool:
        """Wait until this exact pending user message appears after the send attempt."""
        if not self._pending_prompt:
            return False
        deadline = time.monotonic() + float(self.SEND_CONFIRMATION_TIMEOUT_SECONDS)
        while True:
            self._check_cancelled()
            if self._submission_is_confirmed(page):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._sleep(min(self.poll_interval, remaining))

    def _composer_contains_prompt(self, textbox, prompt: str) -> bool:
        """Return whether the draft still matches the prompt we attempted to send."""
        actual = ""
        try:
            actual = textbox.input_value(timeout=self.SEND_ACTION_TIMEOUT_MS)
        except Exception:
            try:
                actual = textbox.evaluate(
                    "(element) => ('value' in element ? element.value : "
                    "(element.innerText || element.textContent || ''))"
                )
            except Exception:
                return False
        expected = self._compact_text(prompt)
        current = self._compact_text(actual)
        if not expected or not current:
            return False
        head = expected[:120]
        tail = expected[-120:] if len(expected) > 240 else ""
        return head in current and (not tail or tail in current)

    def _submission_failure(self, message: str) -> RuntimeError:
        """Attachment-backed sends must fail as fatal attachment errors, not retry sans files."""
        if self._attachment_session_active:
            return AttachmentUploadError(message)
        return RuntimeError(message)

    def _wait_for_send_button_and_confirmation(self, page, textbox, prompt: str) -> None:
        """Wait for a formerly-disabled Send control, clicking at most once.

        After a successful click, continue checking the message list without
        clicking again; this avoids duplicate submissions if the UI renders
        the submitted message slowly.
        """
        timeout = max(0.1, float(self.SEND_BUTTON_READY_TIMEOUT_SECONDS))
        deadline = time.monotonic() + timeout
        clicked = False
        wait_logged = False
        while True:
            self._check_cancelled()
            if self._submission_is_confirmed(page):
                return

            if not clicked and self._composer_contains_prompt(textbox, prompt):
                if self._click_send_button(page):
                    clicked = True
                    debug.log(
                        "BrowserModel",
                        "SEND BUTTON CLICKED → waiting for user-message confirmation",
                    )
                elif not wait_logged:
                    wait_logged = True
                    debug.log(
                        "BrowserModel",
                        "SEND BUTTON WAIT → control is disabled/unavailable; "
                        f"retrying for up to {timeout:.0f}s",
                    )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if clicked:
                    message = (
                        "已点击 DeepSeek 发送按钮，但在等待期限内仍未确认用户消息已提交；"
                        "为避免重复发送，本次请求已停止。"
                    )
                else:
                    message = (
                        f"DeepSeek 发送按钮在 {timeout:.0f} 秒内仍不可用；"
                        "附件可能仍在上传或处理中，或页面未解除发送禁用状态；"
                        "本次请求未确认发送。"
                    )
                raise self._submission_failure(message)
            self._sleep(min(self.operation_timing.sample("send_button_poll"), remaining))

    def _click_send_button(self, page) -> bool:
        """Click a visible, enabled Send control as a fallback to Enter."""
        stop_tokens = tuple(token.casefold() for token in self.DEFAULT_SEND_EXCLUDE_LABELS)
        for selector in self.DEFAULT_SEND_BUTTON_SELECTORS:
            try:
                locator = page.locator(selector)
                for index in range(locator.count() - 1, -1, -1):
                    button = locator.nth(index)
                    if not button.is_visible():
                        continue
                    class_name = str(button.get_attribute("class") or "")
                    label = " ".join(str(value or "") for value in (
                        button.inner_text(),
                        button.get_attribute("aria-label"),
                        button.get_attribute("title"),
                        button.get_attribute("data-testid"),
                        class_name,
                    )).strip()
                    if any(token in label.casefold() for token in stop_tokens):
                        continue
                    # DeepSeek's current div[role=button] marks the disabled state
                    # through a CSS class, not necessarily disabled/aria-disabled
                    # attributes. Playwright is_enabled() alone cannot catch that.
                    if self.DISABLED_BUTTON_CLASS in class_name.split():
                        if not getattr(self, "_send_button_disabled_logged", False):
                            debug.log(
                                "BrowserModel",
                                f"SEND BUTTON SKIP → disabled class, selector={selector!r}",
                            )
                            self._send_button_disabled_logged = True
                        continue
                    disabled = button.get_attribute("disabled")
                    aria_disabled = str(button.get_attribute("aria-disabled") or "").casefold()
                    if disabled is not None or aria_disabled == "true":
                        continue
                    is_enabled = getattr(button, "is_enabled", None)
                    if callable(is_enabled):
                        try:
                            if not is_enabled():
                                continue
                        except Exception:
                            pass
                    button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                    self._wait_random("click")
                    debug.log(
                        "BrowserModel",
                        f"SEND BUTTON CLICKED → selector={selector!r}, label={label[:80]!r}",
                    )
                    return True
            except Exception as exc:
                debug.log(
                    "BrowserModel",
                    f"SEND BUTTON CANDIDATE SKIP → selector={selector!r}, "
                    f"error={type(exc).__name__}: {exc}",
                )

        if self._click_button_with_matching_label(page, self.DEFAULT_SEND_LABELS):
            debug.log("BrowserModel", "SEND BUTTON CLICKED → accessible label")
            return True
        return False

    def _rate_limit_visible(self, page) -> bool:
        try:
            body = page.locator(self.BODY_SELECTOR)
            if body.count() <= 0:
                return False
            text = self._compact_text(body.inner_text())
            return any(label.casefold() in text.casefold() for label in self.DEFAULT_RATE_LIMIT_LABELS)
        except Exception:
            return False

    def _retry_rate_limited_prompt(self, page) -> bool:
        if not self._rate_limit_visible(page):
            return False
        # Rate-limit retries have a fixed cooldown; do not add the normal
        # randomized send interval on top of this wait.
        delay = self.operation_timing.wait_before_retry(
            sleep=self._sleep,
            check_cancelled=self._check_cancelled,
        )
        debug.log("BrowserModel", f"RATE LIMIT → waited {delay:.1f}s before retry")
        if not self._rate_limit_visible(page):
            return False
        if not self._click_rate_limit_retry_button(page):
            debug.log(
                "BrowserModel",
                "RATE LIMIT → retry button not found "
                f"(labels + XPath checked for up to "
                f"{self.RATE_LIMIT_RETRY_BUTTON_WAIT_SECONDS:.1f}s)",
            )
            return False
        debug.log("BrowserModel", "RATE LIMIT → resend requested")
        return True

    def _click_rate_limit_retry_button(self, page) -> bool:
        """Find the resend action by accessible label, then the observed DOM XPath.

        DeepSeek sometimes exposes an icon/element without a useful accessible
        name, so a text-only lookup is not sufficient. The short polling window
        also allows the retry control to appear after the rate-limit notice.
        """
        deadline = time.monotonic() + self.RATE_LIMIT_RETRY_BUTTON_WAIT_SECONDS
        while True:
            self._check_cancelled()
            if not self._rate_limit_visible(page):
                return False

            if self._click_button_with_matching_label(page, self.DEFAULT_RESEND_LABELS):
                debug.log("BrowserModel", "RATE LIMIT → resend clicked by accessible label")
                return True

            for xpath in self.DEFAULT_RATE_LIMIT_RETRY_XPATHS:
                try:
                    locator = page.locator(f"xpath={xpath}")
                    for index in range(locator.count() - 1, -1, -1):
                        button = locator.nth(index)
                        if not button.is_visible():
                            continue
                        scroll = getattr(button, "scroll_into_view_if_needed", None)
                        if callable(scroll):
                            try:
                                scroll(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                            except Exception:
                                # A visible target can still be clickable when
                                # the browser refuses to scroll it.
                                pass
                        button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                        self._wait_random("click")
                        debug.log(
                            "BrowserModel",
                            f"RATE LIMIT → resend clicked by XPath {xpath!r}",
                        )
                        return True
                except Exception as exc:
                    debug.log(
                        "BrowserModel",
                        f"RATE LIMIT → XPath retry click skipped "
                        f"({type(exc).__name__}: {exc})",
                    )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._sleep(min(self.operation_timing.sample("retry_poll"), remaining))

    def _click_button_with_matching_label(self, page, labels: tuple[str, ...]) -> bool:
        """Click only a button whose rendered label actually matches the requested action."""
        import re as _re
        patterns = [_re.compile(re.escape(label), _re.IGNORECASE) for label in labels]
        try:
            buttons = page.locator(self.BUTTON_CONTAINER_SELECTOR)
            for index in range(buttons.count() - 1, -1, -1):
                button = buttons.nth(index)
                if not button.is_visible():
                    continue
                label = " ".join(str(value or "") for value in (
                    button.inner_text(),
                    button.get_attribute("aria-label"),
                    button.get_attribute("title"),
                )).strip()
                if any(pattern.search(label) for pattern in patterns):
                    button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                    self._wait_random("click")
                    return True
        except Exception as exc:
            debug.log("BrowserModel", f"BUTTON LABEL SCAN SKIP → {type(exc).__name__}: {exc}")
        for pattern in patterns:
            try:
                locator = page.get_by_role("button", name=pattern)
                for index in range(locator.count() - 1, -1, -1):
                    button = locator.nth(index)
                    if not button.is_visible():
                        continue
                    label = " ".join(str(value or "") for value in (
                        button.inner_text(),
                        button.get_attribute("aria-label"),
                        button.get_attribute("title"),
                    )).strip()
                    if pattern.search(label):
                        button.click(timeout=self.CLICK_ACTION_TIMEOUT_MS)
                        return True
            except Exception:
                continue
        return False

    def _continue_generation_if_available(self, page) -> bool:
        try:
            return self._click_button_with_matching_label(page, self.DEFAULT_CONTINUE_LABELS)
        except Exception as exc:
            debug.log("BrowserModel", f"CONTINUE LOOKUP SKIP → {type(exc).__name__}: {exc}")
            return False

    def _response_counts(self, page) -> list[int]:
        return [
            page.locator(selector).count()
            for selector in self.response_selectors
        ]

    def _response_snapshot(self, page) -> list[tuple[int, str]]:
        snapshot: list[tuple[int, str]] = []
        for selector in self.response_selectors:
            try:
                locator = page.locator(selector)
                count = locator.count()
                latest = ""
                if count > 0:
                    latest = locator.nth(count - 1).inner_text().strip()
                snapshot.append((count, latest))
            except Exception:
                snapshot.append((0, ""))
        return snapshot

    def _loading_visible(self, page) -> bool:
        for selector in self.loading_selectors:
            try:
                locator = page.locator(selector)
                for index in range(locator.count() - 1, -1, -1):
                    if locator.nth(index).is_visible():
                        return True
            except Exception:
                continue
        return False

    @staticmethod
    def _compact_text(value: str) -> str:
        return " ".join(str(value or "").split())

    def _current_user_message_index(self, page) -> int | None:
        prompt = self._compact_text(self._pending_prompt or "")
        if not prompt:
            return None

        head = prompt[:120]
        tail = prompt[-120:] if len(prompt) > 240 else ""
        try:
            messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            for index in range(messages.count() - 1, -1, -1):
                message = messages.nth(index)
                if not message.is_visible():
                    continue
                text = self._compact_text(message.inner_text())
                has_response = any(
                    message.locator(selector).count() > 0
                    for selector in self.response_selectors
                )
                if not has_response and head in text and (not tail or tail in text):
                    return index
        except Exception:
            return None
        return None

    def _latest_response_message(self, page):
        user_index = self._current_user_message_index(page)
        if user_index is None:
            return None

        try:
            messages = page.locator(self.CHAT_MESSAGE_SELECTOR)
            for index in range(messages.count() - 1, user_index, -1):
                message = messages.nth(index)
                if not message.is_visible():
                    continue
                for response_selector in self.response_selectors:
                    try:
                        if message.locator(response_selector).count() > 0:
                            return message
                    except Exception:
                        continue
        except Exception:
            return None
        return None

    def _latest_response(
        self,
        page,
        before_snapshot: list[tuple[int, str]],
    ) -> str:
        user_index = self._current_user_message_index(page)
        unbound_current_turn = False
        if user_index is not None:
            current_message = self._latest_response_message(page)
            if current_message is not None:
                for selector in self.response_selectors:
                    try:
                        locator = current_message.locator(selector)
                        count = locator.count()
                        if count <= 0:
                            continue
                        candidate = locator.nth(count - 1).inner_text().strip()
                        if candidate:
                            return candidate
                    except Exception:
                        continue
            else:
                unbound_current_turn = True

            # Fast web responses can briefly appear before DeepSeek finishes
            # splitting the assistant DOM node away from the current user node.
            # Fall back to the global selector only when its text actually
            # changed; a rerender that merely duplicates an old response must
            # not be mistaken for the answer to this turn.
            if not getattr(self, "_response_binding_fallback_logged", False):
                self._response_binding_fallback_logged = True
                debug.log(
                    "BrowserModel",
                    "RESPONSE BINDING FALLBACK → current user turn has no separate assistant node yet",
                )

        for index, selector in enumerate(self.response_selectors):
            try:
                locator = page.locator(selector)
                count = locator.count()
                if count <= 0:
                    continue
                candidate = locator.nth(count - 1).inner_text().strip()
                if not candidate:
                    continue

                before_count, before_text = (
                    before_snapshot[index]
                    if index < len(before_snapshot)
                    else (0, "")
                )
                if unbound_current_turn:
                    if candidate != before_text:
                        return candidate
                elif count > before_count or candidate != before_text:
                    return candidate
            except Exception:
                continue
        return ""

    def _wait_for_response(
        self,
        page,
        before_snapshot: list[tuple[int, str]],
    ) -> str:
        deadline = time.monotonic() + self.timeout
        saw_new_response = False
        latest = ""
        stable_since: float | None = None
        previous = ""

        debug.log(
            "BrowserModel",
            f"WAIT RESPONSE → timeout={self.timeout}s, json_mode={getattr(self, '_json_mode_active', False)}",
        )
        candidate_seen = False
        loading_block_logged = False
        json_wait_logged = False
        invalid_json_since: float | None = None
        rate_limit_retries = 0

        while time.monotonic() < deadline:
            if (
                rate_limit_retries < self.MAX_RATE_LIMIT_RETRIES
                and self._rate_limit_visible(page)
            ):
                if self._retry_rate_limited_prompt(page):
                    rate_limit_retries += 1
                    deadline = time.monotonic() + self.timeout
                    saw_new_response = False
                    latest = ""
                    stable_since = None
                    previous = ""
                    invalid_json_since = None
                    continue
                rate_limit_retries = self.MAX_RATE_LIMIT_RETRIES
            if self.cancellation_event is not None and self.cancellation_event.is_set():
                self._stop_current_generation(page)
                raise RunCancelled("Coder 任务已被用户中止。")
            self._check_cancelled()
            candidate = self._latest_response(page, before_snapshot)
            if candidate:
                saw_new_response = True
                if not candidate_seen:
                    candidate_seen = True
                    json_mode = bool(getattr(self, "_json_mode_active", False))
                    json_ready = not json_mode or self._is_json_object(candidate)
                    preview = " ".join(candidate[:120].split())
                    debug.log(
                        "BrowserModel",
                        "WAIT RESPONSE → candidate detected "
                        f"chars={len(candidate)}, json_ready={json_ready}, "
                        f"preview={preview!r}",
                    )
                if candidate != previous:
                    previous = candidate
                    latest = candidate
                    stable_since = None
                    loading_block_logged = False
                elif stable_since is None:
                    stable_since = time.monotonic()

                json_mode = bool(getattr(self, "_json_mode_active", False))
                json_candidate = self._extract_json_object(candidate) if json_mode else ""
                json_ready = not json_mode or bool(json_candidate)
                now = time.monotonic()
                stable = (
                    stable_since is not None
                    and now - stable_since >= self.stable_seconds
                )
                loading = self._loading_visible(page)

                if json_mode and not json_ready:
                    if invalid_json_since is None:
                        invalid_json_since = now
                    invalid_json_age = now - invalid_json_since
                    # Do not wait for the full 180s request timeout when the web
                    # model has clearly settled on a non-JSON answer. Return the
                    # stable text so generate() can immediately use the existing
                    # structured-output recovery path.
                    if stable and not loading and invalid_json_age >= self.invalid_json_grace_seconds:
                        debug.log(
                            "BrowserModel",
                            "JSON WAIT → stable non-JSON response; returning for recovery "
                            f"after {invalid_json_age:.1f}s",
                        )
                        return latest
                else:
                    invalid_json_since = None

                # A complete JSON object is not enough to conclude that
                # DeepSeek has finished the current turn: the thinking area can
                # contain a complete Agent-looking object before the final
                # answer is emitted. In JSON mode, prefer the response's Copy
                # control as the browser-side completion signal.
                copy_ready = False
                if json_mode and json_ready:
                    try:
                        copy_ready = self._find_copy_button(page) is not None
                    except Exception:
                        copy_ready = False
                    if copy_ready:
                        # A visible Copy control is an explicit completion signal;
                        # preserve the fast-path and only verify the response snapshot.
                        refreshed = self._latest_response(page, before_snapshot)
                        if (
                            refreshed
                            and refreshed == candidate
                            and self._find_copy_button(page) is not None
                        ):
                            debug.log(
                                "BrowserModel",
                                "WAIT RESPONSE → response complete; Copy target available",
                            )
                            return self._extract_json_object(refreshed) if json_mode else refreshed
                        previous = refreshed or ""
                        if refreshed:
                            latest = refreshed
                        stable_since = None
                        continue
                    if not json_wait_logged:
                        json_wait_logged = True
                        debug.log(
                            "BrowserModel",
                            "WAIT RESPONSE → valid JSON candidate found; waiting for response completion signal",
                        )

                ready = stable and json_ready and not loading
                if ready:
                    self._wait_random("response_stability")
                    refreshed = self._latest_response(page, before_snapshot)
                    if (
                        not refreshed
                        or refreshed != latest
                        or self._loading_visible(page)
                    ):
                        previous = refreshed or ""
                        if refreshed:
                            latest = refreshed
                        stable_since = None
                        continue
                    debug.log(
                        "BrowserModel",
                        f"WAIT RESPONSE → ready chars={len(refreshed)}, loading=False",
                    )
                    return refreshed
                if json_mode and json_ready and stable and loading and not loading_block_logged:
                    loading_block_logged = True
                    debug.log(
                        "BrowserModel",
                        "WAIT RESPONSE → valid JSON ready; ignoring lingering loading indicator",
                    )

            self._sleep(self.poll_interval)

        if saw_new_response and latest:
            if getattr(self, "_json_mode_active", False):
                debug.log(
                    "BrowserModel",
                    "JSON WAIT → response remained incomplete; returning latest for recovery",
                )
                return latest
            raise TimeoutError(
                f"等待 DeepSeek Web 回答完成超时：{self.timeout}s"
            )
        raise TimeoutError(
            f"等待 DeepSeek Web 回答超时：{self.timeout}s"
        )

    def _stop_current_generation(self, page) -> bool:
        """Click the visible DeepSeek Stop control before abandoning the current turn."""
        try:
            stopped = self._click_first_visible_configured(
                page,
                self.DEFAULT_STOP_LABELS,
                role="button",
            )
            if not stopped:
                stopped = self._click_first_visible_configured(
                    page,
                    self.DEFAULT_STOP_LABELS,
                )
            if stopped:
                debug.log("BrowserModel", "CANCEL → DeepSeek generation stop requested")
            else:
                debug.log("BrowserModel", "CANCEL → DeepSeek Stop control not found")
            return stopped
        except Exception as exc:
            debug.log(
                "BrowserModel",
                f"CANCEL STOP SKIP → {type(exc).__name__}: {exc}",
            )
            return False

    @staticmethod
    def _find_visible(page, selectors: tuple[str, ...]):
        for selector in selectors:
            locator = page.locator(selector)
            count = locator.count()
            for index in range(count - 1, -1, -1):
                candidate = locator.nth(index)
                try:
                    if candidate.is_visible():
                        return candidate
                except Exception:
                    continue
        return None

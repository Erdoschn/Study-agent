import os
import re
import time
from pathlib import Path
from typing import Any

from .__debug__ import debug
from .reasoner import ModelClient


class BrowserModel(ModelClient):
    """LLM client backed by a normal local browser session.

    This adapter only drives the visible DeepSeek web UI. It does not call
    DeepSeek's private HTTP endpoints or manufacture browser headers/cookies.
    The first run opens a persistent local profile so the user can log in
    manually; later calls reuse that browser session.
    """

    DEFAULT_URL = "https://chat.deepseek.com/"
    DEFAULT_RESPONSE_SELECTORS = (
        ".ds-assistant-message-main-content",
        ".ds-markdown",
        '[data-message-author-role="assistant"]',
    )
    DEFAULT_LOADING_SELECTORS = (
        ".ds-message-loading",
        '[class*="message-loading"]',
        '[class*="streaming"]',
        '[class*="thinking"]',
    )
    DEFAULT_NEW_CHAT_LABELS = ("New chat", "新对话", "新建对话")
    DEFAULT_MORE_LABELS = ("More", "更多", "⋯", "...")
    DEFAULT_DELETE_LABELS = ("Delete chat", "Delete", "删除聊天", "删除对话", "删除")
    DEFAULT_COPY_PATH_PREFIX = "M6.14929 4.02032"

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
        poll_interval: float = 0.5,
        stable_seconds: float = 1.2,
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
        self.response_selectors = response_selectors or self.DEFAULT_RESPONSE_SELECTORS
        self.loading_selectors = loading_selectors or self.DEFAULT_LOADING_SELECTORS
        self.session_pause_seconds = max(0.0, float(session_pause_seconds))
        self.cleanup_pause_seconds = max(0.0, float(cleanup_pause_seconds))
        self.post_cleanup_pause_seconds = max(0.0, float(post_cleanup_pause_seconds))
        self.cleanup_after_generate = bool(cleanup_after_generate)
        self.poll_interval = max(0.1, float(poll_interval))
        self.stable_seconds = max(0.3, float(stable_seconds))

        self._playwright = None
        self._context = None
        self._page = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        del reasoning_effort
        prompt = self._build_prompt(system_prompt, user_prompt, json_mode=json_mode)
        page = self._ensure_page()

        debug.log(
            "BrowserModel",
            f"REQUEST → model={self.model}, chars={len(prompt)}, url={self.url}",
        )

        self._ensure_logged_in(page)
        self._start_fresh_chat(page)
        completed = False
        try:
            before_snapshot = self._response_snapshot(page)
            self._send_prompt(page, prompt)
            answer = self._wait_for_response(page, before_snapshot)

            markdown = self._copy_latest_response_markdown(page)
            if markdown:
                answer = markdown
            elif not answer:
                raise RuntimeError("DeepSeek Web 返回为空。")

            completed = True
            debug.log(
                "BrowserModel",
                f"SUCCESS → model={self.model}, chars={len(answer)}",
            )
            return answer
        finally:
            if completed and self.cleanup_after_generate:
                self._cleanup_current_chat(page)

    def close(self) -> None:
        """Close the browser context owned by this client."""
        if self._context is not None:
            self._context.close()
        self._context = None
        self._page = None
        if self._playwright is not None:
            self._playwright.stop()
        self._playwright = None

    @staticmethod
    def _build_prompt(
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
    ) -> str:
        system = str(system_prompt or "").strip()
        user = str(user_prompt or "").strip()
        parts = []
        if system:
            parts.append(f"System instructions:\n{system}")
        if user:
            parts.append(f"User request:\n{user}")
        if json_mode:
            parts.append("Output requirement: return only valid JSON.")
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

        Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
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

        pages = self._context.pages
        self._page = pages[0] if pages else self._context.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
        return self._page

    def _browser_launch_kwargs(self) -> dict[str, Any]:
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": self.user_data_dir,
            "headless": False,
        }
        if self.browser_channel:
            launch_kwargs["channel"] = self.browser_channel
        # Edge on Windows does not need Chromium's --no-sandbox default arg.
        # Filtering it avoids Edge's unsupported-command warning while keeping
        # the visible browser session fully sandboxed by the OS/browser.
        if os.name == "nt":
            launch_kwargs["ignore_default_args"] = ["--no-sandbox"]
        return launch_kwargs

    def _ensure_logged_in(self, page) -> None:
        """Give the user time to complete the first manual web login.

        A browser-backed model cannot safely infer or automate authentication.
        If the chat input is not visible, keep the browser alive and wait for
        the user to finish logging in. Pressing Enter in the terminal resumes
        the request; the input is checked again before sending the prompt.
        """
        if self._find_visible(
            page,
            (
                "textarea",
                '[contenteditable="true"]',
            ),
        ) is not None:
            return

        print(
            "\n[BrowserModel] DeepSeek Web 尚未检测到聊天输入框。"
            "\n[BrowserModel] 请在打开的 Edge 中完成登录。"
            "\n[BrowserModel] 登录完成后回到终端按 Enter 继续。"
        )
        input()

        if self._find_visible(
            page,
            (
                "textarea",
                '[contenteditable="true"]',
            ),
        ) is None:
            raise RuntimeError(
                "登录后仍未找到 DeepSeek Web 输入框。"
                "请确认已经进入聊天页面，再重试。"
            )

    @staticmethod
    def _session_id_from_url(url: str) -> str | None:
        match = re.search(r"/a/chat/s/([^/?#]+)", str(url or ""))
        return match.group(1) if match else None

    @staticmethod
    def _click_first_visible(page, labels: tuple[str, ...], *, role: str | None = None) -> bool:
        import re as _re
        patterns = [_re.compile(re.escape(label), _re.IGNORECASE) for label in labels]
        for pattern in patterns:
            try:
                locator = page.get_by_role(role, name=pattern) if role else page.get_by_text(pattern)
                for index in range(locator.count() - 1, -1, -1):
                    candidate = locator.nth(index)
                    if candidate.is_visible():
                        candidate.click()
                        return True
            except Exception:
                continue
        try:
            buttons = page.locator("button, [role='button']")
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
                    button.click()
                    return True
        except Exception:
            pass
        return False

    def _start_fresh_chat(self, page) -> None:
        if self.session_pause_seconds:
            time.sleep(self.session_pause_seconds)

        old_session = self._session_id_from_url(getattr(page, "url", ""))
        if not self._click_first_visible(page, self.DEFAULT_NEW_CHAT_LABELS, role="button"):
            if not self._click_first_visible(page, self.DEFAULT_NEW_CHAT_LABELS):
                raise RuntimeError(
                    "DeepSeek Web 未找到“New chat/新对话”控件，无法保证每个 Agent 角色使用独立上下文。"
                )

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
                    page, ("textarea", '[contenteditable="true"]')
                ) is not None:
                    return

            # Also accept a cleared message list: some DeepSeek UI states
            # create the new session lazily and update the URL later.
            visible_messages = page.locator(".ds-message")
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
                page, ("textarea", '[contenteditable="true"]')
            ) is not None:
                debug.log("BrowserModel", "FRESH CHAT → previous visible messages cleared")
                return

            time.sleep(self.poll_interval)

        raise TimeoutError("创建新的 DeepSeek Web 对话后未确认旧消息已清除。")

    def _cleanup_current_chat(self, page) -> None:
        if self.cleanup_pause_seconds:
            time.sleep(self.cleanup_pause_seconds)
        session_id = self._session_id_from_url(getattr(page, "url", ""))
        if not session_id:
            debug.log("BrowserModel", "CLEANUP SKIP → current page has no session id")
            return

        row_link = None
        try:
            links = page.locator(f'a[href*="/a/chat/s/{session_id}"]')
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
        if not self._click_first_visible(page, self.DEFAULT_DELETE_LABELS, role="menuitem"):
            if not self._click_first_visible(page, self.DEFAULT_DELETE_LABELS):
                debug.log("BrowserModel", f"CLEANUP SKIP → delete action not found: {session_id}")
                return
        if not self._click_first_visible(page, self.DEFAULT_DELETE_LABELS, role="button"):
            if not self._click_first_visible(page, self.DEFAULT_DELETE_LABELS):
                debug.log("BrowserModel", f"CLEANUP SKIP → delete confirmation not found: {session_id}")
                return

        deadline = time.monotonic() + min(15.0, float(self.timeout))
        while time.monotonic() < deadline:
            try:
                if page.locator(f'a[href*="/a/chat/s/{session_id}"]').count() == 0:
                    if self.post_cleanup_pause_seconds:
                        time.sleep(self.post_cleanup_pause_seconds)
                    debug.log("BrowserModel", f"CLEANUP SUCCESS → session={session_id}")
                    return
            except Exception:
                pass
            time.sleep(self.poll_interval)
        debug.log("BrowserModel", f"CLEANUP WARNING → session may still exist: {session_id}")

    def _click_session_more(self, page, row_link) -> bool:
        import re as _re
        patterns = [_re.compile(re.escape(label), _re.IGNORECASE) for label in self.DEFAULT_MORE_LABELS]
        try:
            parent = row_link.locator("xpath=..")
            for _ in range(6):
                buttons = parent.locator("button, [role='button']")
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
                        button.click()
                        return True
                parent = parent.locator("xpath=..")
        except Exception:
            pass
        return False

    def _find_copy_button(self, page):
        """Find Copy in the item belonging to the latest visible assistant message."""
        message = None
        messages = page.locator(".ds-message")

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
                'xpath=ancestor::*[@data-virtual-list-item-key][1]'
            )
            if item.count() == 0:
                debug.log(
                    "BrowserModel",
                    "COPY SKIP → latest ds-message has no virtual-list item",
                )
                return None

            copy_selector = (
                '[role="button"]:has(svg path[d^="'
                + self.DEFAULT_COPY_PATH_PREFIX
                + '"])'
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

    def _copy_latest_response_markdown(self, page) -> str:
        """Click DeepSeek's native Copy button and capture its Markdown payload."""
        deadline = time.monotonic() + min(3.0, float(self.timeout))

        while time.monotonic() < deadline:
            button = self._find_copy_button(page)
            if button is not None:
                try:
                    button.click()
                except Exception as exc:
                    debug.log("BrowserModel", f"COPY BUTTON SKIP → {exc}")
                    return ""

                read_deadline = time.monotonic() + min(2.0, float(self.timeout))
                while time.monotonic() < read_deadline:
                    markdown = self._read_browser_clipboard(page)
                    if markdown.strip():
                        debug.log(
                            "BrowserModel",
                            f"COPY SUCCESS → markdown_chars={len(markdown)}",
                        )
                        return markdown
                    time.sleep(min(0.1, self.poll_interval))

                debug.log("BrowserModel", "COPY WARNING → clipboard remained empty")
                return ""

            time.sleep(self.poll_interval)

        debug.log("BrowserModel", "COPY SKIP → native Copy button not found")
        return ""

    def _send_prompt(self, page, prompt: str) -> None:
        textbox = self._find_visible(
            page,
            (
                "textarea",
                '[contenteditable="true"]',
            ),
        )
        if textbox is None:
            raise RuntimeError(
                "DeepSeek Web 输入框未找到。页面结构可能已变化，"
                "可检查 BrowserModel 的输入选择器。"
            )

        textbox.fill(prompt)
        textbox.press("Enter")

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
            messages = page.locator(".ds-message")
            for index in range(messages.count() - 1, -1, -1):
                message = messages.nth(index)
                if not message.is_visible():
                    continue
                text = self._compact_text(message.inner_text())
                if head in text and (not tail or tail in text):
                    return index
        except Exception:
            return None
        return None

    def _latest_response_message(self, page):
        user_index = self._current_user_message_index(page)
        if user_index is None:
            return None

        try:
            messages = page.locator(".ds-message")
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
            return ""

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
                if count > before_count or candidate != before_text:
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

        while time.monotonic() < deadline:
            candidate = self._latest_response(page, before_snapshot)
            if candidate:
                saw_new_response = True
                if candidate != previous:
                    previous = candidate
                    latest = candidate
                    stable_since = None
                elif stable_since is None:
                    stable_since = time.monotonic()

                if (
                    stable_since is not None
                    and time.monotonic() - stable_since >= self.stable_seconds
                    and not self._loading_visible(page)
                ):
                    return latest

            time.sleep(self.poll_interval)

        if saw_new_response and latest:
            raise TimeoutError(
                f"等待 DeepSeek Web 回答完成超时：{self.timeout}s"
            )
        raise TimeoutError(
            f"等待 DeepSeek Web 回答超时：{self.timeout}s"
        )

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

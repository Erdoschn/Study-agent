import os
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
        '[data-message-author-role="assistant"]',
        ".ds-markdown",
    )

    def __init__(
        self,
        model: str = "deepseek-web",
        *,
        url: str | None = None,
        user_data_dir: str | None = None,
        browser_channel: str | None = None,
        timeout: int = 180,
        response_selectors: tuple[str, ...] | None = None,
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
        before_counts = self._response_counts(page)
        self._send_prompt(page, prompt)
        answer = self._wait_for_response(page, before_counts)

        if not answer:
            raise RuntimeError("DeepSeek Web 返回为空。")

        debug.log(
            "BrowserModel",
            f"SUCCESS → model={self.model}, chars={len(answer)}",
        )
        return answer

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
        counts = []
        for selector in self.response_selectors:
            try:
                counts.append(page.locator(selector).count())
            except Exception:
                counts.append(0)
        return counts

    def _wait_for_response(self, page, before_counts: list[int]) -> str:
        deadline = time.monotonic() + self.timeout
        latest = ""

        while time.monotonic() < deadline:
            for index, selector in enumerate(self.response_selectors):
                try:
                    locator = page.locator(selector)
                    count = locator.count()
                    baseline = before_counts[index] if index < len(before_counts) else 0
                    if count <= baseline:
                        continue

                    candidate = locator.nth(count - 1).inner_text().strip()
                    if not candidate:
                        continue

                    latest = candidate
                    if self._is_stable(page, selector, count - 1, candidate, deadline):
                        return latest
                except Exception:
                    continue

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"等待 DeepSeek Web 回答超时：{self.timeout}s"
        )

    def _is_stable(
        self,
        page,
        selector: str,
        index: int,
        initial: str,
        deadline: float,
    ) -> bool:
        stable_until = time.monotonic() + self.stable_seconds
        previous = initial
        while time.monotonic() < stable_until and time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                current = page.locator(selector).nth(index).inner_text().strip()
            except Exception:
                return False
            if not current:
                return False
            if current != previous:
                previous = current
                stable_until = min(
                    deadline,
                    time.monotonic() + self.stable_seconds,
                )
        return True

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

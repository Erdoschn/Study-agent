from __future__ import annotations

import queue
import threading
from threading import Event
from typing import Any, Callable

from core.web_model import BrowserModel


class CoderBrowserSession:
    """Own one Playwright BrowserModel on a dedicated thread.

    Playwright's synchronous API is greenlet-bound, so callers must not move
    the BrowserModel itself across worker threads. This proxy keeps all browser
    operations on the thread that created the BrowserModel while exposing the
    small model interface CoderReasoner needs.
    """

    def __init__(self, **model_kwargs: Any):
        self._tasks: queue.Queue[tuple[Callable[[], Any] | None, queue.Queue]] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._init_error: BaseException | None = None
        self._cancellation_event = Event()
        self.model = "deepseek-web"
        self.user_data_dir = str(model_kwargs.get("user_data_dir", ".coder-browser"))
        self.reuse_chat = bool(model_kwargs.get("reuse_chat", True))
        self.min_send_interval_seconds = float(
            model_kwargs.get("min_send_interval_seconds", 5.0)
        )
        self._thread = threading.Thread(
            target=self._run_browser_thread,
            name="coder-browser",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            error = self._init_error
            raise RuntimeError(
                f"Coder 浏览器线程初始化失败：{type(error).__name__}: {error}"
            ) from error

    def _run_browser_thread(self) -> None:
        browser: BrowserModel | None = None
        try:
            browser = BrowserModel(
                model="deepseek-web",
                user_data_dir=self.user_data_dir,
                timeout=180,
                cleanup_after_generate=False,
                reuse_chat=self.reuse_chat,
                min_send_interval_seconds=self.min_send_interval_seconds,
                debug_mode=False,
                cancellation_event=self._cancellation_event,
            )
            self._browser = browser
            browser.prepare_browser()
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            func, response_queue = self._tasks.get()
            if func is None:
                try:
                    browser.close()
                except Exception:
                    pass
                response_queue.put((True, None))
                return
            try:
                response_queue.put((True, func()))
            except BaseException as exc:
                response_queue.put((False, exc))

    def _call(self, func: Callable[[BrowserModel], Any]) -> Any:
        if self._closed:
            raise RuntimeError("Coder 浏览器会话已经关闭。")
        response_queue: queue.Queue = queue.Queue(maxsize=1)
        self._tasks.put((lambda: func(self._browser), response_queue))
        ok, value = response_queue.get()
        if ok:
            return value
        raise value

    @property
    def cancellation_event(self) -> Event:
        return self._cancellation_event

    def begin_run(self) -> None:
        if self._closed:
            raise RuntimeError("Coder 浏览器会话已经关闭。")
        self._cancellation_event.clear()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        return self._call(
            lambda browser: browser.generate(
                system_prompt,
                user_prompt,
                json_mode=json_mode,
                reasoning_effort=reasoning_effort,
            )
        )

    def new_chat(self) -> None:
        self._call(lambda browser: browser.new_chat())

    def prepare_browser(self) -> None:
        self._call(lambda browser: browser.prepare_browser())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        response_queue: queue.Queue = queue.Queue(maxsize=1)
        self._tasks.put((None, response_queue))
        try:
            response_queue.get(timeout=10)
        except queue.Empty:
            pass
        self._thread.join(timeout=10)

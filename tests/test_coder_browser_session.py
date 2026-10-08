import threading
import time

import pytest

from coder.browser_session import CoderBrowserSession
from core.cancellation import RunCancelled


def test_coder_browser_session_cancel_interrupts_active_browser_call(monkeypatch):
    class FakeBrowser:
        def __init__(self, **kwargs):
            self.cancellation_event = kwargs["cancellation_event"]

        def prepare_browser(self):
            return None

        def generate(self, *args, **kwargs):
            while not self.cancellation_event.is_set():
                time.sleep(0.01)
            raise RunCancelled("cancelled")

        def close(self):
            return None

    monkeypatch.setattr("coder.browser_session.BrowserModel", FakeBrowser)

    session = CoderBrowserSession(
        user_data_dir=".coder-browser-test",
        reuse_chat=True,
        min_send_interval_seconds=0,
    )
    errors = []

    def invoke():
        try:
            session.generate("system", "user", json_mode=True)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    deadline = time.monotonic() + 1
    while worker.is_alive() and time.monotonic() < deadline:
        session.cancel()
        time.sleep(0.01)
    worker.join(timeout=1)

    try:
        assert not worker.is_alive()
        assert errors
        assert isinstance(errors[0], RunCancelled)
        assert session.cancellation_event.is_set()
    finally:
        session.close()


def test_coder_browser_session_begin_run_clears_previous_cancellation(monkeypatch):
    class FakeBrowser:
        def __init__(self, **kwargs):
            self.cancellation_event = kwargs["cancellation_event"]

        def prepare_browser(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("coder.browser_session.BrowserModel", FakeBrowser)

    session = CoderBrowserSession(user_data_dir=".coder-browser-test")
    try:
        session.cancel()
        assert session.cancellation_event.is_set()
        session.begin_run()
        assert not session.cancellation_event.is_set()
    finally:
        session.close()

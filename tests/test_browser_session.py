from __future__ import annotations

import threading
import time

import pytest

from coder import browser_session as browser_session_module


class FakeBrowser:
    def __init__(self, **kwargs):
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def prepare_browser(self):
        return None

    def generate(self, *args, **kwargs):
        self.started.set()
        self.release.wait(timeout=2)
        return "done"

    def new_chat(self):
        return None

    def close(self):
        self.closed = True


def test_browser_session_rejects_calls_after_close(monkeypatch):
    monkeypatch.setattr(browser_session_module, "BrowserModel", FakeBrowser)
    session = browser_session_module.CoderBrowserSession()
    session.close()

    with pytest.raises(RuntimeError, match="已经关闭"):
        session.new_chat()

    assert not session._thread.is_alive()


def test_browser_session_close_queues_sentinel_after_accepted_call(monkeypatch):
    browser_instances = []

    def make_browser(**kwargs):
        browser = FakeBrowser(**kwargs)
        browser_instances.append(browser)
        return browser

    monkeypatch.setattr(browser_session_module, "BrowserModel", make_browser)
    session = browser_session_module.CoderBrowserSession()
    result = []

    caller = threading.Thread(
        target=lambda: result.append(session.generate("", "test")),
        daemon=True,
    )
    caller.start()
    assert browser_instances[0].started.wait(timeout=1)

    closer = threading.Thread(target=session.close, daemon=True)
    closer.start()
    time.sleep(0.05)
    browser_instances[0].release.set()

    caller.join(timeout=2)
    closer.join(timeout=2)

    assert result == ["done"]
    assert not caller.is_alive()
    assert not closer.is_alive()
    assert browser_instances[0].closed
    assert not session._thread.is_alive()



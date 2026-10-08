import time

import pytest

from core import web_model as web_model_module
from core.model_factory import ModelClientFactory
from core.model_registry import ModelRegistry
from core.web_model import BrowserModel


class FakeLocator:
    def __init__(self, values, *, visible=True, on_press=None, index=None):
        self.values = values
        self.visible = visible
        self.on_press = on_press
        self.index = index

    def count(self):
        return len(self.values)

    def nth(self, index):
        return FakeLocator(
            self.values,
            visible=self.visible,
            on_press=self.on_press,
            index=index,
        )

    def is_visible(self):
        return self.visible

    def fill(self, value):
        if self.index is None:
            self.values[:] = [value]
        else:
            self.values[self.index] = value

    def press(self, key):
        if self.on_press is not None:
            self.on_press(key)

    def click(self):
        if self.on_press is not None:
            self.on_press("click")

    def inner_text(self):
        if not self.values:
            return ""
        index = 0 if self.index is None else self.index
        return str(self.values[index])


class FakePage:
    def __init__(self, *, logged_in=True):
        self.prompt = ""
        self.responses = []
        self.logged_in = logged_in
        self.input = FakeLocator(
            [""],
            on_press=self._press,
        ) if logged_in else FakeLocator([], on_press=self._press)

    def get_by_role(self, role, name=None):
        # Unit-test stand-in for DeepSeek's visible "New chat" button.
        if role == "button" and name is not None:
            return FakeLocator(["New chat"])
        return FakeLocator([])

    def get_by_text(self, pattern):
        return FakeLocator([])

    def locator(self, selector):
        if selector in {"textarea", '[contenteditable="true"]'}:
            if self.logged_in and not self.input.values:
                self.input = FakeLocator([""], on_press=self._press)
            return self.input
        if selector == '[data-message-author-role="assistant"]':
            return FakeLocator(self.responses)
        return FakeLocator([])

    def _press(self, key):
        if key == "Enter":
            self.responses.append(f"answer for: {self.input.values[0]}")


def test_build_prompt_keeps_system_and_user_separate():
    prompt = BrowserModel._build_prompt(
        "You are a reasoner.",
        "What is attention?",
        json_mode=True,
    )

    assert "System instructions:" in prompt
    assert "User request:" in prompt
    assert "You are a reasoner." in prompt
    assert "What is attention?" in prompt
    assert "return only valid JSON" in prompt


def test_browser_model_extracts_session_id_from_deepseek_url():
    assert BrowserModel._session_id_from_url(
        "https://chat.deepseek.com/a/chat/s/123e4567-e89b-12d3-a456-426614174000"
    ) == "123e4567-e89b-12d3-a456-426614174000"
    assert BrowserModel._session_id_from_url("https://chat.deepseek.com/") is None


def test_browser_model_session_cleanup_defaults_are_paced():
    model = BrowserModel()
    assert model.session_pause_seconds == 1.5
    assert model.cleanup_pause_seconds == 3.0
    assert model.post_cleanup_pause_seconds == 1.5
    assert model.cleanup_after_generate is True


def test_browser_model_generate_runs_fresh_chat_lifecycle(monkeypatch):
    model = BrowserModel(
        timeout=1,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.05,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.1,
    )
    page = FakePage()
    events = []

    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: "## 原始 Markdown\n\n$x^2$",
    )
    monkeypatch.setattr(
        model,
        "_start_fresh_chat",
        lambda _page: events.append("start"),
    )
    monkeypatch.setattr(
        model,
        "_send_prompt",
        lambda _page, _prompt: events.append("send"),
    )
    monkeypatch.setattr(
        model,
        "_wait_for_response",
        lambda _page, _snapshot: "answer",
    )
    monkeypatch.setattr(
        model,
        "_cleanup_current_chat",
        lambda _page: events.append("cleanup"),
    )

    assert model.generate("", "hello") == "## 原始 Markdown\n\n$x^2$"
    assert events == ["start", "send", "cleanup"]


def test_browser_model_binds_response_to_current_user_turn():
    class Item:
        def __init__(self, key, copy_button):
            self.key = key
            self.copy_button = copy_button

        def count(self):
            return 1

        def get_attribute(self, name):
            assert name == "data-virtual-list-item-key"
            return self.key

        def locator(self, selector):
            if selector == '[role="button"]:has(svg path[d^="M6.14929 4.02032"])':
                return self.copy_button
            return FakeLocator([])

    class Message:
        def __init__(self, text, item, has_response=False):
            self.text = text
            self.item = item
            self.has_response = has_response

        def is_visible(self):
            return True

        def inner_text(self):
            return self.text

        def locator(self, selector):
            if selector == 'xpath=ancestor::*[@data-virtual-list-item-key][1]':
                return self.item
            if selector == ".ds-markdown":
                return FakeLocator(["answer"] if self.has_response else [])
            return FakeLocator([])

    class Messages:
        def __init__(self, values):
            self.values = values

        def count(self):
            return len(self.values)

        def nth(self, index):
            return self.values[index]

    class Page:
        def __init__(self):
            old_button = FakeLocator(["old"])
            new_button = FakeLocator(["new"])
            self.messages = Messages([
                Message("old user", Item("1", old_button)),
                Message("old assistant answer", Item("2", old_button), has_response=True),
                Message("current request: explain MHA", Item("3", new_button)),
                Message("new assistant answer", Item("4", new_button), has_response=True),
            ])

        def locator(self, selector):
            if selector == ".ds-message":
                return self.messages
            return FakeLocator([])


    model = BrowserModel(
        response_selectors=(".ds-markdown",),
        poll_interval=0.01,
        timeout=1,
    )
    model._pending_prompt = "current request: explain MHA"

    page = Page()

    assert model._latest_response(page, [(0, "")]) == "answer"
    assert model._find_copy_button(page) is page.messages.values[-1].item.copy_button


def test_browser_model_finds_copy_button_inside_latest_message_item():
    class Button:
        def __init__(self, name):
            self.name = name

        def count(self):
            return 1

        def nth(self, index):
            assert index == 0
            return self

        def is_visible(self):
            return True

    class Item:
        def __init__(self, name):
            self.name = name
            self.copy_button = Button(name)

        def count(self):
            return 1

        def get_attribute(self, name):
            assert name == "data-virtual-list-item-key"
            return self.name

        def locator(self, selector):
            assert selector == (
                '[role="button"]:has(svg path[d^="M6.14929 4.02032"])'
            )
            return self.copy_button

    class Message:
        def __init__(self, item, has_response=True):
            self.item = item
            self.has_response = has_response

        def is_visible(self):
            return True

        def locator(self, selector):
            if selector == 'xpath=ancestor::*[@data-virtual-list-item-key][1]':
                return self.item
            if selector == ".ds-markdown":
                return FakeLocator(["answer"]) if self.has_response else FakeLocator([])
            return FakeLocator([])

    class Messages:
        def __init__(self, messages):
            self.messages = messages

        def count(self):
            return len(self.messages)

        def nth(self, index):
            return self.messages[index]

    class Page:
        def __init__(self):
            self.old_item = Item("old")
            self.latest_item = Item("latest")
            self.messages = Messages([
                Message(self.old_item),
                Message(self.latest_item),
            ])

        def locator(self, selector):
            assert selector == ".ds-message"
            return self.messages

    model = BrowserModel(response_selectors=(".ds-markdown",))
    page = Page()

    button = model._find_copy_button(page)

    assert button is page.latest_item.copy_button
    assert button is not page.old_item.copy_button


def test_browser_model_start_fresh_chat_waits_for_previous_messages_to_clear():
    class Page:
        def __init__(self):
            self.messages_present = True
            self.input = FakeLocator([""])

            class NewChatButton:
                def is_visible(inner_self):
                    return True

                def click(inner_self):
                    page.messages_present = False

            page = self
            self.new_chat = NewChatButton()

        def get_by_role(self, role, name=None):
            if role == "button" and name is not None:
                return FakeLocator(
                    ["new chat"],
                    on_press=lambda key: self.new_chat.click() if key == "click" else None,
                )

        def get_by_text(self, pattern):
            return FakeLocator([])

        def locator(self, selector):
            if selector == ".ds-message":
                return FakeLocator(["message"] if self.messages_present else [])
            if selector in {"textarea", '[contenteditable="true"]'}:
                return self.input
            return FakeLocator([])

    model = BrowserModel(
        timeout=1,
        poll_interval=0.05,
        session_pause_seconds=0,
    )
    page = Page()

    model._start_fresh_chat(page)

    assert page.messages_present is False


def test_browser_model_copies_and_returns_raw_markdown(monkeypatch):
    model = BrowserModel(timeout=1)
    page = object()
    events = []
    button = FakeLocator(
        ["copy"],
        on_press=lambda key: events.append(key),
    )

    monkeypatch.setattr(model, "_find_copy_button", lambda _page: button)
    monkeypatch.setattr(
        model,
        "_read_browser_clipboard",
        lambda _page: "## 原始 Markdown\n\n$x^2$",
    )

    assert model._copy_latest_response_markdown(page) == (
        "## 原始 Markdown\n\n$x^2$"
    )
    assert events == ["click"]


def test_browser_model_prefers_markdown_response_selector():
    assert BrowserModel.DEFAULT_RESPONSE_SELECTORS[0] == ".ds-assistant-message-main-content"
    assert ".ds-markdown" in BrowserModel.DEFAULT_RESPONSE_SELECTORS


def test_browser_model_waits_until_loading_indicator_disappears(monkeypatch):
    class Page:
        def __init__(self):
            self.responses = ["partial"]
            self.loading = True
            self.input = FakeLocator([""], on_press=lambda key: None)

        def get_by_role(self, role, name=None):
            if role == "button" and name is not None:
                return FakeLocator(["New chat"])
            return FakeLocator([])

        def get_by_text(self, pattern):
            return FakeLocator([])

        def locator(self, selector):
            if selector in {"textarea", '[contenteditable="true"]'}:
                return self.input
            if selector == ".ds-markdown":
                return FakeLocator(self.responses)
            if selector == ".ds-message-loading":
                return FakeLocator(["loading"] if self.loading else [])
            return FakeLocator([])

    model = BrowserModel(
        timeout=2,
        response_selectors=(".ds-markdown",),
        loading_selectors=(".ds-message-loading",),
        poll_interval=0.05,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.15,
    )
    page = Page()
    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: "",
    )
    monkeypatch.setattr(model, "_send_prompt", lambda _page, _prompt: None)

    def finish_generation():
        time.sleep(0.25)
        page.responses[:] = ["partial complete JSON"]
        page.loading = False

    import threading
    threading.Thread(target=finish_generation, daemon=True).start()

    assert model.generate("", "hello") == "partial complete JSON"


def test_browser_model_waits_for_manual_login(monkeypatch):
    model = BrowserModel(
        timeout=2,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.1,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.3,
    )
    page = FakePage(logged_in=False)
    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: "",
    )

    calls = []

    def fake_input():
        calls.append("enter")
        page.logged_in = True

    monkeypatch.setattr("builtins.input", fake_input)

    answer = model.generate("", "hello")

    assert calls == ["enter"]
    assert answer == "answer for: User request:\nhello"


def test_browser_model_does_not_reuse_previous_response(monkeypatch):
    model = BrowserModel(
        timeout=1,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.05,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.1,
    )
    page = FakePage()
    page.responses.append("previous answer")
    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: "",
    )

    def send(_page, _prompt):
        # Simulate a UI implementation that mutates/replaces the latest
        # assistant element after the new prompt is submitted.
        page.responses[-1] = "new answer"

    monkeypatch.setattr(model, "_send_prompt", send)
    assert model.generate("", "hello") == "new answer"


def test_browser_model_uses_page_ui_without_http(monkeypatch):
    model = BrowserModel(
        timeout=2,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.1,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.3,
    )
    page = FakePage()
    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: "",
    )

    answer = model.generate(
        "You are a teacher.",
        "Explain self-attention.",
    )

    assert answer == (
        "answer for: System instructions:\n"
        "You are a teacher.\n\n"
        "User request:\nExplain self-attention."
    )


def test_browser_model_reuses_the_same_page_for_multiple_turns(monkeypatch):
    model = BrowserModel(
        timeout=2,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.1,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.3,
    )
    page = FakePage()
    calls = []

    def ensure_page():
        calls.append("ensure")
        return page

    monkeypatch.setattr(model, "_ensure_page", ensure_page)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: "",
    )

    first = model.generate("", "first")
    second = model.generate("", "second")

    assert first == "answer for: User request:\nfirst"
    assert second == "answer for: User request:\nsecond"
    assert calls == ["ensure", "ensure"]
    assert page.responses == [
        "answer for: User request:\nfirst",
        "answer for: User request:\nsecond",
    ]


def test_browser_model_filters_playwright_no_sandbox_on_windows(monkeypatch):
    model = BrowserModel()
    monkeypatch.setattr(web_model_module.os, "name", "nt")
    options = model._browser_launch_kwargs()
    assert options["headless"] is False
    assert options["channel"] == model.browser_channel
    assert options["ignore_default_args"] == ["--no-sandbox"]


def test_browser_model_reuses_existing_page_without_relaunch():
    model = BrowserModel(timeout=2)
    page = FakePage()
    page.is_closed = lambda: False
    model._page = page

    assert model._ensure_page() is page


def test_browser_model_close_releases_browser_resources():
    class FakeContext:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakePlaywright:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    model = BrowserModel()
    context = FakeContext()
    playwright = FakePlaywright()
    model._context = context
    model._page = object()
    model._playwright = playwright

    model.close()

    assert context.closed is True
    assert playwright.stopped is True
    assert model._context is None
    assert model._page is None
    assert model._playwright is None


def test_browser_model_raises_if_login_was_not_completed(monkeypatch):
    model = BrowserModel(timeout=1)
    page = FakePage(logged_in=False)
    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr("builtins.input", lambda: None)

    with pytest.raises(RuntimeError, match="登录后仍未找到 DeepSeek Web 输入框"):
        model.generate("", "hello")


def test_browser_model_raises_when_response_does_not_arrive(monkeypatch):
    model = BrowserModel(
        timeout=1,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.1,
    )
    page = FakePage()
    page._press = lambda key: None
    page.input.on_press = page._press
    monkeypatch.setattr(model, "_ensure_page", lambda: page)

    with pytest.raises(TimeoutError, match="等待 DeepSeek Web 回答超时"):
        model.generate("", "hello")


def test_factory_caches_browser_client():
    config = {
        "providers": {
            "deepseek_web": {
                "type": "browser",
                "url": "https://chat.deepseek.com/",
                "browser_channel": "msedge",
                "enabled": True,
            }
        },
        "models": {
            "deepseek-web": {
                "provider": "deepseek_web",
                "model": "deepseek-web",
                "enabled": True,
                "paid": False,
            }
        },
    }
    registry = ModelRegistry(config)
    factory = ModelClientFactory(config)
    info = registry.get("deepseek-web")

    first = factory.create(info)
    second = factory.create(info)

    assert isinstance(first, BrowserModel)
    assert first is second
    assert first.url == "https://chat.deepseek.com/"
    assert first.browser_channel == "msedge"


def test_browser_provider_does_not_require_api_endpoint():
    config = {
        "providers": {
            "web": {
                "type": "browser",
                "enabled": True,
            }
        },
        "models": {
            "web-model": {
                "provider": "web",
                "model": "deepseek-web",
                "enabled": True,
                "paid": False,
            }
        },
    }

    registry = ModelRegistry(config)

    assert registry.available(allow_paid=False)[0].name == "web-model"

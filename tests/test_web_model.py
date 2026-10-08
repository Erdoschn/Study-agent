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

    def _value(self):
        if not self.values:
            return None
        index = 0 if self.index is None else self.index
        return self.values[index]

    def fill(self, value, *, timeout=None):
        target = self._value()
        if hasattr(target, "fill"):
            target.fill(value, timeout=timeout)
            return
        if self.index is None:
            self.values[:] = [value]
        else:
            self.values[self.index] = value

    def press(self, key, *, timeout=None):
        target = self._value()
        if hasattr(target, "press"):
            target.press(key, timeout=timeout)
            return
        if self.on_press is not None:
            self.on_press(key)

    def click(self, timeout=None):
        target = self._value()
        if hasattr(target, "click"):
            target.click(timeout=timeout)
            return
        if self.on_press is not None:
            self.on_press("click")

    def is_visible(self):
        target = self._value()
        if hasattr(target, "is_visible"):
            return target.is_visible()
        return self.visible

    def inner_text(self):
        target = self._value()
        if hasattr(target, "inner_text"):
            return target.inner_text()
        return "" if target is None else str(target)

    def get_attribute(self, name):
        target = self._value()
        if hasattr(target, "get_attribute"):
            return target.get_attribute(name)
        return ""


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




def test_browser_model_default_chat_policy_is_fresh():
    model = BrowserModel()
    assert model.reuse_chat is False


def test_browser_model_send_actions_use_bounded_timeouts(monkeypatch):
    model = BrowserModel(timeout=1)
    calls = []

    class Textbox:
        def fill(self, value, *, timeout):
            calls.append(("fill", value, timeout))

        def press(self, key, *, timeout):
            calls.append(("press", key, timeout))

    monkeypatch.setattr(model, "_find_visible", lambda _page, _selectors: Textbox())
    monkeypatch.setattr(model, "_wait_for_send_slot", lambda: None)

    model._send_prompt(object(), "hello")

    assert calls == [
        ("fill", "hello", BrowserModel.SEND_ACTION_TIMEOUT_MS),
        ("press", "Enter", BrowserModel.SEND_ACTION_TIMEOUT_MS),
    ]


def test_browser_model_wraps_send_fill_timeout_as_runtime_error(monkeypatch):
    model = BrowserModel(timeout=1)

    class Textbox:
        def fill(self, value, *, timeout):
            raise TimeoutError("fill blocked")

        def press(self, key, *, timeout):
            raise AssertionError("press should not run")

    monkeypatch.setattr(model, "_find_visible", lambda _page, _selectors: Textbox())
    monkeypatch.setattr(model, "_wait_for_send_slot", lambda: None)

    with pytest.raises(RuntimeError, match="输入框填充超时或不可操作"):
        model._send_prompt(object(), "hello")


def test_browser_model_wraps_send_enter_timeout_as_runtime_error(monkeypatch):
    model = BrowserModel(timeout=1)

    class Textbox:
        def fill(self, value, *, timeout):
            return None

        def press(self, key, *, timeout):
            raise TimeoutError("enter blocked")

    monkeypatch.setattr(model, "_find_visible", lambda _page, _selectors: Textbox())
    monkeypatch.setattr(model, "_wait_for_send_slot", lambda: None)

    with pytest.raises(RuntimeError, match="输入框发送超时或不可操作"):
        model._send_prompt(object(), "hello")


def test_browser_model_send_interval_is_configurable():
    model = BrowserModel(min_send_interval_seconds=7.5)
    assert model.min_send_interval_seconds == 7.5


def test_browser_model_throttles_consecutive_sends(monkeypatch):
    model = BrowserModel(min_send_interval_seconds=5.0)
    model._last_send_monotonic = 100.0
    sleeps = []
    clock = iter([102.0, 107.0])
    monkeypatch.setattr(web_model_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(web_model_module.time, "sleep", lambda value: sleeps.append(value))

    model._wait_for_send_slot()

    assert sleeps == [3.0]
    assert model._last_send_monotonic == 107.0


def test_browser_model_json_mode_prioritizes_structured_output_contract():
    prompt = BrowserModel._build_prompt(
        "Choose SEARCH or ANSWER.",
        '{"question":"kernel k-means是什么？"}',
        json_mode=True,
    )

    assert prompt.startswith("CRITICAL RESPONSE CONTRACT:")
    assert "You are a browser-backed agent runtime." in prompt
    assert "Do not directly answer or teach the user's question." in prompt
    assert "You are controlling StudyAgent through a browser UI." not in prompt
    assert "exactly one valid JSON object" in prompt
    assert prompt.endswith(
        "Return ONLY the JSON object requested by the Agent decision "
        "instructions. Do not answer the user's question directly."
    )

def test_build_prompt_keeps_system_and_user_separate():
    prompt = BrowserModel._build_prompt(
        "You are a reasoner.",
        "What is attention?",
        json_mode=True,
    )

    assert "Agent decision instructions:" in prompt
    assert "Current Agent state:" in prompt
    assert "You are a reasoner." in prompt
    assert "What is attention?" in prompt
    assert (
        "Return ONLY the JSON object requested by the Agent decision instructions."
        in prompt
    )


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
    button = model._find_copy_button(page)
    assert button.values == page.messages.values[-1].item.copy_button.values


def test_browser_model_click_helpers_use_bounded_timeout():
    model = BrowserModel()

    class ButtonLocator:
        def __init__(self):
            self.timeout = None
            self.clicked = False

        def count(self):
            return 1

        def nth(self, index):
            assert index == 0
            return self

        def is_visible(self):
            return True

        def click(self, *, timeout):
            self.timeout = timeout
            self.clicked = True

        def inner_text(self):
            return "New chat"

        def get_attribute(self, name):
            return ""

    button = ButtonLocator()

    class Page:
        def get_by_role(self, role, name=None):
            assert role == "button"
            return FakeLocator([], visible=False)

        def get_by_text(self, pattern):
            return FakeLocator([], visible=False)

        def locator(self, selector):
            assert selector == "button, [role='button']"
            return button

    assert model._click_first_visible(Page(), ("New chat",), role=None) is True
    assert button.clicked is True
    assert button.timeout == BrowserModel.CLICK_ACTION_TIMEOUT_MS

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


def test_browser_model_accepts_existing_blank_chat_as_fresh():
    class Page:
        def __init__(self):
            self.input = FakeLocator([""])

        def get_by_role(self, role, name=None):
            return FakeLocator([], visible=False)

        def get_by_text(self, pattern):
            return FakeLocator([], visible=False)

        def locator(self, selector):
            if selector == ".ds-message":
                return FakeLocator([])
            if selector in {"textarea", '[contenteditable="true"]'}:
                return self.input
            return FakeLocator([])

    model = BrowserModel(timeout=1, session_pause_seconds=0)
    page = Page()

    model._start_fresh_chat(page)

    visible_input = model._find_visible(page, ("textarea", '[contenteditable="true"]'))
    assert visible_input is not None
    assert visible_input.values == page.input.values


def test_browser_model_start_fresh_chat_waits_for_previous_messages_to_clear():
    class NewChatLocator:
        def __init__(self, page):
            self.page = page

        def count(self):
            return 1

        def nth(self, index):
            assert index == 0
            return self

        def is_visible(self):
            return True

        def click(self, *, timeout):
            assert timeout == BrowserModel.CLICK_ACTION_TIMEOUT_MS
            self.page.messages_present = False

        def inner_text(self):
            return "new chat"

        def get_attribute(self, name):
            return ""

    class Page:
        def __init__(self):
            self.messages_present = True
            self.input = FakeLocator([""])
            self.new_chat = NewChatLocator(self)

        def get_by_role(self, role, name=None):
            if role == "button" and name is not None:
                return self.new_chat
            return FakeLocator([])

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

def test_browser_model_does_not_accept_stale_clipboard(monkeypatch):
    model = BrowserModel(timeout=1, poll_interval=0.01)
    button = FakeLocator(["copy"], on_press=lambda key: None)
    monkeypatch.setattr(model, "_find_copy_button", lambda _page: button)
    cleared = []

    def clear(_page):
        cleared.append(True)
        return "__study_agent_clipboard_pending__"

    monkeypatch.setattr(model, "_clear_browser_clipboard", clear)
    monkeypatch.setattr(
        model,
        "_read_browser_clipboard",
        lambda _page: "__study_agent_clipboard_pending__",
    )

    assert model._copy_latest_response_markdown(object()) == ""
    assert cleared == [True]


def test_browser_model_uses_bounded_copy_click_timeout(monkeypatch):
    model = BrowserModel(timeout=1)
    page = object()
    calls = []

    class Button:
        def click(self, *, timeout):
            calls.append(timeout)

    button = Button()
    monkeypatch.setattr(model, "_find_copy_button", lambda _page: button)
    monkeypatch.setattr(
        model,
        "_clear_browser_clipboard",
        lambda _page: "__study_agent_clipboard_pending__",
    )
    monkeypatch.setattr(
        model,
        "_read_browser_clipboard",
        lambda _page: '{"action":"LIST_FILES"}',
    )

    assert model._copy_latest_response_markdown(page) == '{"action":"LIST_FILES"}'
    assert calls == [BrowserModel.COPY_CLICK_TIMEOUT_MS]


def test_browser_model_falls_back_to_dom_click_when_pointer_click_is_blocked(monkeypatch):
    model = BrowserModel(timeout=1)
    page = object()
    events = []

    class Button:
        def click(self, *, timeout):
            events.append(("pointer", timeout))
            raise RuntimeError("intercepted by overlay")

        def evaluate(self, script):
            events.append(("dom", script))

    button = Button()
    monkeypatch.setattr(model, "_find_copy_button", lambda _page: button)
    monkeypatch.setattr(
        model,
        "_clear_browser_clipboard",
        lambda _page: "__study_agent_clipboard_pending__",
    )
    monkeypatch.setattr(
        model,
        "_read_browser_clipboard",
        lambda _page: '{"action":"LIST_FILES"}',
    )

    assert model._copy_latest_response_markdown(page) == '{"action":"LIST_FILES"}'
    assert events == [
        ("pointer", BrowserModel.COPY_CLICK_TIMEOUT_MS),
        ("dom", "(element) => element.click()"),
    ]


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


def test_browser_model_falls_back_to_response_snapshot_when_fast_reply_has_no_separate_assistant_node():
    class Message:
        def __init__(self, text):
            self.text = text

        def is_visible(self):
            return True

        def inner_text(self):
            return self.text

        def locator(self, selector):
            return FakeLocator([])

    class Messages:
        def __init__(self):
            self.values = [Message("current prompt")]

        def count(self):
            return len(self.values)

        def nth(self, index):
            return self.values[index]

    class Page:
        def __init__(self):
            self.messages = Messages()
            self.responses = ["new answer"]

        def locator(self, selector):
            if selector == ".ds-message":
                return self.messages
            if selector == ".ds-markdown":
                return FakeLocator(self.responses)
            return FakeLocator([])

    model = BrowserModel(response_selectors=(".ds-markdown",))
    model._pending_prompt = "current prompt"

    page = Page()
    assert model._latest_response(page, [(1, "old answer")]) == "new answer"

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


def test_browser_model_leaves_browser_visible_until_after_login():
    model = BrowserModel()
    options = model._browser_launch_kwargs()
    assert options["args"] == []


def test_browser_model_debug_mode_keeps_browser_visible():
    model = BrowserModel(debug_mode=True)
    assert model.debug_mode is True


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
    model._chat_initialized = True
    model._last_send_monotonic = 123.0

    model.close()

    assert context.closed is True
    assert playwright.stopped is True
    assert model._context is None
    assert model._page is None
    assert model._playwright is None
    assert model._chat_initialized is False
    assert model._last_send_monotonic is None


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


def test_factory_closes_cached_browser_clients():
    config = {
        "providers": {"web": {"type": "browser", "enabled": True}},
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
    factory = ModelClientFactory(config)
    client = factory.create(registry.get("web-model"))

    closed = []
    client.close = lambda: closed.append(True)

    factory.close()

    assert closed == [True]
    assert factory._browser_clients == {}


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


def test_browser_model_uses_current_deepseek_history_action_xpath():
    class Button:
        def __init__(self):
            self.clicked = False

        def count(self):
            return 1

        def is_visible(self):
            return True

        def click(self, timeout=None):
            self.clicked = True

    class Row:
        def __init__(self):
            self.button = Button()
            self.requested = []

        def locator(self, selector):
            self.requested.append(selector)
            if selector == "xpath=./div[3]/div":
                return self.button
            return FakeLocator([])

    model = BrowserModel()
    row = Row()

    assert model._click_session_more(object(), row) is True
    assert row.requested == ["xpath=./div[3]/div"]
    assert row.button.clicked is True


def test_browser_model_minimizes_only_new_edge_window_on_windows(monkeypatch):
    class User32:
        def __init__(self):
            self.calls = []

        def IsWindow(self, hwnd):
            return hwnd == 5678

        def GetWindowTextW(self, hwnd, buffer, size):
            buffer.value = "DeepSeek"

        def ShowWindow(self, hwnd, command):
            self.calls.append((hwnd, command))

    class FakeCtypes:
        def __init__(self):
            self.windll = type("Windll", (), {"user32": User32()})()

    fake_ctypes = FakeCtypes()
    monkeypatch.setattr(web_model_module.os, "name", "nt")
    monkeypatch.setitem(__import__("sys").modules, "ctypes", fake_ctypes)

    model = BrowserModel()
    model._created_edge_window_handles = {5678}
    model._minimize_browser_window(object())

    assert fake_ctypes.windll.user32.calls == [(5678, 6)]


def test_browser_model_does_not_guess_between_multiple_new_edge_windows(monkeypatch):
    class User32:
        def __init__(self):
            self.calls = []

        def IsWindow(self, hwnd):
            return True

        def GetWindowTextW(self, hwnd, buffer, size):
            buffer.value = "Unrelated"

        def ShowWindow(self, hwnd, command):
            self.calls.append((hwnd, command))

    class FakeCtypes:
        def __init__(self):
            self.windll = type("Windll", (), {"user32": User32()})()

    fake_ctypes = FakeCtypes()
    monkeypatch.setattr(web_model_module.os, "name", "nt")
    monkeypatch.setitem(__import__("sys").modules, "ctypes", fake_ctypes)

    model = BrowserModel()
    model._created_edge_window_handles = {5678, 6789}
    model._minimize_browser_window(object())

    assert fake_ctypes.windll.user32.calls == []


def test_browser_model_debug_mode_does_not_minimize_edge_window(monkeypatch):
    class User32:
        def __init__(self):
            self.calls = []

        def ShowWindow(self, hwnd, command):
            self.calls.append((hwnd, command))

    class FakeCtypes:
        def __init__(self):
            self.windll = type("Windll", (), {"user32": User32()})()

    fake_ctypes = FakeCtypes()
    monkeypatch.setattr(web_model_module.os, "name", "nt")
    monkeypatch.setitem(__import__("sys").modules, "ctypes", fake_ctypes)

    model = BrowserModel(debug_mode=True)
    model._edge_window_handles = {5678}
    model._minimize_browser_window(object())

    assert fake_ctypes.windll.user32.calls == []


def test_browser_model_tracks_only_new_edge_windows(monkeypatch):
    monkeypatch.setattr(
        BrowserModel,
        "_edge_window_handles",
        staticmethod(lambda: {2, 3, 4}),
    )

    assert BrowserModel._find_new_edge_window_handles({1, 2}) == {3, 4}


def test_browser_model_detects_invalid_json_for_recovery():
    assert BrowserModel._needs_json_recovery(
        "Kernel k-means 是标准 k-means 的非线性扩展。"
    ) is True
    assert BrowserModel._needs_json_recovery(
        '{"action":"ANSWER","answer":"ok"}'
    ) is False
    assert BrowserModel._needs_json_recovery(
        '{"answer":"ok"}'
    ) is True


def test_browser_model_recovers_invalid_json_in_same_chat(monkeypatch):
    model = BrowserModel(
        timeout=1,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
    )
    events = []
    responses = [
        "Kernel k-means 是标准 k-means 的非线性扩展。",
        '{"action":"ANSWER","reasoning_summary":"direct answer","answer":"ok"}',
    ]
    monkeypatch.setattr(model, "_ensure_page", lambda: object())
    monkeypatch.setattr(model, "_ensure_logged_in", lambda _page: None)
    monkeypatch.setattr(model, "_minimize_browser_window", lambda _page: None)
    monkeypatch.setattr(model, "_start_fresh_chat", lambda _page: events.append("fresh"))
    monkeypatch.setattr(model, "_response_snapshot", lambda _page: [])
    monkeypatch.setattr(
        model,
        "_send_prompt",
        lambda _page, prompt: events.append(("send", prompt)),
    )
    monkeypatch.setattr(
        model,
        "_wait_for_response",
        lambda _page, _snapshot: responses.pop(0),
    )
    monkeypatch.setattr(model, "_copy_latest_response_markdown", lambda _page: "")
    monkeypatch.setattr(model, "_cleanup_current_chat", lambda _page: events.append("cleanup"))

    answer = model.generate("You are a reasoner.", "choose an action", json_mode=True)

    assert '"action":"ANSWER"' in answer
    assert events[0] == "fresh"
    assert events[-1] == "cleanup"
    sent_prompts = [event[1] for event in events if isinstance(event, tuple)]
    assert len(sent_prompts) == 2
    assert sent_prompts[1].startswith("STRUCTURED OUTPUT RECOVERY.")


def test_browser_model_marks_recovery_failure_as_request_failure(monkeypatch):
    model = BrowserModel(
        timeout=1,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
    )
    events = []
    responses = ["plain answer", "still plain answer"]
    monkeypatch.setattr(model, "_ensure_page", lambda: object())
    monkeypatch.setattr(model, "_ensure_logged_in", lambda _page: None)
    monkeypatch.setattr(model, "_minimize_browser_window", lambda _page: None)
    monkeypatch.setattr(model, "_start_fresh_chat", lambda _page: None)
    monkeypatch.setattr(model, "_response_snapshot", lambda _page: [])
    monkeypatch.setattr(model, "_send_prompt", lambda _page, prompt: None)
    monkeypatch.setattr(model, "_wait_for_response", lambda _page, _snapshot: responses.pop(0))
    monkeypatch.setattr(model, "_copy_latest_response_markdown", lambda _page: "")
    monkeypatch.setattr(model, "_cleanup_current_chat", lambda _page: events.append("cleanup"))

    with pytest.raises(RuntimeError, match="JSON 恢复失败"):
        model.generate("", "choose an action", json_mode=True)

    assert events == ["cleanup"]


def test_browser_model_accepts_coder_agent_actions_in_json_mode():
    response = '{"action":"LIST_FILES","arguments":{},"reasoning_summary":"inspect workspace"}'

    assert BrowserModel._is_json_object(response)
    assert BrowserModel._needs_json_recovery(response) is False


def test_browser_model_json_wait_accepts_stable_json_despite_lingering_loading(monkeypatch):
    model = BrowserModel(
        timeout=1,
        poll_interval=0.05,
        stable_seconds=0.1,
    )
    model._json_mode_active = True
    answer = '{"action":"LIST_FILES","arguments":{},"reasoning_summary":"inspect workspace"}'

    monkeypatch.setattr(model, "_latest_response", lambda _page, _snapshot: answer)
    monkeypatch.setattr(model, "_loading_visible", lambda _page: True)

    started = time.monotonic()
    assert model._wait_for_response(object(), [(0, "")]) == answer
    assert time.monotonic() - started < 0.05 + 0.1

def test_browser_model_json_wait_does_not_accept_incomplete_stream():
    class Page:
        def __init__(self):
            self.response = (
                '{"action":"LIST_FILES","arguments":{},'
                '"reasoning_summary":"inspect'
            )

        def locator(self, selector):
            if selector == ".ds-markdown":
                return FakeLocator([self.response])
            return FakeLocator([])

    model = BrowserModel(
        response_selectors=(".ds-markdown",),
        loading_selectors=(),
        timeout=2,
        poll_interval=0.05,
        stable_seconds=0.1,
    )
    page = Page()

    def finish():
        time.sleep(0.25)
        page.response = (
            '{"action":"LIST_FILES","arguments":{},'
            '"reasoning_summary":"inspect workspace"}'
        )

    import threading
    threading.Thread(target=finish, daemon=True).start()

    model._json_mode_active = True
    answer = model._wait_for_response(page, [(0, "")])

    assert answer.endswith('"reasoning_summary":"inspect workspace"}')


def test_browser_model_does_not_replace_complete_json_with_partial_clipboard(monkeypatch):
    model = BrowserModel(
        response_selectors=('[data-message-author-role="assistant"]',),
        timeout=1,
        poll_interval=0.05,
        session_pause_seconds=0,
        cleanup_pause_seconds=0,
        post_cleanup_pause_seconds=0,
        stable_seconds=0.1,
    )
    page = FakePage()
    complete = '{"action":"LIST_FILES","arguments":{},"reasoning_summary":"inspect workspace"}'

    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(model, "_start_fresh_chat", lambda _page: None)
    monkeypatch.setattr(model, "_send_prompt", lambda _page, _prompt: None)
    monkeypatch.setattr(model, "_wait_for_response", lambda _page, _snapshot: complete)
    monkeypatch.setattr(
        model,
        "_copy_latest_response_markdown",
        lambda _page: '{"action":"LIST_FILES","arguments":{},"reasoning_summary":"inspect',
    )
    monkeypatch.setattr(model, "_cleanup_current_chat", lambda _page: None)

    answer = model.generate(
        '{"action":"PLAN|LIST_FILES|FINISH","arguments":{}}',
        "inspect workspace",
        json_mode=True,
    )

    assert answer == complete


def test_browser_model_reuses_chat_when_enabled(monkeypatch):
    model = BrowserModel(
        timeout=1,
        reuse_chat=True,
        cleanup_after_generate=False,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.01,
        session_pause_seconds=0,
        stable_seconds=0.1,
    )
    page = FakePage()
    events = []

    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_start_fresh_chat",
        lambda _page: events.append("new_chat"),
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
        "_copy_latest_response_markdown",
        lambda _page: "",
    )
    monkeypatch.setattr(model, "_minimize_browser_window", lambda _page: None)

    assert model.generate("", "first") == "answer"
    assert model.generate("", "second") == "answer"
    assert events == ["new_chat", "send", "send"]
    assert model._chat_initialized is True


def test_browser_model_new_chat_explicitly_resets_reused_conversation(monkeypatch):
    model = BrowserModel(
        timeout=1,
        reuse_chat=True,
        cleanup_after_generate=False,
        session_pause_seconds=0,
    )
    page = FakePage()
    starts = []

    monkeypatch.setattr(model, "_ensure_page", lambda: page)
    monkeypatch.setattr(
        model,
        "_start_fresh_chat",
        lambda _page: starts.append("new_chat"),
    )
    monkeypatch.setattr(model, "_minimize_browser_window", lambda _page: None)

    model._chat_initialized = True
    model.new_chat()

    assert starts == ["new_chat"]
    assert model._chat_initialized is True
    assert model._chat_reset_count == 1

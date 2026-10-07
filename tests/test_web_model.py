import pytest

from core.model_factory import ModelClientFactory
from core.model_registry import ModelInfo, ModelRegistry
from core.web_model import BrowserModel


class FakeLocator:
    def __init__(self, values, *, visible=True, on_press=None):
        self.values = values
        self.visible = visible
        self.on_press = on_press

    def count(self):
        return len(self.values)

    def nth(self, index):
        return FakeLocator(
            [self.values[index]],
            visible=self.visible,
            on_press=self.on_press,
        )

    def is_visible(self):
        return self.visible

    def fill(self, value):
        self.values[:] = [value]

    def press(self, key):
        if self.on_press is not None:
            self.on_press(key)

    def inner_text(self):
        return str(self.values[0]) if self.values else ""


class FakePage:
    def __init__(self):
        self.prompt = ""
        self.responses = []
        self.input = FakeLocator(
            [""],
            on_press=self._press,
        )

    def locator(self, selector):
        if selector in {"textarea", '[contenteditable="true"]'}:
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


def test_browser_model_uses_page_ui_without_http(monkeypatch):
    model = BrowserModel(
        timeout=2,
        response_selectors=('[data-message-author-role="assistant"]',),
        poll_interval=0.1,
        stable_seconds=0.3,
    )
    page = FakePage()
    monkeypatch.setattr(model, "_ensure_page", lambda: page)

    answer = model.generate(
        "You are a teacher.",
        "Explain self-attention.",
    )

    assert answer == (
        "answer for: System instructions:\n"
        "You are a teacher.\n\n"
        "User request:\nExplain self-attention."
    )


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

import pytest

from coder import agent as agent_module


def test_coder_agent_initializes_debug_mode_before_reasoner(tmp_path, monkeypatch):
    seen = []

    class FakeReasoner:
        def __init__(
            self,
            *,
            debug_mode=False,
            reuse_chat=True,
            min_send_interval_seconds=5.0,
        ):
            seen.append((debug_mode, reuse_chat, min_send_interval_seconds))

    monkeypatch.setattr(agent_module, "CoderReasoner", FakeReasoner)

    agent_module.CoderAgent(
        workspace=tmp_path,
        search_router=object(),
        harness=object(),
        debug_mode=True,
    )

    assert seen == [(True, True, 5.0)]


def test_coder_agent_defaults_debug_mode_to_false(tmp_path, monkeypatch):
    seen = []

    class FakeReasoner:
        def __init__(
            self,
            *,
            debug_mode=False,
            reuse_chat=True,
            min_send_interval_seconds=5.0,
        ):
            seen.append((debug_mode, reuse_chat, min_send_interval_seconds))

    monkeypatch.setattr(agent_module, "CoderReasoner", FakeReasoner)

    agent_module.CoderAgent(
        workspace=tmp_path,
        search_router=object(),
        harness=object(),
    )

    assert seen == [(False, True, 5.0)]


def test_coder_agent_handles_model_selected_new_chat(tmp_path):
    actions = iter([
        {"action": "NEW_CHAT", "arguments": {}},
        {"action": "FINISH", "arguments": {}},
    ])
    new_chat_calls = []

    class FakeReasoner:
        def decide(self, state, tool_specs):
            return next(actions)

        def new_chat(self):
            new_chat_calls.append(True)

    class FakeHarness:
        sandbox = object()
        backup = object()

        def tool_specs(self):
            return []

        def execute(self, action, arguments, state):
            assert action == "VERIFY_GOAL"
            return {"verified": True}

    agent = agent_module.CoderAgent(
        workspace=tmp_path,
        reasoner=FakeReasoner(),
        harness=FakeHarness(),
        max_runtime_seconds=30,
    )
    state = agent.run("修复一个小 bug")

    assert state.finished is True
    assert state.goal_verified is True
    assert state.chat_resets == 1
    assert state.metrics["chat_resets"] == 1
    assert new_chat_calls == [True]
    assert [step.action for step in state.steps] == ["NEW_CHAT", "FINISH"]


def test_coder_reasoner_accepts_new_chat_action():
    from coder.reasoner import CoderReasoner

    decision = CoderReasoner._parse(
        '{"action":"NEW_CHAT","arguments":{},"reasoning_summary":"上下文需要重置"}'
    )

    assert decision["action"] == "NEW_CHAT"
    assert decision["arguments"] == {}


def test_coder_reasoner_new_chat_delegates_to_browser_model():
    from coder.reasoner import CoderReasoner

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def new_chat(self):
            self.calls += 1

    model = FakeModel()
    reasoner = CoderReasoner(model=model)

    reasoner.new_chat()

    assert model.calls == 1


def test_coder_reasoner_defaults_to_reused_chat():
    from coder.reasoner import CoderReasoner

    class FakeModel:
        reuse_chat = True

    reasoner = CoderReasoner(model=FakeModel())

    assert reasoner.model.reuse_chat is True


def test_coder_reasoner_accepts_configurable_chat_policy(monkeypatch):
    from coder.reasoner import CoderReasoner

    seen = {}

    class FakeBrowserModel:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("coder.reasoner.BrowserModel", FakeBrowserModel)
    reasoner = CoderReasoner(
        reuse_chat=False,
        min_send_interval_seconds=9.0,
    )

    assert reasoner.model is not None
    assert seen["reuse_chat"] is False
    assert seen["min_send_interval_seconds"] == 9.0

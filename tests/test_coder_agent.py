from coder import agent as agent_module


def test_coder_agent_initializes_debug_mode_before_reasoner(monkeypatch):
    seen = []

    class FakeReasoner:
        def __init__(self, *, debug_mode=False):
            seen.append(debug_mode)

    monkeypatch.setattr(agent_module, "CoderReasoner", FakeReasoner)

    agent_module.CoderAgent(
        workspace=".",
        search_router=object(),
        harness=object(),
        debug_mode=True,
    )

    assert seen == [True]


def test_coder_agent_defaults_debug_mode_to_false(monkeypatch):
    seen = []

    class FakeReasoner:
        def __init__(self, *, debug_mode=False):
            seen.append(debug_mode)

    monkeypatch.setattr(agent_module, "CoderReasoner", FakeReasoner)

    agent_module.CoderAgent(
        workspace=".",
        search_router=object(),
        harness=object(),
    )

    assert seen == [False]


def test_coder_agent_handles_model_selected_new_chat():
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
        workspace=".",
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

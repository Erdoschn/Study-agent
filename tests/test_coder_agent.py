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

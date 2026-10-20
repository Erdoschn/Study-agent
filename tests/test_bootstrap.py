from __future__ import annotations

from pathlib import Path

import study_agent.bootstrap as bootstrap


def test_build_agent_wires_one_shared_model_stack_and_optional_storage(monkeypatch, tmp_path):
    events = []

    class Registry:
        def __init__(self, config):
            events.append(("registry", config))

    class Router:
        def __init__(self, registry):
            events.append(("router", registry))

    class Factory:
        def __init__(self, config):
            events.append(("factory", config))

    class Reasoner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("reasoner", kwargs))

    class Teacher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("teacher", kwargs))

    class SearchRouter:
        def __init__(self):
            self.providers = []

        def register(self, provider):
            self.providers.append(provider)

    class ArxivProvider:
        pass

    class WikipediaProvider:
        pass

    class Executor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("executor", kwargs))

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("agent", kwargs))

    monkeypatch.setattr(bootstrap, "ModelRegistry", Registry)
    monkeypatch.setattr(bootstrap, "ModelRouter", Router)
    monkeypatch.setattr(bootstrap, "ModelClientFactory", Factory)
    monkeypatch.setattr(bootstrap, "AgentReasoner", Reasoner)
    monkeypatch.setattr(bootstrap, "Teacher", Teacher)
    monkeypatch.setattr(bootstrap, "SearchRouter", SearchRouter)
    monkeypatch.setattr(bootstrap, "ArxivSearchProvider", ArxivProvider)
    monkeypatch.setattr(bootstrap, "WikipediaSearchProvider", WikipediaProvider)
    monkeypatch.setattr(bootstrap, "ToolExecutor", Executor)
    monkeypatch.setattr(bootstrap, "StudyAgent", Agent)

    config = {"providers": ["fake"]}
    db_path = tmp_path / "learner.sqlite3"
    agent = bootstrap.build_agent(config, allow_paid=True, knowledge_graph_path=db_path)

    reasoner = next(value for name, value in events if name == "reasoner")
    teacher = next(value for name, value in events if name == "teacher")
    assert reasoner.kwargs["model_router"] is teacher.kwargs["model_router"]
    assert reasoner.kwargs["model_factory"] is teacher.kwargs["model_factory"]
    assert reasoner.kwargs["allow_paid"] is True
    assert teacher.kwargs["allow_paid"] is True
    assert agent.kwargs["knowledge_graph_path"] == db_path
    assert agent.kwargs["max_steps"] is None
    assert len(agent.kwargs["tool_executor"].kwargs["search_router"].providers) == 2


def test_main_keeps_legacy_builder_exports():
    # Existing scripts may import these functions from main.py while entry
    # points migrate to the shared composition root.
    import main

    assert main.build_agent is bootstrap.build_agent
    assert main.build_search_router is bootstrap.build_search_router

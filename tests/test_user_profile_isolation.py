from __future__ import annotations

import threading
from types import SimpleNamespace

from core.agent import StudyAgent
from examples.study_agent_unified_api import (
    build_profile_agent,
    get_or_create_profile_agent,
    resolve_request_username,
)


def _template():
    agent = StudyAgent(reasoner=object(), teacher=object(), tool_executor=object(),
                       max_steps=9, knowledge_graph_path=None)
    agent._web_model_factory = object()
    agent._web_model_registry = object()
    return agent


def test_profile_agents_have_separate_mutable_state_and_graph_files(tmp_path):
    template = _template()
    alice = build_profile_agent(template, "Alice", data_root=tmp_path, max_profiles=8)
    bob = build_profile_agent(template, "Bob", data_root=tmp_path, max_profiles=8)
    assert alice is not bob
    assert alice.knowledge_graph is not bob.knowledge_graph
    assert alice.knowledge_graph.storage_path != bob.knowledge_graph.storage_path
    assert alice._web_model_factory is template._web_model_factory
    assert alice.student_state is None and bob.student_state is None


def test_same_profile_and_model_reuses_agent_but_other_profiles_do_not(tmp_path):
    template = _template()
    server = SimpleNamespace(user_agents={}, user_agents_lock=threading.RLock(),
                             data_root=tmp_path, max_user_profiles=8)
    alice = get_or_create_profile_agent(server, template, "Alice", "deepseek-web")
    assert alice is get_or_create_profile_agent(server, template, "ALICE", "deepseek-web")
    bob = get_or_create_profile_agent(server, template, "Bob", "deepseek-web")
    assert alice is not bob
    assert alice.knowledge_graph.storage_path != bob.knowledge_graph.storage_path
    assert len(server.user_agents) == 2


def test_api_requires_username_unless_single_user_default_is_configured(monkeypatch):
    monkeypatch.delenv("STUDY_AGENT_DEFAULT_USERNAME", raising=False)
    import pytest
    with pytest.raises(ValueError, match="缺少 username"):
        resolve_request_username({"model": "deepseek-web"})
    assert resolve_request_username({"username": "Alice"}) == "alice"
    monkeypatch.setenv("STUDY_AGENT_DEFAULT_USERNAME", "LegacyUser")
    assert resolve_request_username({"model": "deepseek-web"}) == "legacyuser"

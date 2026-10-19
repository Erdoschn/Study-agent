from __future__ import annotations

import threading
from types import SimpleNamespace

from core.agent import StudyAgent
from core.knowledge_graph import KnowledgeNode
from examples.deepseek_web_study_agent_api import (
    build_profile_agent,
    get_or_create_profile_agent,
)
from web_test_helpers import read_frontend_source


def _template_agent():
    template = StudyAgent(
        reasoner=object(),
        teacher=object(),
        tool_executor=object(),
        max_steps=9,
        knowledge_graph_path=None,
    )
    template._web_model_factory = object()
    template._web_model_registry = object()
    return template


def test_profile_agents_isolate_graph_state_and_preserve_shared_model_factory(tmp_path):
    template = _template_agent()
    alice = build_profile_agent(template, "Alice", data_root=tmp_path, max_profiles=8)
    bob = build_profile_agent(template, "Bob", data_root=tmp_path, max_profiles=8)

    assert alice is not bob
    assert alice.knowledge_graph is not bob.knowledge_graph
    assert alice.knowledge_graph.storage_path != bob.knowledge_graph.storage_path
    assert alice._web_model_factory is template._web_model_factory
    assert bob._web_model_factory is template._web_model_factory
    assert alice.student_state is None
    assert bob.student_state is None

    alice.knowledge_graph.nodes["alice-concept"] = KnowledgeNode(
        id="alice-concept", name="Alice-only concept"
    )
    alice.knowledge_graph.save()
    bob.knowledge_graph.load()

    assert "alice-concept" in alice.knowledge_graph.nodes
    assert "alice-concept" not in bob.knowledge_graph.nodes


def test_same_username_shares_graph_between_model_modes_but_not_runtime_state(tmp_path):
    template = _template_agent()
    api_template = _template_agent()
    api_template._api_model_factory = object()
    server = SimpleNamespace(
        user_agents={},
        user_agents_lock=threading.RLock(),
        data_root=tmp_path,
        max_user_profiles=8,
    )

    alice_web = get_or_create_profile_agent(server, template, "Alice", "deepseek-web")
    alice_web_duplicate = get_or_create_profile_agent(server, template, "alice", "deepseek-web")
    alice_api = get_or_create_profile_agent(server, api_template, "ALICE", "study-agent-api")
    bob_web = get_or_create_profile_agent(server, template, "Bob", "deepseek-web")

    assert alice_web is alice_web_duplicate
    assert alice_web is not alice_api
    assert alice_web.knowledge_graph is not alice_api.knowledge_graph
    assert alice_web.knowledge_graph.storage_path == alice_api.knowledge_graph.storage_path
    assert alice_web.knowledge_graph.storage_path != bob_web.knowledge_graph.storage_path
    assert alice_api._api_model_factory is api_template._api_model_factory
    assert alice_web.student_state is not bob_web.student_state or (
        alice_web.student_state is None and bob_web.student_state is None
    )
    assert len(server.user_agents) == 3


def test_frontend_requires_profile_and_partitions_chat_history():
    source = read_frontend_source()
    assert 'id="profilePicker"' in source
    assert 'id="usernameInput"' in source
    assert 'id="changeUser"' in source
    assert 'KEY_PREFIX="studyagent.ui.v3.user."' in source
    assert "function profileStorageKey()" in source
    assert "payload={username:state.username,model:state.mode" in source
    assert "if(state.username){$(\"#userLabel\")" in source


def test_api_defaults_legacy_clients_to_one_explicit_local_profile():
    from core.user_identity import DEFAULT_USERNAME, normalize_username

    assert normalize_username(DEFAULT_USERNAME) == "local"
    assert "STUDY_AGENT_DEFAULT_USERNAME" in read_api_source()


def read_frontend_source():
    from examples.deepseek_web_study_agent_api import WEB_ROOT
    return (WEB_ROOT / "index.html").read_text(encoding="utf-8")


def read_api_source():
    import examples.deepseek_web_study_agent_api as api
    from pathlib import Path
    return Path(api.__file__).read_text(encoding="utf-8")

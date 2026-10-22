from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.assessment_generator import AssessmentGenerator
from core.question_files import (
    load_question_generation_state,
    profile_token,
    read_assessment_file,
    save_assessment_files,
    save_question_generation_state,
)
from core.prompt_config import get_prompt
from examples.study_agent_unified_api import _completed_conversation_turn_count
from study_agent.application.runtime_epistemic import persist_runtime_epistemic_artifacts
from study_agent.domain.answer_revision import CoverageOutcome
from study_agent.domain.prerequisite_graph import (
    KnowledgePoint,
    PrerequisiteEdge,
    PrerequisiteGraph,
)
from study_agent.infrastructure.answer_revision_sqlite import (
    SQLiteAnswerRevisionRepository,
    SQLiteCoverageLedgerRepository,
)
from study_agent.infrastructure.dossier_sqlite import SQLiteTaskDossierRepository
from study_agent.infrastructure.prerequisite_graph_sqlite import SQLitePrerequisiteGraphRepository


def _assessment() -> dict:
    return {
        "primary_concept": "test topic",
        "difficulty_level": "graduate",
        "question_type": "open_ended",
        "question": "Explain why a conductor has zero electric field in electrostatic equilibrium.",
        "expected_answer": "ANSWER_ONLY_MARKER_4f8e: mobile charges rearrange until the internal field vanishes.",
        "rubric": ["State the electrostatic-equilibrium condition.", "Explain charge redistribution."],
    }


def test_question_and_answer_markdown_are_separate_and_clickable(tmp_path):
    output = tmp_path / "出题"
    files = save_assessment_files(_assessment(), "Alice", root=output)

    question_path = read_assessment_file(files["user_token"], files["question_file"], root=output)
    answer_path = read_assessment_file(files["user_token"], files["answer_file"], root=output)
    question = question_path.read_text(encoding="utf-8")
    answer = answer_path.read_text(encoding="utf-8")

    assert question_path.suffix == answer_path.suffix == ".md"
    assert "Explain why a conductor" in question
    assert "ANSWER_ONLY_MARKER_4f8e" not in question
    assert "ANSWER_ONLY_MARKER_4f8e" in answer
    assert files["question_path"].startswith("出题/")
    assert files["answer_path"].startswith("出题/")
    assert files["question_url_path"].endswith(files["question_file"])
    assert profile_token("ALICE") == profile_token("alice")
    with pytest.raises(ValueError):
        read_assessment_file(files["user_token"], "../outside.md", root=output)


def test_question_generation_state_is_persisted_per_user_and_conversation(tmp_path):
    output = tmp_path / "出题"
    state = {
        "completed_turn_count": 2,
        "question_file": "question-20261022_213000-012345abcdef.md",
        "answer_file": "answer-20261022_213000-012345abcdef.md",
        "question_text": "Question text",
    }
    save_question_generation_state("alice", "conversation-a", state, root=output)

    assert load_question_generation_state("ALICE", "conversation-a", root=output) == {
        **state,
        "schema_version": 1,
    }
    assert load_question_generation_state("alice", "conversation-b", root=output) is None
    assert load_question_generation_state("bob", "conversation-a", root=output) is None


def test_turn_counter_does_not_count_question_cards_as_new_user_turns():
    messages = [
        {"role": "user", "content": "Explain Gauss's law."},
        {"role": "assistant", "content": "Gauss's law relates flux to enclosed charge."},
        {"role": "assistant", "content": "Generated quiz card"},
    ]
    assert _completed_conversation_turn_count(messages) == 1
    messages.extend([
        {"role": "user", "content": "My answer is ..."},
        {"role": "assistant", "content": "Feedback on the answer."},
    ])
    assert _completed_conversation_turn_count(messages) == 2
    assert _completed_conversation_turn_count(messages + [
        {"role": "user", "content": "not answered yet"},
    ]) == 2


class _Registry:
    def __init__(self):
        self.successes = []
        self.failures = []

    def record_success(self, model_name, capability):
        self.successes.append((model_name, capability))

    def record_failure(self, model_name, capability, provider_level=False):
        self.failures.append((model_name, capability, provider_level))

    def is_provider_level_failure(self, exc):
        return False


class _QuestionClient:
    def __init__(self, result):
        self.result = result
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt, user_prompt, **kwargs):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return json.dumps(self.result, ensure_ascii=False)


class _QuestionRouter:
    def __init__(self, client):
        self.registry = _Registry()
        self.client = client
        self.model = SimpleNamespace(name="test-model", reasoning_effort_param=None)

    def select_choice_candidates(self, capability, **kwargs):
        assert capability == "assessment"
        return [SimpleNamespace(model=self.model, effort="medium")]


class _QuestionFactory:
    def __init__(self, client):
        self.client = client

    def create(self, model):
        return self.client


def test_question_generator_uses_distinct_first_and_regeneration_prompts():
    data = _assessment()
    data["difficulty"] = data["difficulty_level"]
    client = _QuestionClient(data)
    router = _QuestionRouter(client)
    generator = AssessmentGenerator(router, _QuestionFactory(client))

    first = generator.generate_from_conversation([
        {"role": "user", "content": "Teach me electrostatic equilibrium."},
        {"role": "assistant", "content": "A conductor redistributes charge until the internal field vanishes."},
    ])
    first_prompt = client.system_prompt
    assert first["question"] == data["question"]
    assert first_prompt == get_prompt("study_agent.question_generation")
    assert router.registry.successes[-1] == ("test-model", "assessment")

    replacement_data = {
        **data,
        "question": "A different, applied conductor scenario asks for the physical condition and its consequences.",
    }
    client.result = replacement_data
    replacement = generator.generate_from_conversation([
        {"role": "user", "content": "Teach me electrostatic equilibrium."},
        {"role": "assistant", "content": "A conductor redistributes charge until the internal field vanishes."},
    ], regenerate=True, previous_question=first["question"])
    assert replacement["question"] != first["question"]
    assert client.system_prompt == get_prompt("study_agent.question_regeneration")
    payload = json.loads(client.user_prompt)
    assert payload["previous_question"] == first["question"]
    assert payload["regeneration_requested"] is True


def test_runtime_persists_dossier_revision_and_unresolved_coverage(tmp_path):
    result = SimpleNamespace(
        final_answer="A careful answer with assumptions.",
        claims=[{"claim": "A verifiable statement", "type": "factual"}],
        domain="physics",
        student=SimpleNamespace(known_topics=set(), weak_topics=set()),
    )
    artifacts = persist_runtime_epistemic_artifacts(
        username="alice",
        task_id="task-123",
        question="Explain the field and include assumptions.",
        result=result,
        data_root=tmp_path / "data",
        max_profiles=10,
        curriculum_db_path=tmp_path / "missing-curriculum.sqlite3",
    )
    graph_path = (tmp_path / "data" / "users" / profile_token("alice") / "knowledge_graph.sqlite3")
    database = graph_path.with_name("epistemic.sqlite3")
    dossier = SQLiteTaskDossierRepository(database).load_latest(user_id="alice", task_id="task-123")
    revision = SQLiteAnswerRevisionRepository(database).load_latest(user_id="alice", task_id="task-123")
    assert dossier is not None and dossier.version == 1
    assert dossier.sources[0].kind == "user_request"
    assert revision is not None
    assert revision.answer_text == result.final_answer
    assert revision.revision_number == 1
    assert artifacts.answer_revision_id == revision.revision_id
    ledger = SQLiteCoverageLedgerRepository(database).load(
        user_id="alice", task_id="task-123", answer_revision_id=revision.revision_id
    )
    assert ledger is not None
    assert all(entry.outcome is CoverageOutcome.UNRESOLVED for entry in ledger.entries)
    assert artifacts.coverage_unresolved_count == len(ledger.entries)


def test_runtime_prerequisite_advice_uses_configured_dag_and_says_unknown_is_unassessed(tmp_path):
    curriculum_path = tmp_path / "curriculum.sqlite3"
    graph = PrerequisiteGraph(
        graph_id="curriculum",
        version=1,
        points=(
            KnowledgePoint(id="electric-field", name="Electric field"),
            KnowledgePoint(id="gauss-law", name="Gauss's law"),
        ),
        edges=(PrerequisiteEdge(prerequisite_id="electric-field", target_id="gauss-law"),),
    )
    SQLitePrerequisiteGraphRepository(curriculum_path).save(graph)
    result = SimpleNamespace(
        final_answer="Gauss's law relates electric flux to enclosed charge.",
        claims=[],
        domain="physics",
        student=SimpleNamespace(known_topics=set(), weak_topics=set()),
    )
    artifacts = persist_runtime_epistemic_artifacts(
        username="alice",
        task_id="task-456",
        question="Please explain Gauss's law and include the assumptions.",
        result=result,
        data_root=tmp_path / "data",
        max_profiles=10,
        curriculum_db_path=curriculum_path,
    )
    assert artifacts.prerequisite_diagnostic is not None
    assert artifacts.prerequisite_diagnostic.graph_version == 1
    assert artifacts.prerequisite_diagnostic.gaps[0].prerequisite_point_id == "electric-field"
    assert artifacts.prerequisite_diagnostic.gaps[0].learner_stage == "unknown"
    assert "尚未评估" in artifacts.prerequisite_note
    assert "一定不会" in artifacts.prerequisite_note

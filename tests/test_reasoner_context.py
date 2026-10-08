import json
from types import SimpleNamespace

from core.reasoner import AgentReasoner
from core.state import AgentState, AgentStep, StudentState


class FakeKnowledgeGraph:
    def context_for(self, question):
        return {
            "concept": "attention",
            "relations": ["attention -> transformer"],
            "for_question": question,
        }


def test_reasoner_packs_goal_and_relevant_context_for_web_model():
    state = AgentState(
        question="解释 Transformer 的 attention",
        student=StudentState(
            known_topics={"transformer"},
            weak_topics={"softmax"},
            learning_topics={"attention"},
            misconceptions=["把 QK^T 当成最终 attention"],
        ),
        goal="准确解释 self-attention 的计算过程",
        goal_context=["之前已经学过 Q/K/V"],
        task_type="conceptual",
        domain="深度学习",
        task_analysis=SimpleNamespace(
            task_type="conceptual",
            domain="深度学习",
            goal="解释 attention",
            difficulty=3,
        ),
        knowledge_graph=FakeKnowledgeGraph(),
    )
    state.steps.append(
        AgentStep(
            step_id=1,
            action="SEARCH",
            tool="wikipedia",
            arguments={"query": "scaled dot product attention"},
            observation={"results": ["retrieval result"]},
            reasoning_summary="查定义和计算公式",
        )
    )

    prompt = AgentReasoner(None, None)._build_prompt(state, [])
    payload = json.loads(prompt)

    assert payload["question"] == state.question
    assert payload["goal"] == state.goal
    assert payload["goal_context"] == state.goal_context
    assert payload["task_analysis"]["goal"] == "解释 attention"
    assert payload["knowledge_graph"]["concept"] == "attention"
    assert payload["student_state"]["known_topics"] == ["transformer"]
    assert payload["student_state"]["weak_topics"] == ["softmax"]
    assert payload["student_state"]["learning_topics"] == ["attention"]
    assert payload["student_state"]["misconceptions"] == state.student.misconceptions
    assert payload["previous_steps"][0]["action"] == "SEARCH"
    assert payload["previous_steps"][0]["arguments"]["query"] == "scaled dot product attention"

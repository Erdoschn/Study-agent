from __future__ import annotations

import threading
from types import SimpleNamespace

from core.state import AgentState
from core.tool_loop import AgentToolLoop, ToolExecutor


def test_agent_tool_loop_stops_before_model_call_when_cancelled():
    event = threading.Event()
    event.set()
    calls = []

    class Reasoner:
        def decide(self, state):
            calls.append(state)
            raise AssertionError("reasoner should not be called after cancellation")

    state = AgentState(question="test cancellation")
    result = AgentToolLoop(
        Reasoner(),
        ToolExecutor(),
        cancellation_event=event,
    ).run(state)

    assert calls == []
    assert result.finished is True
    assert result.steps[-1].action == "STOP"
    assert "取消" in result.error


def test_agent_tool_loop_discards_decision_if_cancelled_during_model_call():
    event = threading.Event()

    class Reasoner:
        def decide(self, state):
            event.set()
            return SimpleNamespace(model="fake", effort="low")

    state = AgentState(question="test cancellation during inference")
    result = AgentToolLoop(
        Reasoner(),
        ToolExecutor(),
        cancellation_event=event,
    ).run(state)

    assert result.finished is True
    assert result.steps[-1].action == "STOP"
    assert result.final_answer is None

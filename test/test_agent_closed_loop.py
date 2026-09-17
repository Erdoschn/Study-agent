from core.state import AgentState
from core.tool_loop import AgentToolLoop
from core.reasoner import ReasoningDecision


class FakeExecutor:
    def __init__(self): self.calls = []

    def execute(self, tool, arguments):
        self.calls.append((tool, arguments))
        return [{"result": "fresh evidence"}]


class FakeReasoner:
    def __init__(self): self.n = 0

    def decide(self, state):
        self.n += 1
        if self.n == 1:
            return ReasoningDecision(action="SEARCH", reasoning_summary="需要外部证据", tool="search", arguments={"query": "attention"}, model="fake")
        return ReasoningDecision(action="ANSWER", reasoning_summary="观察结果已足够", model="fake")


def test_tool_observation_is_fed_into_next_reasoning_cycle():
    state = AgentState(question="什么是 attention", max_steps=4)
    reasoner = FakeReasoner()
    executor = FakeExecutor()
    state = AgentToolLoop(reasoner, executor).run(state)
    assert executor.calls == [("search", {"query": "attention"})]
    assert reasoner.n == 2
    assert state.evidence == [{"result": "fresh evidence"}]
    assert state.steps[0].observation == [{"result": "fresh evidence"}]
    assert state.steps[-1].action == "ANSWER"
    assert state.finished is True


def test_identical_tool_call_cannot_loop_forever():
    class RepeatReasoner:
        def decide(self, state):
            return ReasoningDecision(action="CALCULATE", reasoning_summary="重复计算", tool="calculate", arguments={"expression": "2+2"}, model="fake")

    state = AgentState(question="2+2", max_steps=8)
    state = AgentToolLoop(RepeatReasoner(), FakeExecutor()).run(state)
    assert state.finished is True
    assert state.error == "Agent 检测到重复工具调用，已停止。"
    assert state.steps[-1].action == "STOP"
    assert len(state.steps) == 2

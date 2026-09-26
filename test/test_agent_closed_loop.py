from core.state import AgentState
from core.tool_loop import AgentToolLoop
from core.reasoner import ReasoningDecision


class FakeExecutor:
    def __init__(self): self.calls = []

    def execute(self, tool, arguments):
        self.calls.append((tool, arguments))
        if tool == "search":
            return [{
                "source": "wikipedia",
                "title": "Attention",
                "abstract": "attention mechanisms",
                "identifier": "attention-1",
                "harness_relevance": "DIRECT",
            }]
        if tool == "verify":
            return {
                "claim": arguments.get("claim", ""),
                "verification_status": "MATCHED",
                "matched_evidence": [0],
            }
        return [{"result": "fresh evidence"}]


class FakeReasoner:
    def __init__(self): self.n = 0

    def decide(self, state):
        self.n += 1
        if self.n == 1:
            return ReasoningDecision(action="SEARCH", reasoning_summary="需要外部证据", tool="search", arguments={"query": "attention"}, model="fake")
        return ReasoningDecision(action="ANSWER", reasoning_summary="观察结果已足够", answer="fresh answer", model="fake")


def test_tool_observation_is_fed_into_next_reasoning_cycle():
    state = AgentState(question="什么是 attention", max_steps=4)
    reasoner = FakeReasoner()
    executor = FakeExecutor()
    state = AgentToolLoop(reasoner, executor).run(state)
    assert executor.calls == [("search", {"query": "attention"})]
    assert reasoner.n == 2
    assert state.evidence == [{
        "source": "wikipedia",
        "title": "Attention",
        "abstract": "attention mechanisms",
        "identifier": "attention-1",
        "harness_relevance": "DIRECT",
    }]
    assert state.steps[0].observation == state.evidence
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


def test_answer_requires_verification_for_current_claims():
    class Reasoner:
        def __init__(self):
            self.n = 0

        def decide(self, state):
            self.n += 1
            if self.n == 1:
                return ReasoningDecision(
                    action="SEARCH",
                    reasoning_summary="获取证据",
                    tool="search",
                    arguments={"query": "attention"},
                )
            if self.n == 2:
                return ReasoningDecision(
                    action="VERIFY",
                    reasoning_summary="核查第一个陈述",
                    tool="verify",
                    arguments={"claim": "attention uses query"},
                )
            if self.n == 3:
                return ReasoningDecision(
                    action="ANSWER",
                    reasoning_summary="尝试回答另一个未经核查的陈述",
                    answer="暂定答案",
                    claims=[{"claim": "attention uses values"}],
                )
            if self.n == 4:
                return ReasoningDecision(
                    action="VERIFY",
                    reasoning_summary="核查当前回答陈述",
                    tool="verify",
                    arguments={"claim": "attention uses values"},
                )
            return ReasoningDecision(
                action="ANSWER",
                reasoning_summary="当前陈述已有对应核查",
                answer="最终答案",
                claims=[{"claim": "attention uses values"}],
            )

    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == "search":
                return [{
                    "source": "wikipedia",
                    "title": "Attention",
                    "abstract": "attention uses query, key, and values",
                    "identifier": "attention-1",
                    "harness_relevance": "DIRECT",
                }]
            return {
                "claim": arguments["claim"],
                "verification_status": "MATCHED",
                "matched_evidence": [0],
            }

    executor = Executor()
    state = AgentToolLoop(Reasoner(), executor).run(
        AgentState(question="attention", max_steps=8)
    )

    assert executor.calls == [
        ("search", {"query": "attention"}),
        ("verify", {"claim": "attention uses query"}),
        ("verify", {"claim": "attention uses values"}),
    ]
    assert state.final_answer == "最终答案"
    assert state.finished is True

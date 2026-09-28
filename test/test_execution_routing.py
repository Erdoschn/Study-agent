from core.agent import StudyAgent
from core.task_analyzer import TaskAnalyzer
from core.reasoner import ReasoningDecision


class Model:
    name = "fake-free"


class Registry:
    def record_success(self, *args): pass
    def record_failure(self, *args, **kwargs): pass
    def is_provider_level_failure(self, exc): return False


class Router:
    registry = Registry()

    def select_candidates(self, capability, allow_paid=False, **kwargs):
        return [Model()]


class Client:
    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt, json_mode=False):
        self.calls.append((system_prompt, user_prompt, json_mode))
        if "任务分析器" in system_prompt:
            return self.analysis
        return "direct answer"


class Factory:
    def __init__(self, analysis):
        self.analysis = analysis
        self.client = Client()
        self.client.analysis = analysis

    def create(self, model):
        return self.client


class Reasoner:
    allow_paid = False

    def __init__(self, analysis):
        self.model_router = Router()
        self.model_factory = Factory(analysis)

    def decide(self, state):
        return ReasoningDecision(action="ANSWER", answer="loop draft", model="fake-free")


class Teacher:
    def __init__(self):
        self.calls = 0

    def generate(self, state, draft_answer=None):
        self.calls += 1
        return "teacher answer"


class Executor:
    def __init__(self):
        self.calls = 0

    def execute(self, tool, arguments, state=None):
        self.calls += 1
        raise AssertionError("unexpected tool execution")


def make_agent(mode, task_type="general", tools=None):
    analysis = (
        '{"task_type":"' + task_type + '","domain":"test",'
        '"required_tools":' + str(tools or []).replace("'", '"') + ','
        '"execution_mode":"' + mode + '"}'
    )
    reasoner = Reasoner(analysis)
    teacher = Teacher()
    executor = Executor()
    return StudyAgent(reasoner, teacher=teacher, tool_executor=executor), teacher, executor


def test_chat_bypasses_teacher_and_tool_loop():
    agent, teacher, executor = make_agent("chat")
    state = agent.run("今天好累啊")
    assert state.final_answer == "direct answer"
    assert state.step_count == 0
    assert teacher.calls == 0
    assert executor.calls == 0


def test_knowledge_direct_uses_teacher_without_tool_loop():
    agent, teacher, executor = make_agent("knowledge_direct", "conceptual")
    state = agent.run("什么是 Transformer？")
    assert state.final_answer == "teacher answer"
    assert state.step_count == 0
    assert teacher.calls == 1
    assert executor.calls == 0
    assert state.knowledge_graph is agent.knowledge_graph


def test_knowledge_agent_enters_harness_loop():
    agent, teacher, executor = make_agent("knowledge_agent", "research", ["search"])
    state = agent.run("研究 Transformer 的最新优化")
    assert state.step_count >= 1
    assert state.final_answer == "loop draft"
    assert teacher.calls == 1


def test_task_analyzer_unknown_mode_falls_back_by_tool_need():
    result = TaskAnalyzer._parse(
        '{"task_type":"research","required_tools":["search"],"external_facts_needed":true}'
    )
    assert result.execution_mode == "knowledge_agent"

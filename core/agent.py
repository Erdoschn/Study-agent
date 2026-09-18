from .state import AgentState
from .task_analyzer import TaskAnalyzer
from .planner import TaskPlanner
from .tool_loop import AgentToolLoop
from .__debug__ import debug


class StudyAgent:
    """LLM-driven Study Agent: Harness executes actions; LLM decides the next action."""

    def __init__(self, reasoner, teacher=None, tool_executor=None, max_steps=None):
        self.reasoner = reasoner
        self.teacher = teacher  # 保留兼容性；新的 Agent Loop 不再依赖 Teacher。
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.task_analyzer = TaskAnalyzer(reasoner)
        self.planner = TaskPlanner()

    def run(self, question: str, student_state=None) -> AgentState:
        with debug.scope("StudyAgent", "RUN"):
            debug.log("StudyAgent", f"QUESTION → {question}")
            state = AgentState(question=question, max_steps=self.max_steps)
            if student_state is not None:
                state.student = student_state

            try:
                state.task_analysis = self.task_analyzer.analyze(question, state.student)
                state.goal = state.task_analysis.goal
                state.task_type = state.task_analysis.task_type
                state.domain = state.task_analysis.domain
                state.search_sources = list(getattr(state.task_analysis, "search_sources", []))
                state.search_sort_by = getattr(state.task_analysis, "search_sort_by", "relevance")
                state.plan = self.planner.create(state.task_analysis)
                debug.log("StudyAgent", f"ANALYSIS → {state.task_type} / {state.domain}")
                debug.log(
                    "StudyAgent",
                    f"SEARCH STRATEGY → sources={state.search_sources or ['router default']} sort={state.search_sort_by}",
                )
            except Exception as exc:
                state.error = f"Task Analysis 执行失败：{type(exc).__name__}: {exc}"
                debug.log("StudyAgent", state.error)

            if not state.error:
                loop = AgentToolLoop(self.reasoner, self.tool_executor)
                state = loop.run(state)

            debug.log("StudyAgent", f"LOOP FINISHED → steps={state.step_count}")

            if state.final_answer is None and not state.error:
                state.error = "Agent 在没有产生最终 ANSWER 的情况下结束。"

            if state.final_answer is None and state.error:
                state.final_answer = f"Agent 未能完成任务。\n\n原因：{state.error}"

            self._update_student_model(state)
            debug.log("StudyAgent", "STUDENT MODEL → updated")
            return state

    def _update_student_model(self, state: AgentState) -> None:
        if state.domain:
            state.student.known_topics.add(state.domain)

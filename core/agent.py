from .state import AgentState
from .tool_loop import AgentToolLoop
from .__debug__ import debug


class StudyAgent:
    """LLM-driven Study Agent: Harness initializes state; the LLM owns the decision loop."""

    def __init__(self, reasoner, teacher=None, tool_executor=None, max_steps=None):
        self.reasoner = reasoner
        self.teacher = teacher  # 兼容旧接口；不再作为固定流程节点。
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.student_state = None

    def run(self, question: str, student_state=None) -> AgentState:
        with debug.scope("StudyAgent", "RUN"):
            debug.log("StudyAgent", f"QUESTION → {question}")
            state = AgentState(question=question, max_steps=self.max_steps)
            if student_state is not None:
                self.student_state = student_state
            elif self.student_state is None:
                self.student_state = state.student
            state.student = self.student_state

            # 不再先调用 TaskAnalyzer/Planner。
            # Agent Brain 在第一轮直接观察用户任务 + 学生状态 + 工具，
            # 自己决定是否需要分析、搜索、计算、验证或直接回答。
            state.goal = "解决用户当前问题，并在需要时获取足够可靠的证据。"
            state.search_sources = []
            state.search_sort_by = "relevance"

            if self.tool_executor is None:
                state.error = "Tool Harness 尚未配置。"
            else:
                try:
                    state = AgentToolLoop(self.reasoner, self.tool_executor).run(state)
                except Exception as exc:
                    state.error = f"Agent Loop 执行失败：{type(exc).__name__}: {exc}"
                    debug.log("StudyAgent", state.error)

            # 兼容旧的 Teacher/TaskAnalyzer 测试与集成适配器；正式 Harness 不走这里。
            if not hasattr(self.tool_executor, "execute") and self.teacher is not None:
                try:
                    from .task_analyzer import TaskAnalyzer
                    from .planner import TaskPlanner
                    from .state import AgentStep
                    analyzer = TaskAnalyzer(self.reasoner)
                    state.task_analysis = analyzer.analyze(question, state.student)
                    state.task_type = state.task_analysis.task_type
                    state.domain = state.task_analysis.domain
                    state.goal = state.task_analysis.goal or state.goal
                    state.search_sources = state.task_analysis.search_sources
                    state.search_sort_by = state.task_analysis.search_sort_by
                    state.plan = TaskPlanner().create(state.task_analysis)
                    state.final_answer = self.teacher.generate(state)
                    state.add_step(AgentStep(
                        step_id=1, action="ANSWER",
                        reasoning_summary="兼容旧式 Teacher 流程。", success=True,
                    ))
                    state.finished = True
                    state.error = None
                except Exception as exc:
                    state.error = f"兼容旧流程失败：{type(exc).__name__}: {exc}"
                    debug.log("StudyAgent", state.error)

            debug.log("StudyAgent", f"LOOP FINISHED → steps={state.step_count}")

            if state.final_answer is None and state.error:
                state.final_answer = f"Agent 未能完成任务。\n\n原因：{state.error}"
            elif state.final_answer is None:
                state.error = "Agent 在没有产生最终 ANSWER 的情况下结束。"
                state.final_answer = f"Agent 未能完成任务。\n\n原因：{state.error}"

            self._update_student_model(state)
            debug.log("StudyAgent", "STUDENT MODEL → updated")
            return state

    def _update_student_model(self, state: AgentState) -> None:
        if state.domain:
            state.student.known_topics.add(state.domain)

from .state import AgentState
from .task_analyzer import TaskAnalyzer
from .planner import TaskPlanner
from .tool_loop import AgentToolLoop
from .__debug__ import debug


class StudyAgent:
    """Study Agent: understand → plan → reason/act/observe → teach → update."""

    def __init__(self, reasoner, teacher, tool_executor, max_steps: int = 8):
        self.reasoner = reasoner
        self.teacher = teacher
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
                state.plan = self.planner.create(state.task_analysis)
                debug.log("StudyAgent", f"ANALYSIS → {state.task_type} / {state.domain}")
                debug.log("StudyAgent", f"PLAN → {' → '.join(state.plan.actions)}")
            except Exception as exc:
                state.error = f"Task Analysis 执行失败：{type(exc).__name__}: {exc}"
                debug.log("StudyAgent", state.error)

            if not state.error:
                loop = AgentToolLoop(self.reasoner, self.tool_executor)
                state = loop.run(state)

            debug.log("StudyAgent", f"LOOP FINISHED → steps={state.step_count}")

            if state.final_answer is None:
                if state.error:
                    state.final_answer = f"Agent 未能完成任务。\n\n原因：{state.error}"
                else:
                    debug.log("StudyAgent", "TEACHER → generate")
                    try:
                        state.final_answer = self.teacher.generate(state)
                    except Exception as exc:
                        state.error = f"Teacher 执行失败：{type(exc).__name__}: {exc}"
                        state.final_answer = f"最终教学回答生成失败。\n\n{state.error}"

            self._update_student_model(state)
            debug.log("StudyAgent", "STUDENT MODEL → updated")
            return state

    def _update_student_model(self, state: AgentState) -> None:
        """Only record the analyzed domain for now; mastery is not inferred from one turn."""
        if state.domain:
            state.student.known_topics.add(state.domain)

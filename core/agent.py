from .state import AgentState
from .tool_loop import AgentToolLoop
from .__debug__ import debug

class StudyAgent:
    """
    最终 Study Agent。

    工作流：

    Understand
        ↓
    Reason
        ↓
    Act
        ↓
    Observe
        ↓
    Re-Reason
        ↓
    Verify
        ↓
    Teach
        ↓
    Update Student Model
    """

    def __init__(
        self,
        reasoner,
        teacher,
        tool_executor,
        max_steps: int = 8,
    ):
        self.reasoner = reasoner
        self.teacher = teacher
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    def run(
        self,
        question: str,
        student_state=None,
    ) -> AgentState:

        with debug.scope(
            "StudyAgent",
            "RUN",
        ):

            debug.log(
                "StudyAgent",
                f"QUESTION → {question}",
            )

            state = AgentState(
                question=question,
                max_steps=self.max_steps,
            )

            if student_state is not None:
                state.student = student_state

            debug.log(
                "StudyAgent",
                "START → AgentToolLoop",
            )

            loop = AgentToolLoop(
                self.reasoner,
                self.tool_executor,
            )

            state = loop.run(
                state
            )

            debug.log(
                "StudyAgent",
                f"LOOP FINISHED → "
                f"steps={state.step_count}",
            )

            if state.final_answer is None:

                if state.error:

                    state.final_answer = (
                        "Agent 未能完成任务。\n\n"
                        f"原因：{state.error}"
                    )

                else:

                    debug.log(
                        "StudyAgent",
                        "TEACHER → generate",
                    )

                    try:

                        state.final_answer = (
                            self.teacher.generate(
                                state
                            )
                        )

                    except Exception as exc:

                        state.error = (
                            f"Teacher 执行失败："
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                        state.final_answer = (
                            "最终教学回答生成失败。\n\n"
                            f"{state.error}"
                        )

            self._update_student_model(
                state
            )

            debug.log(
                "StudyAgent",
                "STUDENT MODEL → updated",
            )

            return state

    def _update_student_model(
        self,
        state: AgentState,
    ) -> None:
        """
        当前先做保守更新。

        不因为一次回答就永久认定学生“掌握/不会”。
        """
        if state.domain:
            state.student.known_topics.add(
                state.domain
            )
import json
from .__debug__ import debug

class Teacher:
    """
    最终教学回答生成器。

    每次生成回答时：
        ModelRouter
            ↓
        动态选择模型
    """

    SYSTEM_PROMPT = """
你是 Study Agent 的教学引擎。

根据：
- 用户问题
- Agent 目标
- 学生状态
- Agent 执行过程
- 搜索证据
- Claim

生成最终教学回答。

要求：

1. 直接回答问题。
2. 发现学生理解错误时明确指出。
3. 区分外部事实、推导、实验和推断。
4. 外部资料尽量给出来源链接。
5. 无证据支持时不要伪装成事实。
6. 根据学生状态调整解释深度。
7. 不输出隐藏思维链。
8. 可以简洁说明 Agent 为什么进行了某些操作。
"""

    def __init__(
        self,
        model_router,
        model_factory,
        allow_paid: bool = False,
    ):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def generate(
        self,
        state,
    ) -> str:

        payload = {
            "question": state.question,
            "goal": state.goal,
            "task_type": state.task_type,
            "domain": state.domain,
            "student": {
                "known_topics": list(
                    state.student.known_topics
                ),
                "weak_topics": list(
                    state.student.weak_topics
                ),
                "misconceptions": (
                    state.student.misconceptions
                ),
            },
            "steps": [
                {
                    "step": step.step_id,
                    "action": step.action,
                    "model": step.model,
                    "tool": step.tool,
                    "reasoning_summary": (
                        step.reasoning_summary
                    ),
                    "observation": (
                        step.observation
                    ),
                    "success": step.success,
                    "error": step.error,
                }
                for step in state.steps
            ],
            "evidence": state.evidence,
            "claims": state.claims,
        }

        prompt = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

        debug.log(
            "Teacher",
            "SELECT MODEL → teaching",
        )
        candidates = (
            self.model_router.select_candidates(
                capability="teaching",
                allow_paid=self.allow_paid,
            )
        )

        if not candidates:
            raise RuntimeError(
                "没有可用于 Teacher 的模型。"
            )

        errors = []

        for model in candidates:

            try:
                client = (
                    self.model_factory.create(
                        model
                    )
                )

                debug.log(
                    "Teacher",
                    f"CALL LLM → {model.name}",
                )

                result = client.generate(
                    self.SYSTEM_PROMPT,
                    prompt,
                )

                debug.log(
                    "Teacher",
                    f"SUCCESS → {model.name}",
                )

                self.model_router.registry.record_success(
                    model.name
                )

                return result

            except Exception as exc:

                self.model_router.registry.record_failure(
                    model.name
                )

                errors.append(
                    f"{model.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        raise RuntimeError(
            "所有 Teacher 候选模型均调用失败：\n"
            + "\n".join(errors)
        )
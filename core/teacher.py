import json
from .__debug__ import debug


class Teacher:
    """最终教学回答生成器。"""

    SYSTEM_PROMPT = """
你是 Study Agent 的教学引擎。
根据用户问题、任务分析、目标、学生状态、Agent 执行过程、搜索证据和 Claim 生成最终教学回答。
1. 直接回答问题；发现理解错误时明确指出。
2. 区分外部事实、推导、实验和推断。
3. 外部资料尽量给出来源链接；无证据支持时不要伪装成事实。
4. 搜索候选的 relevance 只是定性筛选意见，不是事实真伪证明；优先使用 DIRECT，其次根据需要使用 PARTIAL，谨慎使用 UNCERTAIN，通常不要使用 IRRELEVANT。
5. “最新”与“最相关”是两个独立维度，不要因为某结果较新就把它当成更相关。
6. 根据学生状态调整解释深度。
7. 不输出隐藏思维链；可以简洁说明 Agent 为什么进行了某些操作。
"""

    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def generate(self, state) -> str:
        analysis = state.task_analysis
        payload = {
            "question": state.question,
            "task_analysis": analysis.__dict__ if analysis else None,
            "goal": state.goal, "task_type": state.task_type, "domain": state.domain,
            "student": {"known_topics": list(state.student.known_topics), "weak_topics": list(state.student.weak_topics), "misconceptions": state.student.misconceptions},
            "steps": [{"step": s.step_id, "action": s.action, "model": s.model, "tool": s.tool, "reasoning_summary": s.reasoning_summary, "observation": s.observation, "success": s.success, "error": s.error} for s in state.steps],
            "evidence": state.evidence,
            "evidence_relevance": state.evidence_relevance,
            "claims": state.claims,
        }
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        candidates = self.model_router.select_candidates(
            capability="teaching", allow_paid=self.allow_paid,
            task_analysis=analysis, plan=state.plan,
        )
        if not candidates:
            raise RuntimeError("没有可用于 Teacher 的模型。")
        errors = []
        for model in candidates:
            try:
                debug.log("Teacher", f"CALL LLM → {model.name}")
                result = self.model_factory.create(model).generate(self.SYSTEM_PROMPT, prompt)
                self.model_router.registry.record_success(model.name, "teaching")
                debug.log("Teacher", f"SUCCESS → {model.name}")
                return result
            except Exception as exc:
                self.model_router.registry.record_failure(model.name, "teaching")
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("所有 Teacher 候选模型均调用失败：\n" + "\n".join(errors))

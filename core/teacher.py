import json
import inspect
from .__debug__ import debug


class Teacher:
    """教学生成器：把 Agent 的事实、证据和草稿答案转化为面向学生的教学响应。"""

    SYSTEM_PROMPT = """
你是 Study Agent 的 Teacher（教学引擎），不是单纯的答案润色器。
你的任务是根据学生当前状态、Agent 执行过程、外部证据和 Reasoner 草稿，生成真正有教学价值的最终回答。

教学原则：
1. 先判断学生当前最需要什么，再决定解释深度和结构。
2. 已知内容尽量作为起点，弱项和明确误解优先处理；不要重复无关基础。
3. 对概念题优先建立直觉，再给形式化定义、公式或代码。
4. 对数学题可以保留关键一步让学生自己判断，但不要为了互动而故意省略必要结论。
5. 对代码题指出具体错误位置、原因和修改方式，并解释背后的原理。
6. 有明确误解时先纠错，再解释正确模型；不要只给最终结论。
7. 搜索证据只是外部依据。relevance 是 Harness 的定性筛选，不是真伪概率；优先 DIRECT，其次按需使用 PARTIAL，谨慎使用 UNCERTAIN，通常不使用 IRRELEVANT。
8. “最新”和“最相关”是独立维度；不要因为资料更新就默认更相关。
9. 需要外部事实时引用 observation 中的来源；没有证据支持时明确说明不确定性。
10. Reasoner 的 draft_answer 只是草稿，不是必须照抄的答案。可以重组、补充、删减或纠正。
11. 不输出隐藏思维链；可以简洁说明为什么采用某种教学方式。
12. 学生模型只是可修正的工作假设，不是心理事实；不得推测隐私、人格或其他心理事实。
13. 不要把 Reasoner 草稿扩写成新的未经证据支持的事实。新增事实只能来自 evidence / verified claims；教学类例子必须明确标为示例或假设。
13. 如果问题适合互动，可在回答中加入一个很小的检查问题；不要为了“完整”一次性堆满知识。

输出只需要最终教学回答，不要输出 JSON，不要输出“作为 AI”之类的套话。
"""

    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    @staticmethod
    def _derive_strategy(state) -> dict:
        student = getattr(state, "student", None)
        known_topics = set(getattr(student, "known_topics", set()) or set()) if student else set()
        weak_topics = set(getattr(student, "weak_topics", set()) or set()) if student else set()
        misconceptions = list(getattr(student, "misconceptions", []) or []) if student else []
        task_type = str(getattr(state, "task_type", "") or "").lower()

        if misconceptions:
            mode = "纠错优先"
            focus = "先定位并纠正已明确记录的概念误解，再建立正确理解。"
        elif weak_topics:
            mode = "脚手架教学"
            focus = "围绕弱项拆成较小步骤，避免一次跳到最终结论。"
        elif known_topics:
            mode = "建立在已有知识上"
            focus = "以学生已有知识为入口，减少重复基础。"
        else:
            mode = "基础引导"
            focus = "从直觉、最小例子和核心概念开始。"

        interaction = "必要时加入一个简短检查问题。"
        if task_type in {"math", "coding"}:
            interaction = "在数学或代码问题中，可以保留一个关键中间步骤供学生判断。"

        strategy = {
            "mode": mode,
            "focus": focus,
            "interaction": interaction,
        }
        debug.log(
            "Teacher",
            f"STRATEGY → mode={mode}, known={len(known_topics)}, weak={len(weak_topics)}, misconceptions={len(misconceptions)}",
        )
        return strategy

    @staticmethod
    def _verified_claims(state):
        claims = []
        for step in state.steps:
            if (
                step.action == "VERIFY"
                and step.success
                and isinstance(step.observation, dict)
                and step.observation.get("verification_status") == "MATCHED"
            ):
                claim = str(step.observation.get("claim", "")).strip()
                if claim and claim not in claims:
                    claims.append(claim)
        return claims[:12]

    @staticmethod
    def _compact_observation(observation):
        if hasattr(observation, "get") and hasattr(observation, "coverage"):
            results = []
            for item in (observation.get("results", []) or [])[:8]:
                if not isinstance(item, dict):
                    continue
                results.append({
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "identifier": item.get("identifier"),
                    "abstract": str(item.get("abstract", ""))[:500],
                    "harness_relevance": item.get("harness_relevance", "UNCERTAIN"),
                    "harness_recency": item.get("harness_recency", "UNKNOWN"),
                })
            return {
                "results": results,
                "coverage": observation.get("coverage", {}),
            }
        if isinstance(observation, dict):
            result = dict(observation)
            for key, value in list(result.items()):
                if isinstance(value, str):
                    result[key] = value[:1000]
            return result
        if isinstance(observation, str):
            return observation[:1500]
        return observation

    @classmethod
    def _build_payload(cls, state, draft_answer: str | None = None) -> dict:
        strategy = cls._derive_strategy(state)
        analysis = state.task_analysis
        recent_steps = state.steps[-8:]
        evidence = []
        for item in state.evidence[-20:]:
            if not isinstance(item, dict):
                continue
            evidence.append({
                "source": item.get("source"),
                "title": item.get("title"),
                "url": item.get("url"),
                "identifier": item.get("identifier"),
                "abstract": str(item.get("abstract", ""))[:600],
                "published": item.get("published"),
                "updated": item.get("updated"),
                "harness_relevance": item.get("harness_relevance", "UNCERTAIN"),
                "harness_recency": item.get("harness_recency", "UNKNOWN"),
            })
        steps = [
            {
                "step": s.step_id,
                "action": s.action,
                "model": s.model,
                "tool": s.tool,
                "reasoning_summary": s.reasoning_summary,
                "observation": (
                    str(s.observation)[:1500]
                    if isinstance(s.observation, str)
                    else s.observation
                ),
                "success": s.success,
                "error": s.error,
            }
            for s in recent_steps
        ]
        payload = {
            "question": state.question,
            "draft_answer": draft_answer if draft_answer is not None else state.final_answer,
            "task_analysis": analysis.__dict__ if analysis else None,
            "goal": state.goal,
            "task_type": state.task_type,
            "domain": state.domain,
            "teaching_strategy": strategy,
            "knowledge_graph": state.knowledge_graph.context_for(state.question) if state.knowledge_graph is not None else {},
            "student": {
                "known_topics": sorted(state.student.known_topics),
                "weak_topics": sorted(state.student.weak_topics),
                "misconceptions": list(state.student.misconceptions),
                "mind_bdi": state.student.mind.as_dict(),
            },
            "steps": steps,
            "evidence": evidence,
            "evidence_relevance": list(state.evidence_relevance)[-20:],
            "claims": list(state.claims)[-12:],
            "verified_claims": cls._verified_claims(state),
        }
        debug.log(
            "Teacher",
            f"PROMPT DATA → steps={len(steps)}/{len(state.steps)}, evidence={len(evidence)}/{len(state.evidence)}, claims={len(payload['claims'])}",
        )
        return payload

    @classmethod
    def _build_prompt(cls, state, draft_answer: str | None = None) -> str:
        prompt = json.dumps(cls._build_payload(state, draft_answer), ensure_ascii=False, indent=2)
        debug.log("Teacher", f"PROMPT → chars={len(prompt)}")
        return prompt

    def _call_model(self, model, state, prompt) -> str:
        debug.log("Teacher", f"CALL LLM → {model.name}")
        result = self.model_factory.create(model).generate(self.SYSTEM_PROMPT, prompt)
        self.model_router.registry.record_success(model.name, "teaching")
        debug.log("Teacher", f"SUCCESS → {model.name}")
        return result

    def generate(self, state, draft_answer: str | None = None) -> str:
        """根据学生状态和 Reasoner 草稿生成最终教学回答。"""
        prompt = self._build_prompt(state, draft_answer)
        analysis = state.task_analysis
        candidates = self.model_router.select_candidates(
            capability="teaching",
            allow_paid=self.allow_paid,
            task_analysis=analysis,
            plan=state.plan,
        )
        if not candidates:
            raise RuntimeError("没有可用于 Teacher 的模型。")

        errors = []
        for model in candidates:
            try:
                return self._call_model(model, state, prompt)
            except Exception as exc:
                self.model_router.registry.record_failure(model.name, "teaching")
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
                debug.log("Teacher", f"MODEL FAILED → {model.name}")
        raise RuntimeError("所有 Teacher 候选模型均调用失败：\\n" + "\\n".join(errors))

    @staticmethod
    def supports_draft_answer(teacher) -> bool:
        """兼容旧式 Teacher.generate(state) 适配器。"""
        try:
            params = inspect.signature(teacher.generate).parameters.values()
            return any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                or p.kind == inspect.Parameter.VAR_KEYWORD
                or p.name == "draft_answer"
                for p in params
            )
        except (TypeError, ValueError):
            return True

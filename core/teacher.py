import json
import inspect
from .__debug__ import debug
from .model_router import get_model_choices, call_model_with_effort
from .prompt_config import get_prompt
from .teaching_validator import TeachingValidator


class Teacher:
    """教学生成器：把 Agent 的事实、证据和草稿答案转化为面向学生的教学响应。"""

    SYSTEM_PROMPT = get_prompt("study_agent.teacher")



    def __init__(self, model_router, model_factory, allow_paid: bool = False, validate_teaching: bool = True):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid
        self.validate_teaching = bool(validate_teaching)
        self.teaching_validator = (
            TeachingValidator(model_router, model_factory, allow_paid=allow_paid)
            if self.validate_teaching else None
        )

    @staticmethod
    def _derive_strategy(state) -> dict:
        student = getattr(state, "student", None)
        known_topics = set(getattr(student, "known_topics", set()) or set()) if student else set()
        weak_topics = set(getattr(student, "weak_topics", set()) or set()) if student else set()
        learning_topics = set(getattr(student, "learning_topics", set()) or set()) if student else set()
        misconceptions = list(getattr(student, "misconceptions", []) or []) if student else []
        task_type = str(getattr(state, "task_type", "") or "").lower()

        if misconceptions:
            mode = "纠错优先"
            focus = "先定位并纠正已明确记录的概念误解，再建立正确理解。"
        elif weak_topics:
            mode = "脚手架教学"
            focus = "围绕真正薄弱项拆成较小步骤，避免一次跳到最终结论。"
        elif learning_topics:
            mode = "新知巩固"
            focus = "围绕刚接触或正在学习的知识建立连接，通过小例子和回忆巩固。"
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
            f"STRATEGY → mode={mode}, known={len(known_topics)}, weak={len(weak_topics)}, learning={len(learning_topics)}, misconceptions={len(misconceptions)}",
        )
        return strategy

    @staticmethod
    def _claim_key(claim: str) -> str:
        import re
        return re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+",
            "",
            str(claim or "").strip().lower(),
        )

    @classmethod
    def _verified_claims(cls, state):
        current = {
            cls._claim_key(item.get("claim", ""))
            for item in state.claims
            if isinstance(item, dict) and cls._claim_key(item.get("claim", ""))
        }
        claims = []
        seen = set()
        for step in state.steps:
            if (
                step.action == "VERIFY"
                and step.success
                and isinstance(step.observation, dict)
                and step.observation.get("verification_status") == "MATCHED"
            ):
                claim = str(step.observation.get("claim", "")).strip()
                key = cls._claim_key(claim)
                if key and key in current and key not in seen:
                    claims.append(claim)
                    seen.add(key)
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
                "observation": cls._compact_observation(s.observation),
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
                "learning_topics": sorted(state.student.learning_topics),
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
        # Compact JSON preserves every teaching field while reducing prompt prefill latency.
        prompt = json.dumps(cls._build_payload(state, draft_answer), ensure_ascii=False, separators=(",", ":"))
        debug.log("Teacher", f"PROMPT → chars={len(prompt)}")
        return prompt

    def _call_model(self, model, state, prompt, effort: str | None = None) -> str:
        debug.log("Teacher", f"CALL LLM → {model.name} effort={effort or 'default'}")
        result = call_model_with_effort(
            self.model_router,
            self.model_factory.create(model),
            self.SYSTEM_PROMPT,
            prompt,
            reasoning_effort=effort,
            reasoning_effort_param=getattr(model, "reasoning_effort_param", None),
        )
        self.model_router.registry.record_success(model.name, "teaching")
        debug.log(
            "Teacher",
            f"SUCCESS → {model.name}, output_chars={len(str(result or ''))}",
        )
        return result

    @classmethod
    def _build_revision_prompt(cls, state, draft_answer: str, findings: list[dict]) -> str:
        payload = {
            "question": state.question,
            "task_type": state.task_type,
            "domain": state.domain,
            "task_analysis": state.task_analysis.__dict__ if state.task_analysis else None,
            "draft_answer": draft_answer,
            "validator_findings": findings,
            "revision_rules": [
                "只修复 validator 明确指出的 major factual or mathematical errors。",
                "保留原答案的高知识密度、教学结构和有价值的正确内容。",
                "不要因为风格原因大幅重写。",
                "修复时补上必要条件、定义域、量词或概念边界。",
                "不要把直观解释写成严格定义。",
                "只输出修订后的最终教学回答，不解释校验过程。",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    def generate(self, state, draft_answer: str | None = None) -> str:
        """根据学生状态和 Reasoner 草稿生成最终教学回答。"""
        prompt = self._build_prompt(state, draft_answer)
        analysis = state.task_analysis
        choices = get_model_choices(
            self.model_router,
            "teaching",
            allow_paid=self.allow_paid,
            task_analysis=analysis,
            plan=state.plan,
        )
        if not choices:
            raise RuntimeError("没有可用于 Teacher 的模型。")

        errors = []
        attempts = 0
        difficulty = getattr(analysis, "difficulty", 3) or 3
        for choice in choices:
            model = choice.model
            attempts += 1
            try:
                result = self._call_model(model, state, prompt, choice.effort)
                if hasattr(self.model_router.registry, "record_task_outcome"):
                    self.model_router.registry.record_task_outcome(
                        model.name,
                        "teaching",
                        difficulty,
                        attempts,
                        True,
                    )

                state.metrics["teaching_validation_enabled"] = bool(self.validate_teaching)
                if self.teaching_validator is None:
                    state.metrics["teaching_validation_status"] = "DISABLED"
                    return result

                validation = self.teaching_validator.validate(state, result)
                state.metrics["teaching_validation_status"] = validation.get("status", "UNCERTAIN")
                state.metrics["teaching_validation_model"] = validation.get("model")
                state.metrics["teaching_validation_effort"] = validation.get("effort")
                findings = self.teaching_validator.revision_findings(validation)
                state.metrics["teaching_validation_major_errors"] = len(findings)

                if not findings:
                    return result

                debug.log(
                    "Teacher",
                    f"VALIDATION → major_errors={len(findings)}, revising with {model.name}",
                )
                revision_prompt = self._build_revision_prompt(state, result, findings)
                try:
                    revised = self._call_model(model, state, revision_prompt, choice.effort)
                    revised = str(revised or "").strip()
                    if revised:
                        state.metrics["teaching_revision"] = True
                        return revised
                except Exception as revision_exc:
                    state.metrics["teaching_revision"] = False
                    debug.log(
                        "Teacher",
                        f"REVISION FAILED → kept draft: {type(revision_exc).__name__}",
                    )
                return result
            except Exception as exc:
                registry = self.model_router.registry
                provider_level = (
                    registry.is_provider_level_failure(exc)
                    if hasattr(registry, "is_provider_level_failure")
                    else False
                )
                try:
                    registry.record_failure(
                        model.name, "teaching",
                        provider_level=provider_level,
                    )
                except TypeError as record_exc:
                    # Keep compatibility with lightweight legacy registries used by integrations/tests.
                    if "provider_level" not in str(record_exc):
                        raise
                    registry.record_failure(model.name, "teaching")
                if hasattr(self.model_router.registry, "record_task_outcome"):
                    self.model_router.registry.record_task_outcome(
                        model.name,
                        "teaching",
                        difficulty,
                        attempts,
                        False,
                    )
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

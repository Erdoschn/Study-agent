import json

from .__debug__ import debug
from .model_router import get_model_choices, call_model_with_effort


class TeachingValidator:
    """对 Teacher 草稿做 claim-level 学术/数学一致性检查。"""

    SYSTEM_PROMPT = """
你是 Study Agent 的 Teaching Validator（学术严谨性校验器）。
你的任务不是重写答案，而是严格检查 Teacher 已经生成的教学答案。

重点检查：
1. 定义是否准确，是否遗漏必要条件。
2. 数学公式、符号、等式方向和适用条件是否正确。
3. 是否把相关概念错误压缩成相同概念。
4. 是否把“通常”“可以理解为”“近似”等表达错误写成严格等价。
5. 事实陈述是否与提供的 evidence / verified claims 冲突。
6. 例子是否被误写成普遍规律。
7. 是否存在明显的概念边界错误、因果倒置或必要/充分条件混淆。

严格规则：
- 不要因为措辞偏好、风格差异或解释不够漂亮而判错。
- 如果无法确认，使用 UNCERTAIN，而不是猜测。
- 只有实质性的知识/数学错误才判 ERROR。
- 高知识密度是目标，不要要求答案为了通俗而删除必要细节。
- 对“直观解释”和“严格定义”必须分别判断，不能因为直观解释简化就直接判错。
- 不输出隐藏思维链，只输出结论和最小必要依据。

只输出 JSON：
{
  "status": "PASS" | "REVISE" | "UNCERTAIN",
  "claims": [
    {
      "claim": "需要检查的原文陈述",
      "verdict": "PASS" | "ERROR" | "UNCERTAIN",
      "severity": "major" | "minor",
      "correction": "若错误，给出最小的正确表述；否则为空字符串"
    }
  ],
  "summary": "一句话总结"
}

只有出现至少一个确定的 major ERROR 时，status 才应为 REVISE。
minor ERROR 可以记录，但不能单独触发自动修订。
""".strip()

    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    @staticmethod
    def _extract_json(text: str):
        text = str(text or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    @classmethod
    def _normalize_result(cls, data):
        if not isinstance(data, dict):
            return {"status": "UNCERTAIN", "claims": [], "summary": "Validator 没有返回可解析的结构化结果。"}
        status = str(data.get("status", "UNCERTAIN")).upper()
        if status not in {"PASS", "REVISE", "UNCERTAIN"}:
            status = "UNCERTAIN"
        claims = []
        raw_claims = data.get("claims", [])
        if isinstance(raw_claims, list):
            for item in raw_claims[:30]:
                if not isinstance(item, dict):
                    continue
                verdict = str(item.get("verdict", "UNCERTAIN")).upper()
                if verdict not in {"PASS", "ERROR", "UNCERTAIN"}:
                    verdict = "UNCERTAIN"
                severity = str(item.get("severity", "minor")).lower()
                if severity not in {"major", "minor"}:
                    severity = "minor"
                claim = str(item.get("claim", "")).strip()
                correction = str(item.get("correction", "")).strip()
                if claim:
                    claims.append({"claim": claim, "verdict": verdict, "severity": severity, "correction": correction})
        major_errors = [item for item in claims if item["verdict"] == "ERROR" and item["severity"] == "major"]
        if major_errors:
            status = "REVISE"
        elif status == "REVISE":
            status = "UNCERTAIN"
        return {"status": status, "claims": claims, "summary": str(data.get("summary", "")).strip()[:1000]}

    @classmethod
    def _build_prompt(cls, state, draft_answer: str) -> str:
        analysis = getattr(state, "task_analysis", None)
        evidence = []
        for item in list(getattr(state, "evidence", []) or [])[-20:]:
            if not isinstance(item, dict):
                continue
            evidence.append({"source": item.get("source"), "title": item.get("title"), "identifier": item.get("identifier"), "abstract": str(item.get("abstract", ""))[:700], "harness_relevance": item.get("harness_relevance", "UNCERTAIN"), "harness_recency": item.get("harness_recency", "UNKNOWN")})
        payload = {
            "question": getattr(state, "question", ""),
            "task_analysis": analysis.__dict__ if analysis else None,
            "domain": getattr(state, "domain", ""),
            "task_type": getattr(state, "task_type", ""),
            "student": {
                "known_topics": sorted(getattr(state.student, "known_topics", set()) or set()),
                "weak_topics": sorted(getattr(state.student, "weak_topics", set()) or set()),
                "learning_topics": sorted(getattr(state.student, "learning_topics", set()) or set()),
                "misconceptions": list(getattr(state.student, "misconceptions", []) or []),
            },
            "claims": list(getattr(state, "claims", []) or [])[-12:],
            "evidence": evidence,
            "draft_answer": draft_answer,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def validate(self, state, draft_answer: str) -> dict:
        choices = get_model_choices(self.model_router, "verification", allow_paid=self.allow_paid, task_analysis=getattr(state, "task_analysis", None), plan=getattr(state, "plan", None))
        if not choices:
            return {"status": "UNCERTAIN", "claims": [], "summary": "没有可用的 Validator 模型。", "attempted": False}
        prompt = self._build_prompt(state, draft_answer)
        errors = []
        attempts = 0
        for choice in choices:
            model = choice.model
            attempts += 1
            try:
                debug.log("TeachingValidator", f"CALL LLM → {model.name} effort={choice.effort or 'default'}")
                raw = call_model_with_effort(self.model_router, self.model_factory.create(model), self.SYSTEM_PROMPT, prompt, json_mode=True, reasoning_effort=choice.effort, reasoning_effort_param=getattr(model, "reasoning_effort_param", None))
                result = self._normalize_result(self._extract_json(raw))
                self.model_router.registry.record_success(model.name, "verification")
                result["attempted"] = True
                result["attempts"] = attempts
                result["model"] = model.name
                result["effort"] = choice.effort
                debug.log("TeachingValidator", f"RESULT → status={result['status']}, claims={len(result['claims'])}")
                return result
            except Exception as exc:
                registry = self.model_router.registry
                provider_level = registry.is_provider_level_failure(exc) if hasattr(registry, "is_provider_level_failure") else False
                try:
                    registry.record_failure(model.name, "verification", provider_level=provider_level)
                except TypeError:
                    registry.record_failure(model.name, "verification")
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
        return {"status": "UNCERTAIN", "claims": [], "summary": "Validator 调用失败：" + "; ".join(errors), "attempted": True, "attempts": attempts}

    @staticmethod
    def revision_findings(result: dict) -> list[dict]:
        if not isinstance(result, dict):
            return []
        return [item for item in result.get("claims", []) if isinstance(item, dict) and item.get("verdict") == "ERROR" and item.get("severity") == "major"]
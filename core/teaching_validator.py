import json

from .__debug__ import debug
from .model_router import get_model_choices, call_model_with_effort
from .prompt_config import get_prompt


class TeachingValidator:
    """对 Teacher 草稿做 claim-level 学术/数学一致性检查。"""

    SYSTEM_PROMPT = get_prompt("study_agent.teaching_validator")


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
import json

from .__debug__ import debug
from .model_router import get_model_choices, call_model_with_effort
from .prompt_config import get_prompt
from .knowledge_graph import normalize_difficulty


class AssessmentGenerator:
    """为指定知识点生成正式测评题；不会执行文件、shell 或任意代码操作。"""

    SYSTEM_PROMPT = get_prompt("study_agent.assessment")

)

    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    @staticmethod
    def _extract_json(text: str):
        text = str(text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    @classmethod
    def _normalize(cls, data, primary_concept: str, difficulty: str):
        if not isinstance(data, dict):
            raise ValueError("Assessment Generator 未返回结构化题目。")
        question = str(data.get("question", "")).strip()
        expected = str(data.get("expected_answer", "")).strip()
        if not question or not expected:
            raise ValueError("测评题必须包含 question 和 expected_answer。")
        raw_rubric = data.get("rubric", [])
        rubric = [str(x).strip() for x in raw_rubric if str(x).strip()] if isinstance(raw_rubric, list) else []
        if len(rubric) < 2:
            raise ValueError("正式测评至少需要两个独立评分点。")
        # The caller owns the assessment target; the generator may add supporting concepts but cannot retarget the primary concept.
        primary = primary_concept
        supporting = data.get("supporting_concepts", [])
        supporting = [str(x).strip() for x in supporting if str(x).strip()] if isinstance(supporting, list) else []
        normalized_level, normalized_score = normalize_difficulty(difficulty)
        return {
            "concepts": [primary] + [x for x in supporting if x != primary][:7],
            "primary_concept": primary,
            "supporting_concepts": [x for x in supporting if x != primary][:7],
            "difficulty": normalized_score,
            "difficulty_level": normalized_level,
            "question_type": str(data.get("question_type", "open_ended")),
            "question": question,
            "expected_answer": expected,
            "rubric": rubric[:8],
            "relations": [],
        }

    def generate(self, primary_concept: str, difficulty: str, learner_context: dict | None = None) -> dict:
        choices = get_model_choices(self.model_router, "assessment", allow_paid=self.allow_paid)
        if not choices:
            raise RuntimeError("没有可用于正式测评的模型。")
        payload = {
            "primary_concept": primary_concept,
            "difficulty": difficulty,
            "learner_context": learner_context or {},
        }
        prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        errors = []
        for choice in choices:
            model = choice.model
            try:
                debug.log("AssessmentGenerator", f"CALL LLM → {model.name} effort={choice.effort or 'default'}")
                raw = call_model_with_effort(
                    self.model_router,
                    self.model_factory.create(model),
                    self.SYSTEM_PROMPT,
                    prompt,
                    json_mode=True,
                    reasoning_effort=choice.effort,
                    reasoning_effort_param=getattr(model, "reasoning_effort_param", None),
                )
                assessment = self._normalize(self._extract_json(raw), primary_concept, difficulty)
                self.model_router.registry.record_success(model.name, "assessment")
                assessment["generation_model"] = model.name
                assessment["generation_effort"] = choice.effort
                return assessment
            except Exception as exc:
                registry = self.model_router.registry
                provider_level = (
                    registry.is_provider_level_failure(exc)
                    if hasattr(registry, "is_provider_level_failure")
                    else False
                )
                try:
                    registry.record_failure(
                        model.name,
                        "assessment",
                        provider_level=provider_level,
                    )
                except TypeError:
                    registry.record_failure(model.name, "assessment")
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("所有正式测评生成模型均失败：\n" + "\n".join(errors))
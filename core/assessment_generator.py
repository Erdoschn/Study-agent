import json

from .__debug__ import debug
from .model_router import get_model_choices, call_model_with_effort


class AssessmentGenerator:
    """为指定知识点生成正式测评题；不会执行文件、shell 或任意代码操作。"""

    SYSTEM_PROMPT = """
你是 Study Agent 的 Assessment Generator。
你的任务是为指定知识概念生成一道正式学习测评题。

要求：
1. 题目必须能区分学生是否真正理解目标概念，而不是只会背定义。
2. difficulty 必须与请求一致：basic / undergraduate / graduate / postgraduate / postgraduate_plus。
3. 如果目标是 postgraduate 或 postgraduate_plus，题目必须要求完整正确的高阶理解，不能只靠简单关键词作答。
4. 必须提供 expected_answer 和 rubric，rubric 是可独立检查的核心要点。
5. supporting_concepts 只能帮助理解题目，不得因此把这些概念一起认证为掌握。
6. 题目、答案和评分标准必须数学/事实严谨。
7. 不生成任何代码执行、文件操作或系统操作任务；本测试只考察知识与推理。
8. 只输出 JSON，不输出隐藏思维链。

JSON schema：
{
  "primary_concept": "主要考查概念",
  "supporting_concepts": ["辅助概念"],
  "question": "题目",
  "expected_answer": "参考答案",
  "rubric": ["核心评分点1", "核心评分点2"],
  "question_type": "open_ended"
}
""".strip()

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
        primary = str(data.get("primary_concept", "")).strip() or primary_concept
        supporting = data.get("supporting_concepts", [])
        supporting = [str(x).strip() for x in supporting if str(x).strip()] if isinstance(supporting, list) else []
        return {
            "concepts": [primary] + [x for x in supporting if x != primary][:7],
            "primary_concept": primary,
            "supporting_concepts": [x for x in supporting if x != primary][:7],
            "difficulty_level": difficulty,
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
                try:
                    registry.record_failure(model.name, "assessment", provider_level=registry.is_provider_level_failure(exc))
                except TypeError:
                    registry.record_failure(model.name, "assessment")
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("所有正式测评生成模型均失败：\n" + "\n".join(errors))